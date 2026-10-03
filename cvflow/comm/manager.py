"""CommManager: devices + receive rules (what triggers a flow) + send rules (what
is sent when a flow finishes), mirroring the receive-event / send-event model
that production vision software uses to talk to PLCs.

Template syntax for send rules and nodes (Python str.format):
  {status} {ok} {ng} {run_id} {flow} {duration_ms:.1f} {error}
  {out.width:.2f}             value published by a 'Publish Result' node
  {var.product_type}          global variable
  {node[Blob Analysis].count} any node output, by node name
"""
from __future__ import annotations

import logging
import re
import string
import threading
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from ..core import events
from ..core.engine import RunResult
from ..core.events import EventBus
from ..core.runtime import FlowRunner, Trigger, TriggerSource
from ..core.variables import GlobalVariables
from .base import CommDevice, CommError
from .modbus import ModbusTcpClientDevice, ModbusTcpServerDevice
from .serial_port import SerialDevice
from .tcp import TcpClientDevice, TcpServerDevice, UdpDevice

if TYPE_CHECKING:  # pragma: no cover
    pass

log = logging.getLogger("cvflow.comm")

DEVICE_KINDS: dict[str, type[CommDevice]] = {
    cls.kind: cls for cls in (TcpClientDevice, TcpServerDevice, UdpDevice, SerialDevice,
                               ModbusTcpClientDevice, ModbusTcpServerDevice)
}

TEXT_MATCHES = ("any", "startswith", "equals", "contains", "regex")
REGISTER_MATCHES = ("register_rising", "register_change", "register_equals")


@dataclass
class ReceiveRule:
    name: str = "trigger"
    device: str = ""                 # "" = any device
    match: str = "startswith"        # TEXT_MATCHES or REGISTER_MATCHES
    pattern: str = "TRIG"            # text pattern
    address: int = 0                 # register address for register_* matches
    value: int = 1                   # expected value for register_equals
    action: str = "trigger_flow"     # trigger_flow | set_variable
    flow: str = "main"
    variable: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReceiveRule":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SendRule:
    name: str = "result"
    device: str = ""
    flow: str = ""                   # "" = any flow
    when: str = "always"             # always | ok | ng | error
    template: str = "{status},{run_id}\\n"
    registers: list[dict] = field(default_factory=list)   # [{"address": 10, "expr": "{out.count}", "kind": "int16"}]
    enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SendRule":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------- templates
class _NS:
    """Attribute/index access over a dict, returning '' for missing keys."""

    def __init__(self, d: dict | None) -> None:
        self._d = d or {}

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        return self._d.get(k, "")

    def __getitem__(self, k):
        return self._d.get(k, "")

    def __str__(self) -> str:
        return str(self._d)

    def __format__(self, spec: str) -> str:
        return format(str(self._d), spec)


class _NodeNS:
    def __init__(self, result: RunResult | None) -> None:
        self._r = result

    def _get(self, name):
        if self._r is None:
            return _NS({})
        nr = self._r.node(name)
        return _NS(nr.outputs if nr else {})

    __getattr__ = _get
    __getitem__ = _get


class _Mapping(dict):
    def __missing__(self, key):
        return ""


def format_template(template: str, result: RunResult | None = None,
                    variables: dict[str, Any] | None = None, **extra: Any) -> str:
    """Render a send template. Escape sequences (\\n, \\r, \\t) in the template are honoured."""
    tpl = template.encode("utf-8").decode("unicode_escape") if "\\" in template else template
    m = _Mapping(extra)
    if result is not None:
        m.update({"status": result.status_text, "ok": int(result.passed), "ng": int(not result.passed),
                  "run_id": result.run_id, "flow": result.flow, "duration_ms": result.duration_ms,
                  "error": result.error, "out": _NS(result.outputs), "node": _NodeNS(result)})
    else:
        m.update({"out": _NS({}), "node": _NodeNS(None)})
    m["var"] = _NS(variables or (result.variables if result else {}))
    try:
        return string.Formatter().vformat(tpl, (), m)
    except (ValueError, AttributeError, IndexError, KeyError) as e:
        raise CommError(f"模板 {template!r} 有误：{e}") from e


def eval_register_expr(expr: str, result: RunResult | None, variables: dict | None) -> float:
    text = format_template(str(expr), result, variables).strip()
    if text.lower() in ("true", "ok"):
        return 1.0
    if text.lower() in ("false", "ng", ""):
        return 0.0
    return float(text)


# ---------------------------------------------------------------- manager
class CommManager:
    def __init__(self, bus: EventBus, variables: GlobalVariables | None = None) -> None:
        self.bus = bus
        self.variables = variables if variables is not None else GlobalVariables(bus)
        self.devices: dict[str, CommDevice] = {}
        self.receive_rules: list[ReceiveRule] = []
        self.send_rules: list[SendRule] = []
        self.runners: dict[str, FlowRunner] = {}
        self._lock = threading.RLock()
        self.bus.subscribe(events.RUN_FINISHED, self._on_run_finished)

    # ---- flows ----
    def set_runners(self, runners: dict[str, FlowRunner]) -> None:
        self.runners = dict(runners)

    # ---- devices ----
    def add_device(self, name: str, kind: str, config: dict | None = None) -> CommDevice:
        if kind not in DEVICE_KINDS:
            raise CommError(f"未知的设备类型 {kind!r}，可用：{sorted(DEVICE_KINDS)}")
        with self._lock:
            old = self.devices.pop(name, None)
            if old is not None:
                old.disconnect()
            dev = DEVICE_KINDS[kind](name, config, self.bus)
            dev.on_receive(self._on_rx)
            if hasattr(dev, "on_register_change"):
                dev.on_register_change(self._on_register)
            self.devices[name] = dev
            return dev

    def remove_device(self, name: str) -> None:
        with self._lock:
            dev = self.devices.pop(name, None)
        if dev is not None:
            dev.disconnect()

    def get(self, name: str) -> CommDevice | None:
        return self.devices.get(name)

    def connect_all(self) -> list[str]:
        errors = []
        for dev in list(self.devices.values()):
            try:
                dev.connect()
            except Exception as e:
                errors.append(f"{dev.name}: {e}")
                log.error("连接 %s 失败：%s", dev.name, e)
        return errors

    def disconnect_all(self) -> None:
        for dev in list(self.devices.values()):
            try:
                dev.disconnect()
            except Exception:
                log.exception("断开 %s 失败", dev.name)

    def send(self, device: str, data: bytes | str) -> bool:
        dev = self.devices.get(device)
        if dev is None:
            log.warning("发送：未知设备 %r", device)
            return False
        return dev.send(data)

    def modbus_write(self, device: str, address: int, value: Any, kind: str = "int16") -> bool:
        dev = self.devices.get(device)
        if dev is None or not hasattr(dev, "write_value"):
            log.warning("Modbus 写入：%r 不是 Modbus 设备", device)
            return False
        try:
            dev.write_value(int(address), value, kind)
            return True
        except Exception as e:
            dev._error(f"写入失败：{e}")
            return False

    # ---- rules ----
    def add_receive_rule(self, rule: ReceiveRule) -> ReceiveRule:
        self.receive_rules.append(rule)
        return rule

    def add_send_rule(self, rule: SendRule) -> SendRule:
        self.send_rules.append(rule)
        return rule

    # ---- incoming data -> triggers ----
    def _on_rx(self, dev: CommDevice, data: bytes) -> None:
        text = dev._text(data)
        for rule in self.receive_rules:
            if not rule.enabled or rule.match not in TEXT_MATCHES:
                continue
            if rule.device and rule.device != dev.name:
                continue
            if self._text_matches(rule, text):
                self._fire(rule, Trigger(TriggerSource.COMM, device=dev.name, message=text,
                                         payload={"rule": rule.name, "raw": data}))

    @staticmethod
    def _text_matches(rule: ReceiveRule, text: str) -> bool:
        p = rule.pattern
        m = rule.match
        t = text.strip()
        if m == "any":
            return True
        if m == "startswith":
            return t.startswith(p)
        if m == "equals":
            return t == p
        if m == "contains":
            return p in t
        if m == "regex":
            try:
                return re.search(p, t) is not None
            except re.error:
                return False
        return False

    def _on_register(self, dev: CommDevice, address: int, old: int, new: int) -> None:
        for rule in self.receive_rules:
            if not rule.enabled or rule.match not in REGISTER_MATCHES:
                continue
            if (rule.device and rule.device != dev.name) or int(rule.address) != int(address):
                continue
            hit = ((rule.match == "register_rising" and old == 0 and new != 0)
                   or (rule.match == "register_change")
                   or (rule.match == "register_equals" and new == int(rule.value) and old != int(rule.value)))
            if hit:
                self._fire(rule, Trigger(TriggerSource.COMM, device=dev.name, message=f"reg[{address}]={new}",
                                         payload={"rule": rule.name, "address": address, "old": old, "new": new}))

    def _fire(self, rule: ReceiveRule, trigger: Trigger) -> None:
        if rule.action == "set_variable":
            self.variables.set(rule.variable or rule.name, trigger.message.strip())
            return
        runner = self.runners.get(rule.flow)
        if runner is None:
            log.warning("规则 %s：找不到流程 %r", rule.name, rule.flow)
            return
        if not runner.running:
            log.info("规则 %s：流程 %r 未运行，忽略触发", rule.name, rule.flow)
            return
        runner.trigger(trigger)

    # ---- run finished -> send ----
    def _on_run_finished(self, result: RunResult) -> None:
        for rule in self.send_rules:
            if not rule.enabled or (rule.flow and rule.flow != result.flow):
                continue
            if rule.when == "ok" and not result.passed:
                continue
            if rule.when == "ng" and (result.passed or not result.ok):
                continue
            if rule.when == "error" and result.ok:
                continue
            dev = self.devices.get(rule.device)
            if dev is None:
                continue
            try:
                if rule.registers and hasattr(dev, "write_value"):
                    for reg in rule.registers:
                        val = eval_register_expr(reg.get("expr", "0"), result, result.variables)
                        dev.write_value(int(reg.get("address", 0)), val, str(reg.get("kind", "int16")))
                elif rule.template:
                    dev.send(format_template(rule.template, result, result.variables))
            except Exception as e:
                dev._error(f"发送规则 {rule.name!r}：{e}")

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {"devices": [d.to_dict() for d in self.devices.values()],
                "receive_rules": [r.to_dict() for r in self.receive_rules],
                "send_rules": [r.to_dict() for r in self.send_rules]}

    def load_dict(self, cfg: dict | None) -> list[str]:
        """Replace all devices/rules from a solution's comm config. Returns error messages."""
        errors = []
        self.disconnect_all()
        with self._lock:
            self.devices.clear()
        self.receive_rules = [ReceiveRule.from_dict(r) for r in (cfg or {}).get("receive_rules", [])]
        self.send_rules = [SendRule.from_dict(r) for r in (cfg or {}).get("send_rules", [])]
        for d in (cfg or {}).get("devices", []):
            try:
                self.add_device(d["name"], d["kind"], d.get("config"))
            except Exception as e:
                errors.append(f"{d.get('name')}: {e}")
        return errors

    def shutdown(self) -> None:
        self.disconnect_all()
        self.bus.unsubscribe(events.RUN_FINISHED, self._on_run_finished)


_current: CommManager | None = None


def set_manager(mgr: CommManager | None) -> None:
    global _current
    _current = mgr


def get_manager() -> CommManager | None:
    return _current

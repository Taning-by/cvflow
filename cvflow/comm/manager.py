"""CommManager：设备 + 接收规则（什么触发流程）+ 发送规则（结果怎么回去）+ 检测握手。

一个方案里只有一个管理器，所有连接由它持有——**流程节点不自己建连接**，而是通过
``get_manager()`` 拿到它再按名字找设备。这保证了同一台设备可以被多条流程引用，也不会出现
两段代码抢读同一个套接字。

接收路径（后台线程）
    设备收到字节 → Framer 切出完整报文 → ``_on_rx``
      → 逐条匹配接收规则（文本匹配 / 正则）
      → 需要时按解析规则取出字段（请求编号、产品型号…）
      → 有握手就走握手（Ready/Busy 判断、去重、冻结请求编号与来源对端）
      → 没握手就按忙时策略直接入队
    寄存器/数据点变化 → ``_on_point``（上升沿、下降沿、变化、数值条件）

发送路径（流程结束，事件总线回调）
    ``_on_run_finished`` → 先执行发送规则（写测量值 / 发报文）→ 再让握手写 Done 与完成编号

模板语法（文本模板与发送规则共用，基于 ``str.format`` 的受限命名空间，不做表达式求值）：
  {status} {ok} {ng} {run_id} {flow} {duration_ms:.1f} {error} {request_id}
  {out.width:.2f}             流程里"发布结果"节点发布的值
  {var.product_type}          全局变量
  {field.request_id}          本次请求解析出来的字段
  {node[Blob Analysis].count} 任意节点的任意输出端口
"""
from __future__ import annotations

import csv
import logging
import re
import string
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core import events
from ..core.engine import RunResult
from ..core.events import EventBus
from ..core.runtime import FlowRunner, Trigger, TriggerSource
from ..core.variables import GlobalVariables
from .base import CommDevice, CommError, Peer
from .framing import to_hex
from .handshake import (ACCEPTED, BUSY_POLICIES, ERROR_CODES, ERROR_TEXT, Handshake,
                        HandshakeConfig, REJECTED_BAD_PARAMS)
from .mc import McDevice
from .modbus import (ModbusRtuClientDevice, ModbusTcpClientDevice, ModbusTcpServerDevice,
                     _PointHost)
from .modbus_map import DataPoint
from .parsing import FormatRule, ParseError, ParseRule
from .s7 import S7Device
from .serial_port import SerialDevice
from .tcp import TcpClientDevice, TcpServerDevice, UdpDevice

log = logging.getLogger("cvflow.comm")

DEVICE_KINDS: dict[str, type[CommDevice]] = {
    cls.kind: cls for cls in (TcpClientDevice, TcpServerDevice, UdpDevice, SerialDevice,
                              ModbusTcpClientDevice, ModbusTcpServerDevice, ModbusRtuClientDevice,
                              McDevice, S7Device)
}

DEVICE_KIND_LABELS = {
    "tcp_client": "TCP 客户端", "tcp_server": "TCP 服务端", "udp": "UDP", "serial": "串口",
    "modbus_tcp_client": "Modbus TCP 主站（读写对端）",
    "modbus_tcp_server": "Modbus TCP 从站（对端读写本机）",
    "modbus_rtu_client": "Modbus RTU 主站（串口）",
    "mc": "三菱 MC 协议（3E 二进制）", "s7": "西门子 S7（snap7）",
}

#: 协议的实现状态。界面按这个分组显示，**未实现的角色不显示为可用**。
DEVICE_STATUS = {
    "tcp_client": "ready", "tcp_server": "ready", "udp": "ready", "serial": "ready",
    "modbus_tcp_client": "ready", "modbus_tcp_server": "ready",
    "modbus_rtu_client": "ready",
    "mc": "untested", "s7": "untested",
}
DEVICE_STATUS_LABELS = {"ready": "", "untested": "（未经真机验证）"}

#: 还没实现的协议：只用于在界面上说明"没有"，不会出现在可新建的设备类型里。
PLANNED_PROTOCOLS = {
    "modbus_rtu_server": "Modbus RTU 从站（串口）",
    "modbus_ascii": "Modbus ASCII",
    "ethernet_ip": "罗克韦尔 EtherNet/IP",
    "fins": "欧姆龙 FINS",
    "opcua": "OPC UA",
    "profinet": "PROFINET",
}

TEXT_MATCHES = ("any", "startswith", "equals", "contains", "regex")
TEXT_MATCH_LABELS = {"any": "收到任何报文", "startswith": "前缀匹配", "equals": "完全相等",
                     "contains": "包含", "regex": "正则匹配"}

#: 数据点（寄存器）类触发条件。
DATA_MATCHES = ("rising", "falling", "changed", "equals", "not_equals", "greater", "less", "in_range")
DATA_MATCH_LABELS = {"rising": "上升沿（0 → 非 0）", "falling": "下降沿（非 0 → 0）",
                     "changed": "数值变化", "equals": "等于", "not_equals": "不等于",
                     "greater": "大于", "less": "小于", "in_range": "落入区间"}

#: 老方案里按裸地址写的触发条件，继续支持。
REGISTER_MATCHES = ("register_rising", "register_change", "register_equals")
_LEGACY_MATCH_MAP = {"register_rising": "rising", "register_change": "changed",
                     "register_equals": "equals"}

RULE_SOURCES = ("message", "datapoint")
RULE_SOURCE_LABELS = {"message": "报文", "datapoint": "数据点 / 寄存器"}

ACTIONS = ("trigger_flow", "set_variable")
ACTION_LABELS = {"trigger_flow": "触发流程", "set_variable": "写入全局变量"}

SEND_TARGETS = ("origin", "default", "broadcast")
SEND_TARGET_LABELS = {"origin": "回复请求来源", "default": "设备默认目标", "broadcast": "广播"}

WHEN_CHOICES = ("always", "ok", "ng", "error")
WHEN_LABELS = {"always": "每次", "ok": "仅 OK", "ng": "仅 NG", "error": "仅异常"}

LOG_DIRECTIONS = {"rx": "收", "tx": "发", "err": "错误", "sys": "状态"}


# ---------------------------------------------------------------- 规则
@dataclass
class ReceiveRule:
    """什么情况下触发流程（或写变量）。

    ``device`` 填设备 **id**（也接受设备名，老方案里就是名字）；留空表示任意设备。
    ``source`` 为 ``message`` 时用 ``match`` + ``pattern`` 匹配报文，为 ``datapoint`` 时
    用 ``match`` + ``point``/``address`` 判断数据点变化。
    """

    id: str = ""
    name: str = "trigger"
    device: str = ""
    source: str = "message"
    match: str = "startswith"
    pattern: str = "TRIG"
    point: str = ""                  # 数据点名（source == "datapoint"）
    address: int = 0                 # 裸地址（老方案 / 没配数据点时）
    value: float = 1                 # equals / greater / less / in_range 的比较值
    value2: float = 0                # in_range 的上界
    action: str = "trigger_flow"
    flow: str = "main"
    variable: str = ""
    parse: str = ""                  # 解析规则名：把报文拆成字段
    request_field: str = "request_id"  # 哪个字段是请求编号
    busy_policy: str = "queue"
    max_queue: int = 8
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.match in _LEGACY_MATCH_MAP:        # 老方案的 register_* 写法
            self.source = "datapoint"
            self.match = _LEGACY_MATCH_MAP[self.match]
        if self.source not in RULE_SOURCES:
            self.source = "message" if self.match in TEXT_MATCHES else "datapoint"
        if not self.id:
            self.id = uuid.uuid4().hex[:8]

    @property
    def point_ref(self) -> str:
        """数据点名，没配就退回裸地址的字符串形式。"""
        return self.point or str(self.address)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReceiveRule":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class SendRule:
    """流程结束时把结果发出去。

    三种载荷可以并存，按这个顺序执行：``points``（按数据点名写）→ ``registers``（老方案的
    裸地址写）→ ``format``/``template``（文本、JSON 或二进制报文）。
    """

    id: str = ""
    name: str = "result"
    device: str = ""
    flow: str = ""                   # 留空 = 任意流程
    when: str = "always"
    template: str = "{status},{run_id}\\n"
    format: str = ""                 # 格式化规则名；非空时优先于 template
    registers: list[dict] = field(default_factory=list)   # [{"address":10,"expr":"{out.n}","kind":"int16"}]
    points: list[dict] = field(default_factory=list)      # [{"point":"x","source":"out.width"}]
    target: str = "origin"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:8]
        if self.target not in SEND_TARGETS and not self.target:
            self.target = "origin"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SendRule":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------- 模板
class _NS:
    """对字典做属性/下标访问，取不到返回空串（缺字段不该让整条回复发不出去）。"""

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
    """渲染一条文本模板。模板里的 ``\\n`` ``\\r`` ``\\t`` 会被当成真正的控制字符。

    取值范围是固定的几个命名空间，没有表达式求值——模板里写不出可执行代码。
    """
    tpl = template.encode("utf-8").decode("unicode_escape") if "\\" in template else template
    m = _Mapping(extra)
    fields = extra.get("fields") if isinstance(extra.get("fields"), dict) else {}
    if result is not None:
        m.update({"status": result.status_text, "ok": int(result.passed), "ng": int(not result.passed),
                  "run_id": result.run_id, "flow": result.flow, "duration_ms": result.duration_ms,
                  "error": result.error, "out": _NS(result.outputs), "node": _NodeNS(result)})
    else:
        m.setdefault("out", _NS({}))
        m.setdefault("node", _NodeNS(None))
    m["field"] = _NS(fields)
    m["var"] = _NS(variables or (result.variables if result else {}))
    try:
        return string.Formatter().vformat(tpl, (), m)
    except (ValueError, AttributeError, IndexError, KeyError) as e:
        raise CommError(f"模板 {template!r} 有误：{e}") from e


def _address(value):
    """寄存器地址：数字（Modbus/S7 偏移）或字符串（MC 的 D100 / M20）。"""
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def eval_register_expr(expr: str, result: RunResult | None, variables: dict | None,
                       extra: dict | None = None) -> float:
    """把发送规则里的表达式渲染成一个数字。只做模板替换与数值转换，不求值表达式。"""
    text = format_template(str(expr), result, variables, **(extra or {})).strip()
    low = text.lower()
    if low in ("true", "ok"):
        return 1.0
    if low in ("false", "ng", ""):
        return 0.0
    try:
        return float(text)
    except ValueError:
        raise CommError(f"{expr!r} 渲染成 {text!r}，不是数字") from None


# ---------------------------------------------------------------- 收件箱与日志
class MessageInbox:
    """给流程节点取报文用的有界收件箱。

    后台监听线程是唯一读套接字的一方，收到的报文复制一份放进各个收件箱；节点从收件箱取，
    所以不会和监听线程抢读同一个连接。满了丢最旧的（产线上最新的那条才有意义）。
    """

    def __init__(self, name: str, device: str = "", size: int = 16) -> None:
        self.name = name
        self.device = device
        self.size = max(1, int(size))
        self._items: deque = deque(maxlen=self.size)
        self._event = threading.Event()
        self._lock = threading.Lock()
        self.dropped = 0

    def push(self, device: CommDevice, data: bytes, peer: Peer | None, text: str) -> None:
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self.dropped += 1
            self._items.append({"device": device.name, "device_id": device.id, "data": data,
                                "text": text, "peer": peer.id if peer else "", "time": time.time()})
            self._event.set()

    def get(self, timeout: float = 0.0) -> dict | None:
        end = time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                if self._items:
                    item = self._items.popleft()
                    if not self._items:
                        self._event.clear()
                    return item
                self._event.clear()
            left = end - time.monotonic()
            if left <= 0:
                return None
            self._event.wait(min(0.05, left))

    def peek(self) -> dict | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._event.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class CommLog:
    """通信调试日志：有界环形缓冲，界面按定时器批量取，不随每条报文刷新界面。"""

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = max(10, int(capacity))
        self._items: deque = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self.paused = False
        self.dropped = 0

    def add(self, direction: str, device: str, peer: str, data: bytes = b"", text: str = "",
            note: str = "") -> None:
        if self.paused:
            return
        with self._lock:
            self._seq += 1
            if len(self._items) == self._items.maxlen:
                self.dropped += 1
            self._items.append({"seq": self._seq, "time": time.time(), "dir": direction,
                                "device": device, "peer": peer, "hex": to_hex(data) if data else "",
                                "text": text, "note": note, "size": len(data)})

    def entries(self, since_seq: int = 0, limit: int = 0) -> list[dict]:
        with self._lock:
            items = [e for e in self._items if e["seq"] > since_seq]
        return items[-limit:] if limit else items

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.dropped = 0

    def export(self, path: str) -> int:
        rows = self.entries()
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["序号", "时间", "方向", "设备", "对端", "字节数", "文本", "十六进制", "说明"])
            for e in rows:
                w.writerow([e["seq"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["time"]))
                            + f".{int(e['time'] % 1 * 1000):03d}",
                            LOG_DIRECTIONS.get(e["dir"], e["dir"]), e["device"], e["peer"],
                            e["size"], e["text"], e["hex"], e["note"]])
        return len(rows)


@dataclass
class _TriggerState:
    """一条数据点规则的边沿状态。防止"触发位一直是 1"时每次轮询都重复触发。"""

    armed: bool = True
    last_value: Any = None
    last_fire: float = 0.0


# ---------------------------------------------------------------- 管理器
class CommManager:
    """一个方案的全部通信：设备、规则、解析/格式化、握手、收件箱、调试日志。"""

    def __init__(self, bus: EventBus, variables: GlobalVariables | None = None) -> None:
        self.bus = bus
        self.variables = variables if variables is not None else GlobalVariables(bus)
        self.devices: dict[str, CommDevice] = {}        # 按**名字**索引（界面显示用）
        self.by_id: dict[str, CommDevice] = {}          # 按**稳定 id** 索引（引用用）
        self.receive_rules: list[ReceiveRule] = []
        self.send_rules: list[SendRule] = []
        self.parse_rules: dict[str, ParseRule] = {}
        self.format_rules: dict[str, FormatRule] = {}
        self.handshakes: list[Handshake] = []
        self.runners: dict[str, FlowRunner] = {}
        self.log = CommLog()
        self._inboxes: dict[str, MessageInbox] = {}
        self._trigger_state: dict[str, _TriggerState] = {}
        self._lock = threading.RLock()
        self._bg: threading.Thread | None = None
        self._bg_stop = threading.Event()
        self.bus.subscribe(events.RUN_FINISHED, self._on_run_finished)
        # 调试日志：收到的报文在 _on_rx 里记（那里才知道解析结果），其余方向订阅事件
        self._log_subs = {events.COMM_SENT: self._log_sent, events.COMM_ERROR: self._log_error,
                          events.COMM_CONNECTED: self._log_connected,
                          events.COMM_DISCONNECTED: self._log_disconnected}
        for event, cb in self._log_subs.items():
            self.bus.subscribe(event, cb)

    # ---- 调试日志的事件订阅 ----
    def _log_sent(self, device="", data=b"", text="", peer="", **_) -> None:
        self.log.add("tx", device, peer, data, text)

    def _log_error(self, device="", error="", **_) -> None:
        self.log.add("err", device, "", note=error)

    def _log_connected(self, device="", peer="", generation=0, **_) -> None:
        self.log.add("sys", device, peer, note=f"已连接（第 {generation} 次）")

    def _log_disconnected(self, device="", reason="", **_) -> None:
        self.log.add("sys", device, "", note=f"已断开：{reason}" if reason else "已断开")

    # ================================================== 流程
    def set_runners(self, runners: dict[str, FlowRunner]) -> None:
        self.runners = dict(runners)

    def flow_names(self) -> list[str]:
        return list(self.runners)

    # ================================================== 设备
    def resolve(self, ref: str | CommDevice | None) -> CommDevice | None:
        """按 id 或名字找设备。规则与节点都走这里，所以改名不会破坏按 id 的引用。"""
        if ref is None or ref == "":
            return None
        if isinstance(ref, CommDevice):
            return ref
        key = str(ref)
        return self.by_id.get(key) or self.devices.get(key)

    def get(self, ref: str) -> CommDevice | None:
        return self.resolve(ref)

    def device_label(self, ref: str) -> str:
        dev = self.resolve(ref)
        return dev.name if dev is not None else (str(ref) or "—")

    def _unique_name(self, name: str, exclude: str = "") -> str:
        base = (name or "device").strip() or "device"
        candidate, i = base, 2
        while candidate in self.devices and self.devices[candidate].id != exclude:
            candidate = f"{base}_{i}"
            i += 1
        return candidate

    def add_device(self, name: str, kind: str, config: dict | None = None,
                   device_id: str | None = None, enabled: bool = True,
                   points: list[dict] | None = None) -> CommDevice:
        """新增（或按同名替换）一台设备。返回设备对象。"""
        if kind not in DEVICE_KINDS:
            planned = PLANNED_PROTOCOLS.get(kind)
            if planned:
                raise CommError(f"{planned}（{kind}）还没有实现；当前可用：{chr(44).join(sorted(DEVICE_KINDS))}")
            raise CommError(f"未知的设备类型 {kind!r}，可用：{', '.join(sorted(DEVICE_KINDS))}")
        with self._lock:
            old = self.devices.pop(name, None)
            if old is not None:
                self.by_id.pop(old.id, None)
                old.disconnect()
            dev = DEVICE_KINDS[kind](name, config, self.bus, device_id=device_id or (old.id if old else None),
                                     enabled=enabled)
            self._wire(dev)
            errors = []
            if isinstance(dev, _PointHost) and points:
                errors = dev.load_points(points)
            self.devices[name] = dev
            self.by_id[dev.id] = dev
        for e in errors:
            log.error("设备 %s 的数据点：%s", name, e)
            self.log.add("err", name, "", note=f"数据点配置：{e}")
        return dev

    def _wire(self, dev: CommDevice) -> None:
        dev.on_receive(self._on_rx)
        if hasattr(dev, "on_register_change"):
            dev.on_register_change(self._on_register)
        if isinstance(dev, _PointHost):
            dev.variables = self.variables
            dev.on_point_change(self._on_point)

    def update_device(self, ref: str, name: str | None = None, config: dict | None = None,
                      enabled: bool | None = None, points: list[dict] | None = None) -> CommDevice:
        """改配置但**保留 id 与引用**。原来连着的话会按新配置重连。"""
        dev = self.resolve(ref)
        if dev is None:
            raise CommError(f"没有设备 {ref!r}")
        was_connected = dev.connected
        new_name = self._unique_name(name if name is not None else dev.name, exclude=dev.id)
        with self._lock:
            dev.disconnect()
            self.devices.pop(dev.name, None)
            fresh = DEVICE_KINDS[dev.kind](
                new_name, config if config is not None else dev.config, self.bus,
                device_id=dev.id, enabled=dev.enabled if enabled is None else bool(enabled))
            self._wire(fresh)
            if isinstance(fresh, _PointHost):
                src = points if points is not None else (
                    dev.points_to_list() if isinstance(dev, _PointHost) else [])
                for e in fresh.load_points(src):
                    self.log.add("err", new_name, "", note=f"数据点配置：{e}")
            self.devices[new_name] = fresh
            self.by_id[fresh.id] = fresh
        if was_connected and fresh.enabled:
            try:
                fresh.open()
            except Exception as e:
                fresh._error(str(e))
        return fresh

    def rename_device(self, ref: str, new_name: str) -> CommDevice:
        """改名。按 id 的引用自动跟着走；老方案里按**名字**写的引用会一并改写。"""
        dev = self.resolve(ref)
        if dev is None:
            raise CommError(f"没有设备 {ref!r}")
        old_name = dev.name
        target = self._unique_name(new_name, exclude=dev.id)
        if target == old_name:
            return dev
        with self._lock:
            self.devices.pop(old_name, None)
            dev.name = target
            self.devices[target] = dev
        for rule in self.receive_rules:
            if rule.device == old_name:
                rule.device = dev.id
        for rule in self.send_rules:
            if rule.device == old_name:
                rule.device = dev.id
        for hs in self.handshakes:
            if hs.config.device == old_name:
                hs.config.device = dev.id
        for runner in self.runners.values():
            for node in runner.graph.nodes.values():
                for key, value in list(node.values.items()):
                    if "device" in key and value == old_name:
                        node.values[key] = dev.id
        self.log.add("sys", target, "", note=f"设备改名：{old_name} → {target}")
        return dev

    def duplicate_device(self, ref: str) -> CommDevice:
        dev = self.resolve(ref)
        if dev is None:
            raise CommError(f"没有设备 {ref!r}")
        cfg = dict(dev.config)
        for key in ("port", "local_port"):      # 端口不能撞，复制出来的先置 0 让用户改
            if key in cfg and isinstance(cfg.get(key), int) and dev.role in ("server", "slave"):
                cfg[key] = 0
        points = dev.points_to_list() if isinstance(dev, _PointHost) else None
        return self.add_device(self._unique_name(f"{dev.name}_副本"), dev.kind, cfg,
                               enabled=dev.enabled, points=points)

    def set_device_enabled(self, ref: str, enabled: bool) -> None:
        dev = self.resolve(ref)
        if dev is None:
            return
        dev.enabled = bool(enabled)
        if not dev.enabled:
            dev.close()
        self.log.add("sys", dev.name, "", note="已启用" if dev.enabled else "已禁用")

    def remove_device(self, ref: str) -> None:
        dev = self.resolve(ref)
        if dev is None:
            return
        with self._lock:
            self.devices.pop(dev.name, None)
            self.by_id.pop(dev.id, None)
        dev.disconnect()

    def references(self, ref: str) -> list[str]:
        """这台设备被谁引用了。删除前给用户看**具体位置**，而不是一句"仍被使用"。"""
        dev = self.resolve(ref)
        if dev is None:
            return []
        keys = {dev.id, dev.name}
        out: list[str] = []
        for rule in self.receive_rules:
            if rule.device in keys:
                out.append(f"接收规则「{rule.name}」")
        for rule in self.send_rules:
            if rule.device in keys:
                out.append(f"发送规则「{rule.name}」")
        for hs in self.handshakes:
            if hs.config.device in keys:
                out.append(f"检测握手「{hs.name}」")
        for flow, runner in self.runners.items():
            for node in runner.graph.nodes.values():
                for key, value in node.values.items():
                    if "device" in key and value in keys:
                        out.append(f"流程「{flow}」的节点「{node.name}」")
                        break
        return out

    def device_rows(self) -> list[dict]:
        """界面设备列表用的一行一个字典。"""
        rows = []
        for dev in self.devices.values():
            info = dev.info()
            info["kind_label"] = DEVICE_KIND_LABELS.get(dev.kind, dev.kind)
            info["status_note"] = DEVICE_STATUS_LABELS.get(DEVICE_STATUS.get(dev.kind, ""), "")
            info["last_comm"] = max(dev.last_rx, dev.last_tx)
            rows.append(info)
        return rows

    # ================================================== 连接
    def connect(self, ref: str) -> tuple[bool, str]:
        dev = self.resolve(ref)
        if dev is None:
            return False, f"没有设备 {ref!r}"
        if not dev.enabled:
            return False, f"{dev.name} 已禁用"
        try:
            dev.open()
            return True, f"{dev.name} 正在连接"
        except Exception as e:
            dev._error(str(e))
            return False, f"{dev.name}：{e}"

    def disconnect(self, ref: str) -> None:
        dev = self.resolve(ref)
        if dev is not None:
            dev.close()           # 手动断开：之后不再自动重连

    def connect_all(self) -> list[str]:
        """连接所有**启用且配置了自动连接**的设备，并启动后台维护线程。"""
        errors = []
        for dev in list(self.devices.values()):
            if not dev.enabled:
                continue
            if not dev.auto_connect:
                self.log.add("sys", dev.name, "", note="未设置自动连接，已跳过")
                continue
            try:
                dev.manual_stop = False
                dev.connect()
            except Exception as e:
                errors.append(f"{dev.name}: {e}")
                log.error("连接 %s 失败：%s", dev.name, e)
                self.log.add("err", dev.name, "", note=f"连接失败：{e}")
        self._start_background()
        return errors

    def disconnect_all(self) -> None:
        self._stop_background()
        for hs in self.handshakes:
            try:
                hs.shutdown()
            except Exception:
                log.exception("握手 %s 收尾失败", hs.name)
        for dev in list(self.devices.values()):
            try:
                dev.disconnect()
            except Exception:
                log.exception("断开 %s 失败", dev.name)

    def test_connection(self, ref: str) -> tuple[bool, str]:
        dev = self.resolve(ref)
        if dev is None:
            return False, f"没有名为 {ref!r} 的设备"
        try:
            return dev.test_connection()
        except Exception as e:
            return False, f"{dev.name}：{e}"

    # ================================================== 后台维护
    def _start_background(self) -> None:
        if self._bg and self._bg.is_alive():
            return
        self._bg_stop.clear()
        self._bg = threading.Thread(target=self._bg_loop, name="comm-maintenance", daemon=True)
        self._bg.start()

    def _stop_background(self) -> None:
        self._bg_stop.set()
        if self._bg:
            self._bg.join(2.0)
            self._bg = None

    def _bg_loop(self) -> None:
        """一条后台线程同时做三件事：心跳、不完整报文超时、握手状态维护。"""
        last_hb: dict[str, float] = {}
        while not self._bg_stop.is_set():
            now = time.time()
            for dev in list(self.devices.values()):
                try:
                    iv = dev.heartbeat_interval
                    if iv > 0 and dev.connected and now - last_hb.get(dev.id, 0.0) >= iv:
                        last_hb[dev.id] = now
                        dev.heartbeat()
                    dev.check_frame_timeout()
                except Exception as e:
                    dev._error(f"后台维护：{e}")
            for hs in list(self.handshakes):
                if hs.config.enabled:
                    try:
                        hs.tick(now)
                    except Exception:
                        log.exception("握手 %s 维护失败", hs.name)
            self._bg_stop.wait(0.05)

    # ================================================== 发送
    def send(self, device: str, data: bytes | str, peer: str | None = None,
             raw: bool = False) -> bool:
        dev = self.resolve(device)
        if dev is None:
            log.warning("发送：未知设备 %r", device)
            self.log.add("err", str(device), "", note="发送失败：未知设备")
            return False
        if not dev.enabled:
            self.log.add("err", dev.name, "", note="发送失败：设备已禁用")
            return False
        return dev.send(data, peer=peer, raw=raw)

    def modbus_write(self, device: str, address: Any, value: Any, kind: str = "int16") -> bool:
        """老接口：按裸地址写一个值。新流程建议用数据点名（``write_point``）。"""
        dev = self.resolve(device)
        if dev is None or not hasattr(dev, "write_value"):
            log.warning("Modbus 写入：%r 不是寄存器类设备", device)
            return False
        try:
            dev.write_value(_address(address), value, kind)
            return True
        except Exception as e:
            dev._error(f"写入失败：{e}")
            return False

    def write_point(self, device: str, point: str, value: Any) -> bool:
        dev = self.resolve(device)
        if not isinstance(dev, _PointHost):
            log.warning("写数据点：%r 不支持数据点", device)
            return False
        try:
            dev.write_point(point, value)
            self.log.add("tx", dev.name, "", note=f"数据点 {point} = {value}")
            return True
        except Exception as e:
            dev._error(f"写数据点 {point} 失败：{e}")
            return False

    def read_point(self, device: str, point: str, cached: bool = False) -> Any:
        dev = self.resolve(device)
        if not isinstance(dev, _PointHost):
            raise CommError(f"{device!r} 不支持数据点")
        if cached:
            pv = dev.point_value(point)
            if pv.valid:
                return pv.value
        return dev.read_point(point)

    # ================================================== 规则维护
    def add_receive_rule(self, rule: ReceiveRule) -> ReceiveRule:
        self.receive_rules.append(rule)
        return rule

    def add_send_rule(self, rule: SendRule) -> SendRule:
        self.send_rules.append(rule)
        return rule

    def add_parse_rule(self, rule: ParseRule) -> ParseRule:
        self.parse_rules[rule.name] = rule
        return rule

    def add_format_rule(self, rule: FormatRule) -> FormatRule:
        self.format_rules[rule.name] = rule
        return rule

    def add_handshake(self, config: HandshakeConfig) -> Handshake:
        hs = Handshake(config, self)
        self.handshakes.append(hs)
        return hs

    def handshake_for(self, device: CommDevice, flow: str) -> Handshake | None:
        for hs in self.handshakes:
            if not hs.config.enabled:
                continue
            dev = hs.device
            if dev is not None and dev.id == device.id and hs.config.flow == flow:
                return hs
        return None

    def handshake_by_name(self, name: str) -> Handshake | None:
        return next((h for h in self.handshakes if h.id == name or h.name == name), None)

    # ================================================== 收件箱
    def inbox(self, key: str, device: str = "", size: int = 16) -> MessageInbox:
        with self._lock:
            box = self._inboxes.get(key)
            if box is None or box.device != device or box.size != size:
                box = MessageInbox(key, device, size)
                self._inboxes[key] = box
            return box

    def release_inbox(self, key: str) -> None:
        with self._lock:
            self._inboxes.pop(key, None)

    # ================================================== 接收 → 触发
    def _on_rx(self, dev: CommDevice, data: bytes, peer: Peer | None = None) -> None:
        """后台线程：一条完整报文到了。"""
        text = dev._text(data)
        self.log.add("rx", dev.name, peer.id if peer else "", data, text)
        for box in list(self._inboxes.values()):
            if not box.device or box.device in (dev.id, dev.name):
                box.push(dev, data, peer, text)
        if not dev.enabled:
            return
        for rule in list(self.receive_rules):
            if not rule.enabled or rule.source != "message":
                continue
            if rule.device and not self._rule_device_matches(rule.device, dev):
                continue
            if not self._text_matches(rule, text):
                continue
            self._handle_message(rule, dev, data, text, peer)

    @staticmethod
    def _rule_device_matches(ref: str, dev: CommDevice) -> bool:
        return ref in (dev.id, dev.name)

    @staticmethod
    def _text_matches(rule: ReceiveRule, text: str) -> bool:
        p, m, t = rule.pattern, rule.match, text.strip()
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

    def _handle_message(self, rule: ReceiveRule, dev: CommDevice, data: bytes, text: str,
                        peer: Peer | None) -> None:
        if rule.action == "set_variable":
            self._apply_variable(rule, dev, data, text)
            return
        fields: dict[str, Any] = {}
        parse_rule = self.parse_rules.get(rule.parse) if rule.parse else None
        if parse_rule is not None:
            try:
                fields = parse_rule.parse(data)
            except ParseError as e:
                note = f"规则「{rule.name}」解析失败：{e}"
                log.warning("%s", note)
                self.log.add("err", dev.name, peer.id if peer else "", note=note)
                hs = self.handshake_for(dev, rule.flow)
                if hs is not None:
                    hs.reject(REJECTED_BAD_PARAMS, ERROR_CODES["bad_params"],
                              peer=peer.id if peer else "")
                return
            if parse_rule.to_variables:
                for k, v in fields.items():
                    self.variables.set(k, v)
        request_id = self._request_id(rule, fields, text)
        self._dispatch(rule, dev, flow=rule.flow, request_id=request_id, fields=fields,
                       peer=peer, message=text, raw=data)

    def _apply_variable(self, rule: ReceiveRule, dev: CommDevice, data: bytes, text: str) -> None:
        parse_rule = self.parse_rules.get(rule.parse) if rule.parse else None
        if parse_rule is not None:
            try:
                for k, v in parse_rule.parse(data).items():
                    self.variables.set(k, v)
                return
            except ParseError as e:
                self.log.add("err", dev.name, "", note=f"规则「{rule.name}」解析失败：{e}")
                return
        self.variables.set(rule.variable or rule.name, text.strip())

    @staticmethod
    def _request_id(rule: ReceiveRule, fields: dict, text: str) -> int:
        raw = fields.get(rule.request_field) if rule.request_field else None
        if raw is None:
            return 0
        try:
            return int(round(float(raw)))
        except (TypeError, ValueError):
            return 0

    # ---- 数据点变化 ----
    def _on_point(self, dev: CommDevice, point: DataPoint, old: Any = None, new: Any = None) -> None:
        self.log.add("rx", dev.name, "", note=f"数据点 {point.name}：{old} → {new}")
        # 确认位优先：对端确认了上一次结果
        for hs in list(self.handshakes):
            if (hs.config.enabled and hs.config.mode == "register" and hs.config.ack
                    and hs.config.ack == point.name):
                hsdev = hs.device
                if hsdev is not None and hsdev.id == dev.id and hs.on_ack(new):
                    return
        if not dev.enabled:
            return
        for rule in list(self.receive_rules):
            if not rule.enabled or rule.source != "datapoint":
                continue
            if rule.device and not self._rule_device_matches(rule.device, dev):
                continue
            if (rule.point or "") != point.name:
                continue
            self._evaluate_data_rule(rule, dev, point.name, old, new)

    def _on_register(self, dev: CommDevice, address: int, old: int, new: int) -> None:
        """老方案：按裸地址的触发。数据点规则走 ``_on_point``。"""
        for rule in list(self.receive_rules):
            if not rule.enabled or rule.source != "datapoint" or rule.point:
                continue
            if rule.device and not self._rule_device_matches(rule.device, dev):
                continue
            if int(rule.address) != int(address):
                continue
            if self._evaluate_data_rule(rule, dev, f"addr:{address}", old, new):
                if hasattr(dev, "on_trigger"):
                    try:
                        dev.on_trigger(address)    # 简版握手：置忙 / 清零触发寄存器
                    except Exception as e:
                        dev._error(f"握手失败：{e}")

    def _evaluate_data_rule(self, rule: ReceiveRule, dev: CommDevice, key: str,
                            old: Any, new: Any) -> bool:
        """判断一条数据点规则是否命中并触发。返回是否真的触发了。"""
        state = self._trigger_state.setdefault(f"{rule.id}:{dev.id}:{key}", _TriggerState())
        hit, arm_ok = self._condition(rule, old, new)
        if not arm_ok:
            state.armed = True            # 条件已离开 → 重新武装，下次满足才再触发
        if not hit:
            state.last_value = new
            return False
        if not state.armed:
            # 触发位一直保持在触发状态：不重复执行
            state.last_value = new
            return False
        state.armed = False
        state.last_value = new
        state.last_fire = time.time()
        if rule.action == "set_variable":
            self.variables.set(rule.variable or rule.name, new)
            return True
        fields: dict[str, Any] = {key: new}
        request_id = 0
        if rule.request_field and isinstance(dev, _PointHost) and dev.has_point(rule.request_field):
            try:
                pv = dev.point_value(rule.request_field)
                raw = pv.value if pv.valid else dev.read_point(rule.request_field)
                request_id = int(round(float(raw)))
                fields[rule.request_field] = raw
            except Exception as e:
                log.debug("读请求编号 %s 失败：%s", rule.request_field, e)
        self._dispatch(rule, dev, flow=rule.flow, request_id=request_id, fields=fields,
                       peer=None, message=f"{key}={new}", raw=b"")
        return True

    @staticmethod
    def _condition(rule: ReceiveRule, old: Any, new: Any) -> tuple[bool, bool]:
        """返回 ``(本次是否满足, 条件是否仍然成立)``。后者用于边沿重新武装。"""
        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        m = rule.match
        n, o = num(new), num(old)
        if m == "rising":
            return (o == 0 and n != 0), n != 0
        if m == "falling":
            return (o != 0 and n == 0), n == 0
        if m == "changed":
            return (old is not None and old != new), False
        if m == "equals":
            return (n == num(rule.value) and o != num(rule.value)), n == num(rule.value)
        if m == "not_equals":
            return (n != num(rule.value) and o == num(rule.value)), n != num(rule.value)
        if m == "greater":
            return (n > num(rule.value) and not o > num(rule.value)), n > num(rule.value)
        if m == "less":
            return (n < num(rule.value) and not o < num(rule.value)), n < num(rule.value)
        if m == "in_range":
            lo, hi = sorted((num(rule.value), num(rule.value2)))
            inside, was = lo <= n <= hi, lo <= o <= hi
            return (inside and not was), inside
        return False, False

    # ---- 统一入队 ----
    def _dispatch(self, rule: ReceiveRule, dev: CommDevice, flow: str, request_id: int,
                  fields: dict, peer: Peer | None, message: str, raw: bytes) -> str:
        """握手判断 + 忙时策略 + 入队。返回状态字符串。"""
        hs = self.handshake_for(dev, flow)
        peer_id = peer.id if peer else ""
        if hs is not None:
            status, code = hs.begin(request_id, peer_id, fields)
            if status != ACCEPTED:
                self.log.add("sys", dev.name, peer_id,
                             note=f"规则「{rule.name}」未接受：{status}"
                                  f"（{ERROR_TEXT.get(code, code)}）")
                return status
        trigger = Trigger(TriggerSource.COMM, device=dev.name, message=message, payload={
            "rule": rule.name, "rule_id": rule.id, "device": dev.name, "device_id": dev.id,
            "peer": peer_id, "generation": dev.generation, "request_id": request_id,
            "fields": dict(fields), "raw": raw, "handshake": hs.id if hs else "",
            "accepted_at": time.time(),
        })
        status, code = self._enqueue(rule, flow, trigger)
        if status != ACCEPTED:
            self.log.add("sys", dev.name, peer_id,
                         note=f"规则「{rule.name}」{status}：{ERROR_TEXT.get(code, '')}")
            if hs is not None:
                hs.reject(status, code, request_id, peer_id)
        return status

    def _enqueue(self, rule: ReceiveRule, flow: str, trigger: Trigger) -> tuple[str, int]:
        """按忙时策略把一次触发放进流程队列。"""
        runner = self.runners.get(flow)
        if runner is None:
            log.warning("规则 %s：找不到流程 %r", rule.name, flow)
            return "no_flow", ERROR_CODES["not_running"]
        if not runner.running:
            log.info("规则 %s：流程 %r 未运行，忽略触发", rule.name, flow)
            return "not_running", ERROR_CODES["not_running"]
        policy = rule.busy_policy if rule.busy_policy in BUSY_POLICIES else "queue"
        pending, busy = runner.pending(), runner.busy
        if policy == "reject" and (busy or pending):
            return "busy", ERROR_CODES["busy"]
        if policy == "drop" and (busy or pending):
            runner.stats.dropped += 1
            return "dropped", ERROR_CODES["busy"]
        if policy == "queue" and pending >= max(1, int(rule.max_queue)):
            return "queue_full", ERROR_CODES["busy"]
        runner.trigger(trigger)
        return ACCEPTED, ERROR_CODES["none"]

    # ================================================== 流程结束 → 发送
    def _on_run_finished(self, result: RunResult) -> None:
        payload = dict(getattr(result.trigger, "payload", None) or {})
        extra = {"request_id": payload.get("request_id", 0), "fields": payload.get("fields", {}),
                 "peer": payload.get("peer", ""), "device": payload.get("device", ""),
                 "error_code": 0}
        variables = result.variables or self.variables.snapshot()
        for rule in list(self.send_rules):
            if not rule.enabled or (rule.flow and rule.flow != result.flow):
                continue
            if rule.when == "ok" and not result.passed:
                continue
            if rule.when == "ng" and (result.passed or not result.ok):
                continue
            if rule.when == "error" and result.ok:
                continue
            self._run_send_rule(rule, result, variables, extra, payload)
        # 握手收尾：结果值已经写完，这时才写 Done 与完成编号
        hs = self.handshake_by_name(payload.get("handshake", "")) if payload.get("handshake") else None
        if hs is not None:
            try:
                hs.finish(result, payload)
            except Exception:
                log.exception("握手 %s 收尾失败", hs.name)
        # 老方案的简版握手：清掉触发设备的忙标志
        trig_dev = self.resolve(payload.get("device_id") or getattr(result.trigger, "device", "") or "")
        if trig_dev is not None and hs is None and hasattr(trig_dev, "on_done"):
            try:
                trig_dev.on_done(result)
            except Exception as e:
                trig_dev._error(f"握手收尾失败：{e}")

    def _run_send_rule(self, rule: SendRule, result: RunResult, variables: dict,
                       extra: dict, payload: dict) -> None:
        dev = self.resolve(rule.device)
        if dev is None:
            self.log.add("err", str(rule.device), "", note=f"发送规则「{rule.name}」：设备不存在")
            return
        if not dev.enabled or not dev.connected:
            self.log.add("err", dev.name, "", note=f"发送规则「{rule.name}」：设备未连接，未发送")
            return
        gen = payload.get("generation")
        if gen is not None and dev.id == payload.get("device_id") and gen != dev.generation:
            # 触发这次运行的连接已经断过：结果属于旧任务，不发
            self.log.add("sys", dev.name, "", note=f"发送规则「{rule.name}」：原连接已断开，丢弃本次结果")
            return
        peer = payload.get("peer", "") if rule.target == "origin" else ""
        if peer and peer not in dev.peers:
            self.log.add("err", dev.name, peer, note=f"发送规则「{rule.name}」：来源对端已断开")
            peer = ""
            if dev.supports_peers:
                return                   # 定向回复找不到对端，宁可不发，也不要发错地方
        try:
            for item in rule.points or []:
                name = str(item.get("point", "")).strip()
                if not name:
                    continue
                value = self._value_for(item.get("source", item.get("expr", "")), result, variables, extra)
                scale = float(item.get("scale", 1.0) or 1.0)
                if scale != 1.0 and isinstance(value, (int, float)):
                    value = value * scale
                dev.write_point(name, value)                       # type: ignore[attr-defined]
            for reg in rule.registers or []:
                value = eval_register_expr(reg.get("expr", "0"), result, variables, extra)
                dev.write_value(_address(reg.get("address", 0)), value,              # type: ignore[attr-defined]
                                str(reg.get("kind", "int16")))
            fmt = self.format_rules.get(rule.format) if rule.format else None
            if fmt is not None:
                dev.send(fmt.render(result, variables, extra), peer=peer or None, raw=True)
            elif rule.template:
                dev.send(format_template(rule.template, result, variables, **extra),
                         peer=peer or None, raw=True)
        except Exception as e:
            dev._error(f"发送规则 {rule.name!r}：{e}")

    @staticmethod
    def _value_for(spec: Any, result: RunResult | None, variables: dict, extra: dict) -> Any:
        """数据点写入的取值：既接受 ``out.width`` 这种来源，也接受 ``{out.width}`` 模板。"""
        from .parsing import resolve_source
        text = str(spec or "")
        if "{" in text:
            return eval_register_expr(text, result, variables, extra)
        return resolve_source(text, result, variables, extra)

    # ================================================== 持久化
    def to_dict(self) -> dict:
        return {
            "devices": [d.to_dict() for d in self.devices.values()],
            "receive_rules": [r.to_dict() for r in self.receive_rules],
            "send_rules": [r.to_dict() for r in self.send_rules],
            "parse_rules": [r.to_dict() for r in self.parse_rules.values()],
            "format_rules": [r.to_dict() for r in self.format_rules.values()],
            "handshakes": [h.config.to_dict() for h in self.handshakes],
            "log_capacity": self.log.capacity,
        }

    def load_dict(self, cfg: dict | None) -> list[str]:
        """按方案里的通信配置重建一切。返回出错信息（坏的那条跳过，其它照常载入）。"""
        errors: list[str] = []
        self.disconnect_all()
        with self._lock:
            self.devices.clear()
            self.by_id.clear()
            self._inboxes.clear()
            self._trigger_state.clear()
        cfg = cfg or {}
        self.handshakes = []
        self.parse_rules = {}
        self.format_rules = {}
        for d in cfg.get("parse_rules", []):
            try:
                self.add_parse_rule(ParseRule.from_dict(d))
            except Exception as e:
                errors.append(f"解析规则 {d.get('name')}: {e}")
        for d in cfg.get("format_rules", []):
            try:
                self.add_format_rule(FormatRule.from_dict(d))
            except Exception as e:
                errors.append(f"格式化规则 {d.get('name')}: {e}")
        self.receive_rules = []
        for d in cfg.get("receive_rules", []):
            try:
                self.receive_rules.append(ReceiveRule.from_dict(d))
            except Exception as e:
                errors.append(f"接收规则 {d.get('name')}: {e}")
        self.send_rules = []
        for d in cfg.get("send_rules", []):
            try:
                self.send_rules.append(SendRule.from_dict(d))
            except Exception as e:
                errors.append(f"发送规则 {d.get('name')}: {e}")
        for d in cfg.get("devices", []):
            try:
                self.add_device(d["name"], d["kind"], d.get("config"),
                                device_id=d.get("id"), enabled=bool(d.get("enabled", True)),
                                points=d.get("points"))
            except Exception as e:
                errors.append(f"{d.get('name')}: {e}")
        for d in cfg.get("handshakes", []):
            try:
                self.add_handshake(HandshakeConfig.from_dict(d))
            except Exception as e:
                errors.append(f"检测握手 {d.get('name')}: {e}")
        cap = int(cfg.get("log_capacity", 0) or 0)
        if cap:
            self.log = CommLog(cap)
        return errors

    def validate(self) -> list[str]:
        """配置自检：引用的设备/流程/规则是否存在，数据点是否合法。"""
        problems: list[str] = []
        for dev in self.devices.values():
            if isinstance(dev, _PointHost):
                problems += [f"设备「{dev.name}」：{e}" for e in dev.points.validate_all()]
        for rule in self.receive_rules:
            if rule.device and self.resolve(rule.device) is None:
                problems.append(f"接收规则「{rule.name}」引用了不存在的设备 {rule.device!r}")
            if rule.parse and rule.parse not in self.parse_rules:
                problems.append(f"接收规则「{rule.name}」引用了不存在的解析规则 {rule.parse!r}")
            if rule.action == "trigger_flow" and self.runners and rule.flow not in self.runners:
                problems.append(f"接收规则「{rule.name}」引用了不存在的流程 {rule.flow!r}")
        for rule in self.send_rules:
            dev = self.resolve(rule.device)
            if dev is None:
                problems.append(f"发送规则「{rule.name}」引用了不存在的设备 {rule.device!r}")
            if rule.format and rule.format not in self.format_rules:
                problems.append(f"发送规则「{rule.name}」引用了不存在的格式化规则 {rule.format!r}")
            for item in rule.points or []:
                name = str(item.get("point", ""))
                if dev is not None and isinstance(dev, _PointHost) and name and not dev.has_point(name):
                    problems.append(f"发送规则「{rule.name}」：设备「{dev.name}」没有数据点 {name!r}")
        for hs in self.handshakes:
            if hs.device is None:
                problems.append(f"检测握手「{hs.name}」引用了不存在的设备 {hs.config.device!r}")
            if self.runners and hs.config.flow not in self.runners:
                problems.append(f"检测握手「{hs.name}」引用了不存在的流程 {hs.config.flow!r}")
            if hs.config.mode == "text":
                for attr in ("result_format", "busy_format", "error_format", "accept_format"):
                    name = getattr(hs.config, attr)
                    if name and name not in self.format_rules:
                        problems.append(f"检测握手「{hs.name}」的 {attr} 引用了不存在的格式化规则 {name!r}")
        return problems

    def status(self) -> dict:
        return {"devices": self.device_rows(),
                "handshakes": [h.status_dict() for h in self.handshakes],
                "log": {"count": len(self.log.entries()), "dropped": self.log.dropped,
                        "capacity": self.log.capacity, "paused": self.log.paused}}

    def shutdown(self) -> None:
        self._stop_background()
        self.disconnect_all()
        with self._lock:
            self._inboxes.clear()
        self.bus.unsubscribe(events.RUN_FINISHED, self._on_run_finished)
        for event, cb in self._log_subs.items():
            self.bus.unsubscribe(event, cb)


_current: CommManager | None = None


def set_manager(mgr: CommManager | None) -> None:
    global _current
    _current = mgr


def get_manager() -> CommManager | None:
    return _current

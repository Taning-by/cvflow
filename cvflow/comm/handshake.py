"""检测握手：外部请求 → 接受任务 → 执行流程 → 发布结果 → 对端确认。

这是产线上"一次拍照"真正的业务闭环。一个握手把**一台设备**和**一条流程**绑在一起，维护
一组对外可见的信号：

| 信号 | 含义 |
|---|---|
| ``ready``      | 现在能不能接受新任务。**反映真实能力**：设备已连接 + 流程在运行 + 不忙 |
| ``busy``       | 已接受任务、正在执行 |
| ``done``       | 本次结果已经写完，可以读了 |
| ``result``     | 判定结果（OK/NG，默认 1/2，可配） |
| ``error_code`` | 0 正常，其余见 ``ERROR_CODES``（忙、参数错误、流程异常、超时…） |
| ``request_id`` | 对端给的请求编号（由对端写入；握手只读它） |
| ``done_id``    | 本次完成对应的请求编号。**对端用它确认结果属于哪一次请求** |
| ``ack``        | 对端确认位；收到后握手复位 ``done`` / ``done_id`` / ``result``，重新置 ``ready`` |

关键时序（三条都是硬要求）：

1. **先结果、后 Done**。发送规则写完测量值，才写 ``done`` 和 ``done_id``；对端看到 Done 翻转
   时读到的一定是本次的数据。
2. **结果绑定请求**。接受任务的瞬间就把请求编号、输入参数和**来源对端**冻结进本次触发，
   流程跑完按这份冻结信息回复——TCP 服务端因此总能回给正确的那个客户端。
3. **断线不重放**。接受任务时记下设备的连接代号；流程跑完发现代号变了（中间断过线），
   结果直接丢弃并记日志，绝不把旧任务的结果写给新连上来的对端。

``register`` 模式的信号是 Modbus/MC/S7 的数据点（按名字引用，也接受裸地址）；
``text`` 模式的信号是文本报文，由格式化规则渲染（见 ``parsing.FormatRule``）。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .base import CommDevice

if TYPE_CHECKING:  # pragma: no cover
    from .manager import CommManager

log = logging.getLogger("cvflow.comm")

#: 对外的错误码。0 表示正常，其余值写进 ``error_code`` 信号。
ERROR_CODES = {
    "none": 0,
    "busy": 1,            # 忙，本次请求未被接受
    "bad_params": 2,      # 报文解析失败 / 参数非法
    "flow_error": 3,      # 流程里有节点报错
    "timeout": 4,         # 流程超时未完成
    "not_running": 5,     # 流程没在运行模式
    "duplicate": 6,       # 重复请求
    "internal": 7,        # 软件内部错误
    "ng": 8,              # 判定为 NG（可选：有些产线希望 NG 也带错误码）
}
ERROR_TEXT = {
    0: "正常", 1: "忙", 2: "参数错误", 3: "流程异常", 4: "超时",
    5: "流程未运行", 6: "重复请求", 7: "内部错误", 8: "判定 NG",
}

STATES = ("ready", "busy", "done", "wait_ack", "offline")
STATE_LABELS = {"ready": "就绪", "busy": "执行中", "done": "已完成", "wait_ack": "等待确认",
                "offline": "离线"}

#: 接受请求的结果。``accepted`` 以外的都没有真正启动流程。
ACCEPTED = "accepted"
REJECTED_BUSY = "busy"
REJECTED_DUPLICATE = "duplicate"
REJECTED_NOT_RUNNING = "not_running"
REJECTED_BAD_PARAMS = "bad_params"
REJECTED_ERROR = "error"

BUSY_POLICIES = ("reject", "drop", "queue")
BUSY_POLICY_LABELS = {"reject": "拒绝并回复忙", "drop": "静默丢弃",
                      "queue": "有界排队（满了回复忙）"}

DEDUP_ACTIONS = ("resend", "ignore", "accept")
DEDUP_LABELS = {"resend": "重发上次结果", "ignore": "忽略（不回复）", "accept": "当成新请求"}


@dataclass
class HandshakeConfig:
    id: str = ""
    name: str = "handshake"
    device: str = ""                 # 设备 id（兼容旧方案里的设备名）
    flow: str = "main"
    mode: str = "register"           # register | text
    enabled: bool = True

    # ---- register 模式的信号：数据点名，或 "40011" / "4x10" 这样的地址 ----
    ready: str = ""
    busy: str = ""
    done: str = ""
    result: str = ""
    error_code: str = ""
    request_id: str = ""
    done_id: str = ""
    ack: str = ""
    ack_value: int = 1
    ok_value: int = 1
    ng_value: int = 2

    # ---- text 模式：格式化规则名 ----
    accept_format: str = ""
    result_format: str = ""
    busy_format: str = ""
    error_format: str = ""

    # ---- 策略 ----
    require_ack: bool = False
    auto_reset_s: float = 0.0        # 等确认超过这么久自动复位；0 = 一直等
    run_timeout_s: float = 10.0      # 流程多久没跑完算超时；0 = 不判超时
    busy_policy: str = "reject"
    max_queue: int = 1               # busy_policy == "queue" 时的队列上限
    dedup_window_s: float = 5.0      # 同一请求编号在这个时间窗内算重复；0 = 不去重
    dedup_action: str = "resend"
    ng_error_code: bool = False      # 判定 NG 时是否也写错误码 8

    def __post_init__(self) -> None:
        if self.mode not in ("register", "text"):
            raise ValueError(f"握手 {self.name!r}：模式只能是 register 或 text")
        if self.busy_policy not in BUSY_POLICIES:
            raise ValueError(f"握手 {self.name!r}：未知的忙时策略 {self.busy_policy!r}")
        if self.dedup_action not in DEDUP_ACTIONS:
            raise ValueError(f"握手 {self.name!r}：未知的去重策略 {self.dedup_action!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HandshakeConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class _InFlight:
    """一次已接受的任务。接受瞬间冻结，流程跑完按它回复。"""

    request_id: int = 0
    peer: str = ""                   # 来源对端 id（TCP 服务端回复要用）
    generation: int = 0              # 设备的连接代号；断线后变化 → 旧任务作废
    started: float = field(default_factory=time.time)
    fields: dict[str, Any] = field(default_factory=dict)
    run_id: int = 0


class Handshake:
    """一个握手实例：配置 + 运行时状态。所有对外写操作都带异常保护，不会把流程拖垮。"""

    def __init__(self, config: HandshakeConfig, manager: "CommManager") -> None:
        self.config = config
        self.mgr = manager
        self.state = "offline"
        self.lock = threading.RLock()
        self.in_flight: list[_InFlight] = []
        self.last: _InFlight | None = None
        self.last_result_payload: dict[str, Any] = {}
        self.last_result_bytes: bytes = b""       # 上次发出去的结果报文原文（重复请求时重发它）
        self.last_sent_ok = False            # 上一次结果"发出去了"
        self.last_acked = False              # 上一次结果"对端确认了"（和发出去是两件事）
        self.last_error_code = 0
        self.last_status = ""
        self.counters = {"accepted": 0, "rejected_busy": 0, "duplicate": 0, "timeout": 0,
                         "abandoned": 0, "completed": 0, "acked": 0, "send_failed": 0}
        self._ready_written: bool | None = None
        self._wait_ack_since = 0.0

    # ---- 便捷访问 ----
    @property
    def id(self) -> str:
        return self.config.id or self.config.name

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def device(self) -> CommDevice | None:
        return self.mgr.resolve(self.config.device)

    @property
    def runner(self):
        return self.mgr.runners.get(self.config.flow)

    @property
    def busy(self) -> bool:
        return bool(self.in_flight)

    def can_accept(self) -> bool:
        """Ready 的真实含义：设备连着、流程在跑、当前不忙、没有在等确认。"""
        dev, runner = self.device, self.runner
        if dev is None or not dev.enabled or not dev.connected:
            return False
        if runner is None or not runner.running:
            return False
        if self.config.require_ack and self.state == "wait_ack":
            return False
        if not self.busy:
            return True
        if self.config.busy_policy == "queue":
            return len(self.in_flight) < max(1, int(self.config.max_queue))
        return False

    # ---- 信号读写 ----
    def _write(self, signal: str, value: Any) -> bool:
        """写一个 register 模式的信号。信号名留空表示没配，直接跳过。"""
        name = (signal or "").strip()
        if not name or self.config.mode != "register":
            return False
        dev = self.device
        if dev is None or not dev.connected:
            return False
        try:
            if hasattr(dev, "has_point") and dev.has_point(name):     # type: ignore[attr-defined]
                dev.write_point(name, value)                          # type: ignore[attr-defined]
            else:
                from .modbus_map import parse_reference
                area, addr = parse_reference(name)
                if hasattr(dev, "write_area"):
                    bits = area in ("coil", "discrete")
                    dev.write_area(area, addr, [bool(value) if bits else int(round(float(value)))])  # type: ignore[attr-defined]
                elif hasattr(dev, "write_value"):
                    dev.write_value(addr, value, "int16")             # type: ignore[attr-defined]
                else:
                    return False
            return True
        except Exception as e:
            log.warning("握手 %s：写信号 %s=%r 失败：%s", self.name, name, value, e)
            dev._error(f"握手 {self.name}：写 {name} 失败：{e}")
            return False

    def _read(self, signal: str) -> Any:
        name = (signal or "").strip()
        if not name or self.config.mode != "register":
            return None
        dev = self.device
        if dev is None:
            return None
        try:
            if hasattr(dev, "has_point") and dev.has_point(name):     # type: ignore[attr-defined]
                pv = dev.point_value(name)                            # type: ignore[attr-defined]
                if pv.valid:
                    return pv.value
                return dev.read_point(name)                           # type: ignore[attr-defined]
            from .modbus_map import parse_reference
            area, addr = parse_reference(name)
            if hasattr(dev, "read_area"):
                v = dev.read_area(area, addr, 1)[0]                   # type: ignore[attr-defined]
                return v
        except Exception as e:
            log.debug("握手 %s：读信号 %s 失败：%s", self.name, name, e)
        return None

    def _resend_last(self, peer: str = "") -> bool:
        """重复请求：把上次**实际发出去的那串字节**原样再发一遍。

        不重新渲染——流程没有重跑，重新渲染拿不到当时的测量值，只会发出一条空结果。
        """
        if self.config.mode != "text" or not self.last_result_bytes:
            return False
        dev = self.device
        if dev is None or not dev.connected:
            return False
        if peer and peer not in dev.peers:
            self.counters["send_failed"] += 1
            return False
        return dev.send(self.last_result_bytes, peer=peer or None, raw=True)

    def _send_text(self, format_name: str, extra: dict[str, Any], peer: str = "",
                   result: Any = None) -> bool:
        """发一条 text 模式的报文。回复**一定**发给冻结下来的来源对端。"""
        if self.config.mode != "text":
            return False
        rule = self.mgr.format_rules.get((format_name or "").strip())
        if rule is None:
            if format_name:
                log.warning("握手 %s：没有格式化规则 %r", self.name, format_name)
            return False
        dev = self.device
        if dev is None or not dev.connected:
            return False
        try:
            data = rule.render(result, self.mgr.variables.snapshot(), extra)
        except Exception as e:
            dev._error(f"握手 {self.name}：渲染 {format_name} 失败：{e}")
            return False
        target = peer or ""
        if target and target not in dev.peers:
            dev._error(f"握手 {self.name}：来源对端 {target} 已断开，本次结果未发送")
            self.counters["send_failed"] += 1
            return False
        ok = dev.send(data, peer=target or None, raw=True)
        if not ok:
            self.counters["send_failed"] += 1
        elif format_name == self.config.result_format:
            self.last_result_bytes = data
        return ok

    # ---- 对外：接受请求 ----
    def begin(self, request_id: int, peer: str = "", fields: dict[str, Any] | None = None,
              ) -> tuple[str, int]:
        """尝试接受一次请求。返回 ``(状态, 错误码)``；状态为 ``accepted`` 才算真的接受。

        **不在这里启动流程**——由管理器在拿到 ``accepted`` 后入队，这样忙时策略和队列上限
        只有一处判断。
        """
        cfg = self.config
        with self.lock:
            dev = self.device
            if dev is None or not dev.connected:
                return REJECTED_ERROR, ERROR_CODES["internal"]
            runner = self.runner
            if runner is None or not runner.running:
                self.last_status = "流程未运行"
                self._reply_error(ERROR_CODES["not_running"], request_id, peer)
                return REJECTED_NOT_RUNNING, ERROR_CODES["not_running"]

            # 去重：同一个请求编号在时间窗内再来一次
            if (cfg.dedup_window_s > 0 and request_id and self.last is not None
                    and self.last.request_id == request_id
                    and time.time() - self.last.started <= cfg.dedup_window_s
                    and cfg.dedup_action != "accept"):
                self.counters["duplicate"] += 1
                self.last_status = f"请求 {request_id} 重复"
                if cfg.dedup_action == "resend":
                    self._resend_last(peer or self.last.peer)
                return REJECTED_DUPLICATE, ERROR_CODES["duplicate"]

            if not self.can_accept():
                self.counters["rejected_busy"] += 1
                self.last_status = "忙"
                if cfg.busy_policy == "drop":
                    return REJECTED_BUSY, ERROR_CODES["busy"]
                self._write(cfg.error_code, ERROR_CODES["busy"])
                self._send_text(cfg.busy_format, {"request_id": request_id,
                                                  "error_code": ERROR_CODES["busy"],
                                                  "error": "忙"}, peer)
                return REJECTED_BUSY, ERROR_CODES["busy"]

            job = _InFlight(request_id=int(request_id or 0), peer=peer or "",
                            generation=dev.generation, fields=dict(fields or {}))
            self.in_flight.append(job)
            self.last = job
            self.counters["accepted"] += 1
            self.last_status = f"已接受请求 {job.request_id}"
            self.last_acked = False
            self.last_sent_ok = False
            # 开始新任务：先清掉上一次的完成状态，免得对端读到旧结果
            self._write(cfg.done, 0)
            self._write(cfg.done_id, 0)
            self._write(cfg.result, 0)
            self._write(cfg.error_code, ERROR_CODES["none"])
            self._write(cfg.busy, 1)
            self._set_ready(False, force=True)
            self.state = "busy"
            self._send_text(cfg.accept_format, {"request_id": job.request_id,
                                                "fields": job.fields}, peer)
            return ACCEPTED, ERROR_CODES["none"]

    def reject(self, status: str, error_code: int, request_id: int = 0, peer: str = "") -> None:
        """管理器在 begin 之后发现无法入队时调用，把已经置上的忙状态收回去。"""
        with self.lock:
            self.in_flight = [j for j in self.in_flight if j.request_id != request_id or j.peer != peer]
            self._write(self.config.error_code, error_code)
            self._write(self.config.busy, 1 if self.in_flight else 0)
            self.state = "busy" if self.in_flight else "ready"
            self.last_status = f"{status}：{ERROR_TEXT.get(error_code, error_code)}"
            self._reply_error(error_code, request_id, peer)
            self.tick()

    def attach_run(self, request_id: int, run_id: int) -> None:
        with self.lock:
            for job in self.in_flight:
                if job.request_id == request_id and not job.run_id:
                    job.run_id = run_id
                    return

    # ---- 对外：流程结束 ----
    def finish(self, result: Any, payload: dict[str, Any] | None = None) -> None:
        """流程跑完（**测量值已由发送规则写完**）之后调用：写结果状态、Done、完成编号。"""
        cfg = self.config
        with self.lock:
            dev = self.device
            job = self._take_job(payload or {})
            if job is None:
                return
            if dev is None or not dev.connected or dev.generation != job.generation:
                # 中间断过线：这次结果属于已经不存在的那条连接，丢弃
                self.counters["abandoned"] += 1
                self.last_status = f"请求 {job.request_id} 的结果因断线作废"
                log.warning("握手 %s：请求 %s 的结果因断线作废（代号 %s → %s）", self.name,
                            job.request_id, job.generation,
                            dev.generation if dev else "设备已删除")
                self._refresh_state()
                return

            ok = bool(getattr(result, "passed", False))
            errored = not bool(getattr(result, "ok", True))
            code = ERROR_CODES["none"]
            if errored:
                code = ERROR_CODES["flow_error"]
            elif not ok and cfg.ng_error_code:
                code = ERROR_CODES["ng"]
            self.last_error_code = code

            extra = {"request_id": job.request_id, "error_code": code,
                     "error_text": ERROR_TEXT.get(code, ""), "fields": job.fields,
                     "device": dev.name, "peer": job.peer}
            self.last_result_payload = dict(extra)

            # 顺序是硬要求：结果值 → result/error_code → done_id → done
            self._write(cfg.result, cfg.ok_value if ok else cfg.ng_value)
            self._write(cfg.error_code, code)
            self._write(cfg.done_id, job.request_id)
            self._write(cfg.done, 1)
            sent = self._send_text(cfg.result_format, extra, job.peer, result)
            self.last_sent_ok = bool(sent) if cfg.mode == "text" else True
            self.counters["completed"] += 1
            self._write(cfg.busy, 1 if self.in_flight else 0)
            if cfg.require_ack:
                self.state = "wait_ack"
                self._wait_ack_since = time.time()
                self.last_status = f"请求 {job.request_id} 已完成，等待对端确认"
            else:
                self.state = "done"
                self.last_acked = False
                self.last_status = f"请求 {job.request_id} 已完成"
            self._refresh_state()

    def _take_job(self, payload: dict[str, Any]) -> _InFlight | None:
        """按请求编号/运行号取出对应的在飞任务，取不到就退回最早的一条。"""
        rid = payload.get("request_id")
        if rid is not None:
            for i, job in enumerate(self.in_flight):
                if job.request_id == int(rid or 0):
                    return self.in_flight.pop(i)
        if self.in_flight:
            return self.in_flight.pop(0)
        return None

    # ---- 对外：对端确认 ----
    def on_ack(self, value: Any) -> bool:
        """对端把确认位写成 ``ack_value`` 时调用。返回是否真的复位了。"""
        cfg = self.config
        with self.lock:
            try:
                hit = int(round(float(value))) == int(cfg.ack_value)
            except (TypeError, ValueError):
                hit = bool(value)
            if not hit or self.state not in ("wait_ack", "done"):
                return False
            self.counters["acked"] += 1
            self.last_acked = True
            self.last_status = "对端已确认"
            self._write(cfg.done, 0)
            self._write(cfg.done_id, 0)
            self._write(cfg.result, 0)
            self._write(cfg.error_code, ERROR_CODES["none"])
            self._write(cfg.ack, 0)          # 把确认位也清掉，便于下一次产生上升沿
            self.state = "ready" if not self.in_flight else "busy"
            self._refresh_state()
            return True

    # ---- 定时维护 ----
    def tick(self, now: float | None = None) -> None:
        """由管理器的后台线程按 ~50 ms 调用：维护 Ready、判超时、必要时自动复位。"""
        now = time.time() if now is None else now
        cfg = self.config
        with self.lock:
            # 流程超时
            if cfg.run_timeout_s > 0:
                for job in list(self.in_flight):
                    if now - job.started >= cfg.run_timeout_s:
                        self.in_flight.remove(job)
                        self.counters["timeout"] += 1
                        self.last_error_code = ERROR_CODES["timeout"]
                        self.last_status = f"请求 {job.request_id} 超时未完成"
                        log.warning("握手 %s：请求 %s 超过 %.1f 秒未完成", self.name,
                                    job.request_id, cfg.run_timeout_s)
                        self._write(cfg.error_code, ERROR_CODES["timeout"])
                        self._write(cfg.done_id, job.request_id)
                        self._write(cfg.done, 1)
                        self._write(cfg.busy, 1 if self.in_flight else 0)
                        # 超时只上报，**不重发触发**：重发可能让对端执行两次物理动作
                        self._send_text(cfg.error_format,
                                        {"request_id": job.request_id,
                                         "error_code": ERROR_CODES["timeout"],
                                         "error": "超时"}, job.peer)
            # 等确认超时后自动复位
            if (self.state == "wait_ack" and cfg.auto_reset_s > 0
                    and now - self._wait_ack_since >= cfg.auto_reset_s):
                self.last_status = "等待确认超时，已自动复位"
                self._write(cfg.done, 0)
                self._write(cfg.done_id, 0)
                self._write(cfg.result, 0)
                self.state = "ready" if not self.in_flight else "busy"
            self._refresh_state()

    def _refresh_state(self) -> None:
        dev = self.device
        if dev is None or not dev.connected:
            self.state = "offline"
            self._ready_written = None        # 重连后重新写一次 Ready
            return
        if self.state not in ("wait_ack", "done") or not self.config.require_ack:
            if self.busy:
                self.state = "busy"
            elif self.state != "done":
                self.state = "ready"
        self._set_ready(self.can_accept())

    def _set_ready(self, flag: bool, force: bool = False) -> None:
        if not force and self._ready_written == flag:
            return
        if self._write(self.config.ready, 1 if flag else 0) or not self.config.ready:
            self._ready_written = flag

    def _reply_error(self, code: int, request_id: int, peer: str) -> None:
        self._send_text(self.config.error_format,
                        {"request_id": request_id, "error_code": code,
                         "error": ERROR_TEXT.get(code, str(code))}, peer)

    # ---- 状态展示 ----
    def status_dict(self) -> dict[str, Any]:
        dev = self.device
        return {
            "id": self.id, "name": self.name, "device": dev.name if dev else self.config.device,
            "flow": self.config.flow, "mode": self.config.mode, "enabled": self.config.enabled,
            "state": self.state, "state_label": STATE_LABELS.get(self.state, self.state),
            "ready": self.can_accept(), "busy": self.busy, "in_flight": len(self.in_flight),
            "request_id": self.last.request_id if self.last else 0,
            "error_code": self.last_error_code,
            "error_text": ERROR_TEXT.get(self.last_error_code, ""),
            "sent": self.last_sent_ok, "acked": self.last_acked,
            "status": self.last_status, **self.counters,
        }

    def shutdown(self) -> None:
        """退出时把对外信号收干净：Ready 置 0、忙标志清掉。"""
        with self.lock:
            self._write(self.config.ready, 0)
            self._write(self.config.busy, 0)
            self.in_flight.clear()
            self.state = "offline"

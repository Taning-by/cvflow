"""通信设备基类。

一个设备就是一条通往外部一方（PLC、机器人、HMI、上位机、MES）的链路。设备负责：

* **收**：把字节流交给 ``Framer`` 切成完整报文，连同**来源对端**一起分发给 ``CommManager``；
* **发**：把上层给的数据发出去，可以指定发给哪个对端（TCP 服务端的某个客户端 / UDP 的某个来源）；
* **状态**：连接与否、对端地址、收发计数、最近通信时间、最后一次错误，全部对界面可见；
* **自检**：``test_connection`` 做一次真实的、对该协议有意义的探测。

身份与配置的分工：``id`` 是稳定的唯一标识（改名不影响流程里的引用），``name`` 只是显示名；
协议参数在 ``config_schema``，连接与重试策略在 ``COMMON_SCHEMA``（所有设备共有），
分帧参数在 ``FRAMING_SCHEMA``（字节流类设备才有）。
"""
from __future__ import annotations

import inspect
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core import events
from ..core.events import EventBus
from .framing import (FRAMING_KINDS, LENGTH_SEMANTICS, Framer, FramingConfig, decode_text,
                      encode_text, frame_for_send, to_hex)

log = logging.getLogger("cvflow.comm")


class CommError(Exception):
    pass


# --------------------------------------------------------------------- 对端
@dataclass
class Peer:
    """一个对端。TCP 服务端的每个客户端、UDP 的每个来源地址各算一个。"""

    id: str                       # 设备内唯一，形如 "192.168.0.20:51234"
    address: str = ""             # 可读地址
    since: float = field(default_factory=time.time)
    last_rx: float = 0.0
    last_tx: float = 0.0
    rx: int = 0
    tx: int = 0

    def to_dict(self) -> dict:
        return {"id": self.id, "address": self.address, "since": self.since,
                "rx": self.rx, "tx": self.tx, "last_rx": self.last_rx, "last_tx": self.last_tx}


#: 所有设备共有的连接与重试配置。
COMMON_SCHEMA: list[tuple[str, str, str, Any, str]] = [
    ("auto_connect", "bool", "自动连接", True, "打开方案并进入运行模式时自动连接；关掉则只能手动连"),
    ("connect_timeout_s", "float", "连接超时（秒）", 3.0, "建立连接/打开端口的等待上限"),
    ("request_timeout_s", "float", "请求超时（秒）", 1.0, "一次读写等待应答的上限（寄存器类设备用）"),
    ("auto_reconnect", "bool", "自动重连", True, "断线后按重连间隔重试；用户手动断开后不再重连"),
    ("reconnect_s", "float", "重连间隔（秒）", 2.0, ""),
    ("heartbeat_s", "float", "心跳间隔（秒）", 0.0, "0 关闭"),
]

#: 字节流类设备（TCP / 串口）的分帧配置。UDP 不需要：一个数据报就是一条报文。
FRAMING_SCHEMA: list[tuple[str, str, str, Any, str]] = [
    ("framing", "enum:" + ",".join(FRAMING_KINDS), "分帧方式", "delimiter",
     "TCP 和串口是字节流，必须指明一条报文到哪里结束"),
    ("terminator", "string", "分隔符", "\\n", "分帧方式为分隔符时使用，可写 \\n、\\r\\n、\\x03"),
    ("frame_length", "int", "定长字节数", 16, "分帧方式为定长时使用"),
    ("length_offset", "int", "长度字段偏移", 0, "分帧方式为长度前缀时：长度字段在报文头里的字节偏移"),
    ("length_size", "int", "长度字段字节数", 2, "1 / 2 / 4"),
    ("length_endian", "enum:big,little", "长度字段字节序", "big", ""),
    ("length_counts", "enum:" + ",".join(LENGTH_SEMANTICS), "长度字段含义", "payload",
     "payload=仅数据区；total=整条报文；after_field=长度字段之后的全部字节"),
    ("header_size", "int", "报文头字节数", 0, "0 表示长度字段偏移 + 字节数"),
    ("keep_header", "bool", "报文保留报文头", False, "交给流程的报文是否带上报文头"),
    ("encoding", "string", "文本编码", "utf-8", "utf-8 / ascii / gbk / gb18030 / latin-1"),
    ("max_frame", "int", "单条报文上限（字节）", 65536, "超过即判为畸形报文"),
    ("rx_buffer_limit", "int", "接收缓冲上限（字节）", 1048576, "超过即清空缓冲并报错"),
    ("frame_timeout_s", "float", "不完整报文超时（秒）", 0.0, "0 关闭；半条报文攒这么久还没收齐就丢弃"),
]

RxCallback = Callable[..., None]


class CommDevice(ABC):
    kind: str = ""
    role: str = ""                     # client / server / master / slave / peer，仅用于显示
    byte_stream: bool = True           # 走 Framer 分帧（UDP 与寄存器类设备为 False）
    supports_peers: bool = False       # 有多个对端，可以指定发给谁
    supports_registers: bool = False   # 有 read_value / write_value 这组接口
    #: (名字, 类型, 标签, 默认值, 说明)；界面按这张表生成协议参数页
    config_schema: list[tuple[str, str, str, Any, str]] = []

    @classmethod
    def schema(cls) -> list[tuple[str, str, str, Any, str]]:
        """协议参数 + 分帧参数（字节流设备）+ 公共连接参数。"""
        out = list(cls.config_schema)
        if cls.byte_stream:
            out += FRAMING_SCHEMA
        return out + COMMON_SCHEMA

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {name: default for name, _, _, default, _ in cls.schema()}

    def __init__(self, name: str, config: dict[str, Any] | None = None, bus: EventBus | None = None,
                 device_id: str | None = None, enabled: bool = True) -> None:
        self.id = device_id or uuid.uuid4().hex[:8]
        self.name = name
        self.config: dict[str, Any] = self.defaults()
        self.config.update(config or {})
        self.bus = bus
        self.enabled = bool(enabled)
        self.connected = False
        self.last_error = ""
        self.manual_stop = False          # 用户手动断开过：不再自动重连
        self.generation = 0               # 每次成功连接 +1；用来识别"断线前的旧任务"
        self.stats = {"tx": 0, "rx": 0, "errors": 0, "dropped": 0}
        self.last_rx = 0.0                # 最近收到数据的时刻（time.time）
        self.last_tx = 0.0
        self.peers: dict[str, Peer] = {}
        self._rx_callbacks: list[tuple[RxCallback, int]] = []
        self._lock = threading.RLock()
        self._framer = Framer(self.framing)
        self._buffer = b""                # 兼容旧代码：仍暴露这个属性

    # ---- 配置读取 ----
    def cfg(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def cfg_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self.config.get(key, default) or 0.0)
        except (TypeError, ValueError):
            return float(default)

    def cfg_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return int(default)

    @property
    def framing(self) -> FramingConfig:
        try:
            return FramingConfig.from_config(self.config)
        except ValueError as e:
            log.warning("%s：分帧配置有误（%s），退回分隔符 \\n", self.name, e)
            return FramingConfig()

    @property
    def terminator(self) -> bytes:
        """兼容旧接口：分隔符模式下的分隔符字节。"""
        return self.framing.delimiter_bytes

    @property
    def encoding(self) -> str:
        return str(self.config.get("encoding", "utf-8") or "utf-8")

    @property
    def auto_connect(self) -> bool:
        return bool(self.config.get("auto_connect", True))

    @property
    def auto_reconnect(self) -> bool:
        return bool(self.config.get("auto_reconnect", True)) and not self.manual_stop

    @property
    def connect_timeout(self) -> float:
        return max(0.1, self.cfg_float("connect_timeout_s", 3.0))

    @property
    def request_timeout(self) -> float:
        return max(0.05, self.cfg_float("request_timeout_s", 1.0))

    @property
    def reconnect_interval(self) -> float:
        return max(0.05, self.cfg_float("reconnect_s", 2.0))

    @property
    def heartbeat_interval(self) -> float:
        return max(0.0, self.cfg_float("heartbeat_s", 0.0))

    # ---- 分帧 ----
    def reset_framer(self) -> None:
        self._framer = Framer(self.framing)
        self._buffer = b""

    def _feed(self, chunk: bytes, peer: Peer | None = None) -> None:
        """把收到的字节交给分帧器，切出的每条完整报文分发一次。"""
        if not self.byte_stream:
            if chunk:
                self._dispatch_rx(chunk, peer)
            return
        frames = self._framer.feed(chunk)
        self._buffer = bytes(self._framer._buf)      # 兼容旧字段
        for err in self._framer.take_errors():
            self._error(err)
        for frame in frames:
            self._dispatch_rx(frame, peer)

    def check_frame_timeout(self) -> None:
        """由管理器的后台定时调用：丢弃攒太久的不完整报文。"""
        if self.byte_stream:
            for err in self._framer.check_timeout():
                self._error(err)

    # ---- 生命周期 ----
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def _send_bytes(self, data: bytes, peer: Peer | None = None) -> None: ...

    def open(self) -> None:
        """手动连接：清掉"用户已手动断开"的标记，让自动重连重新生效。"""
        self.manual_stop = False
        self.connect()

    def close(self) -> None:
        """手动断开：之后不再自动重连，直到再次手动连接。"""
        self.manual_stop = True
        self.disconnect()

    # ---- 发送 ----
    def send(self, data: bytes | str, peer: str | Peer | None = None, raw: bool = False) -> bool:
        """发送一条报文。

        ``peer`` 指定对端（TCP 服务端的某个客户端、UDP 的某个来源）；留空按设备默认行为
        （TCP 服务端广播、UDP 发给配置的目标地址）。``raw`` 为真时不补分隔符/长度前缀。
        """
        if isinstance(data, str):
            data = encode_text(data, self.encoding)
        target = self._resolve_peer(peer)
        if not raw and self.byte_stream:
            try:
                data = frame_for_send(data, self.framing)
            except Exception as e:
                self._error(f"组帧失败：{e}")
                return False
        try:
            with self._lock:
                self._send_bytes(data, target)
            self.stats["tx"] += 1
            self.last_tx = time.time()
            if target is not None:
                target.tx += 1
                target.last_tx = self.last_tx
            self._emit(events.COMM_SENT, device=self.name, device_id=self.id, data=data,
                       text=self._text(data), hex=to_hex(data), peer=target.id if target else "")
            return True
        except Exception as e:
            self._error(f"发送失败：{e}")
            return False

    def _resolve_peer(self, peer: str | Peer | None) -> Peer | None:
        if peer is None or peer == "":
            return None
        if isinstance(peer, Peer):
            return peer
        found = self.peers.get(str(peer))
        if found is None:
            raise CommError(f"{self.name}：没有对端 {peer!r}（当前 {len(self.peers)} 个）")
        return found

    # ---- 接收 ----
    def on_receive(self, cb: RxCallback) -> None:
        """注册接收回调。回调可以写 ``(device, data)`` 或 ``(device, data, peer)``，都支持。"""
        arity = 3
        try:
            params = [p for p in inspect.signature(cb).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if any(p.kind == p.VAR_POSITIONAL for p in inspect.signature(cb).parameters.values()):
                arity = 3
            else:
                arity = min(3, len(params))
        except (TypeError, ValueError):
            pass
        self._rx_callbacks.append((cb, arity))

    def _dispatch_rx(self, data: bytes, peer: Peer | None = None) -> None:
        self.stats["rx"] += 1
        self.last_rx = time.time()
        if peer is not None:
            peer.rx += 1
            peer.last_rx = self.last_rx
        self._emit(events.COMM_RECEIVED, device=self.name, device_id=self.id, data=data,
                   text=self._text(data), hex=to_hex(data), peer=peer.id if peer else "")
        for cb, arity in list(self._rx_callbacks):
            try:
                cb(self, data, peer) if arity >= 3 else cb(self, data)
            except Exception:
                log.exception("%s 的接收回调出错", self.name)

    def _text(self, data: bytes) -> str:
        return decode_text(data, self.encoding)

    # ---- 对端管理 ----
    def _add_peer(self, peer_id: str, address: str = "") -> Peer:
        with self._lock:
            p = self.peers.get(peer_id)
            if p is None:
                p = Peer(id=peer_id, address=address or peer_id)
                self.peers[peer_id] = p
        return p

    def _drop_peer(self, peer_id: str) -> None:
        with self._lock:
            self.peers.pop(peer_id, None)

    def peer_text(self) -> str:
        """界面"对端"一列显示什么。"""
        if self.supports_peers:
            if not self.peers:
                return "无对端"
            if len(self.peers) == 1:
                return next(iter(self.peers.values())).address
            return f"{len(self.peers)} 个对端"
        host, port = self.config.get("host"), self.config.get("port")
        if host:
            return f"{host}:{port}" if port else str(host)
        if self.config.get("port"):
            return str(self.config.get("port"))
        return ""

    # ---- 状态 ----
    def _set_connected(self, flag: bool, reason: str = "") -> None:
        if flag == self.connected:
            return
        self.connected = flag
        if flag:
            self.last_error = ""
            self.generation += 1
            log.info("%s 已连接", self.name)
            self._emit(events.COMM_CONNECTED, device=self.name, device_id=self.id,
                       peer=self.peer_text(), generation=self.generation)
        else:
            self.peers.clear()
            log.info("%s 已断开 %s", self.name, reason)
            self._emit(events.COMM_DISCONNECTED, device=self.name, device_id=self.id, reason=reason)

    def _error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.last_error = msg
        log.warning("%s: %s", self.name, msg)
        self._emit(events.COMM_ERROR, device=self.name, device_id=self.id, error=msg)

    def _emit(self, event: str, **payload) -> None:
        if self.bus is not None:
            self.bus.emit(event, **payload)

    # ---- 连接测试 / 心跳 ----
    def test_connection(self) -> tuple[bool, str]:
        """默认实现：启动连接并等到连接超时为止。各协议应覆盖成真正有意义的探测。"""
        try:
            self.connect()
        except Exception as e:
            return False, f"{self.name}：{e}"
        end = time.time() + self.connect_timeout
        while time.time() < end and not self.connected:
            time.sleep(0.05)
        if self.connected:
            return True, f"{self.name}：已连接"
        return False, f"{self.name}：{self.last_error or '超时未连接'}"

    def heartbeat(self) -> None:
        """心跳：寄存器类设备自增 heartbeat_address；字节流设备发送 heartbeat_text。"""
        if not self.connected:
            return
        addr = self.cfg_int("heartbeat_address", -1)
        if addr >= 0 and hasattr(self, "write_registers") and hasattr(self, "read_registers"):
            cur = self.read_registers(addr, 1)[0]            # type: ignore[attr-defined]
            self.write_registers(addr, [(cur + 1) & 0xFFFF])  # type: ignore[attr-defined]
            return
        text = str(self.config.get("heartbeat_text", "") or "")
        if text:
            from .framing import unescape
            self.send(unescape(text), raw=True)

    # ---- 序列化 ----
    def info(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind, "role": self.role,
                "enabled": self.enabled, "connected": self.connected, "peer": self.peer_text(),
                "peers": len(self.peers), "error": self.last_error,
                "last_rx": self.last_rx, "last_tx": self.last_tx,
                "manual_stop": self.manual_stop, **self.stats}

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "enabled": self.enabled, "config": dict(self.config)}

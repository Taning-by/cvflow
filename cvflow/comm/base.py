"""Communication device base class.

A device is a connection to one external party (PLC, robot, HMI, MES). It
delivers received frames to the ``CommManager`` which matches them against
receive rules (trigger a flow, set a variable) and it transports frames built by
send rules or nodes.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..core import events
from ..core.events import EventBus

log = logging.getLogger("cvflow.comm")


class CommError(Exception):
    pass


RxCallback = Callable[["CommDevice", bytes], None]


class CommDevice(ABC):
    kind: str = ""
    # (name, kind, label, default, description) - the UI builds the device dialog from this
    config_schema: list[tuple[str, str, str, Any, str]] = []

    def __init__(self, name: str, config: dict[str, Any] | None = None, bus: EventBus | None = None) -> None:
        self.name = name
        self.config: dict[str, Any] = {k: d for k, _, _, d, _ in self.config_schema}
        self.config.update(config or {})
        self.bus = bus
        self.connected = False
        self.last_error = ""
        self.stats = {"tx": 0, "rx": 0, "errors": 0}
        self._rx_callbacks: list[RxCallback] = []
        self._lock = threading.RLock()
        self._buffer = b""

    # ---- framing ----
    @property
    def terminator(self) -> bytes:
        t = str(self.config.get("terminator", "\\n"))
        return t.encode().decode("unicode_escape").encode("latin-1")

    @property
    def encoding(self) -> str:
        return str(self.config.get("encoding", "utf-8"))

    def _feed(self, chunk: bytes) -> None:
        """Split incoming bytes into frames by terminator and dispatch each one."""
        term = self.terminator
        if not term:
            self._dispatch_rx(chunk)
            return
        self._buffer += chunk
        while True:
            idx = self._buffer.find(term)
            if idx < 0:
                break
            frame, self._buffer = self._buffer[:idx], self._buffer[idx + len(term):]
            if frame:
                self._dispatch_rx(frame)
        if len(self._buffer) > 1_000_000:  # protect against a peer that never terminates
            self._buffer = b""

    # ---- lifecycle ----
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def _send_bytes(self, data: bytes) -> None: ...

    def send(self, data: bytes | str) -> bool:
        if isinstance(data, str):
            data = data.encode(self.encoding, errors="replace")
        try:
            with self._lock:
                self._send_bytes(data)
            self.stats["tx"] += 1
            self._emit(events.COMM_SENT, device=self.name, data=data, text=self._text(data))
            return True
        except Exception as e:
            self._error(f"发送失败：{e}")
            return False

    # ---- receive side ----
    def on_receive(self, cb: RxCallback) -> None:
        self._rx_callbacks.append(cb)

    def _dispatch_rx(self, data: bytes) -> None:
        self.stats["rx"] += 1
        self._emit(events.COMM_RECEIVED, device=self.name, data=data, text=self._text(data))
        for cb in list(self._rx_callbacks):
            try:
                cb(self, data)
            except Exception:
                log.exception("%s 的接收回调出错", self.name)

    def _text(self, data: bytes) -> str:
        try:
            return data.decode(self.encoding)
        except UnicodeDecodeError:
            return data.hex(" ")

    # ---- state ----
    def _set_connected(self, flag: bool, reason: str = "") -> None:
        if flag == self.connected:
            return
        self.connected = flag
        if flag:
            self.last_error = ""
            log.info("%s 已连接", self.name)
            self._emit(events.COMM_CONNECTED, device=self.name)
        else:
            log.info("%s 已断开 %s", self.name, reason)
            self._emit(events.COMM_DISCONNECTED, device=self.name, reason=reason)

    def _error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.last_error = msg
        log.warning("%s: %s", self.name, msg)
        self._emit(events.COMM_ERROR, device=self.name, error=msg)

    def _emit(self, event: str, **payload) -> None:
        if self.bus is not None:
            self.bus.emit(event, **payload)

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "connected": self.connected, "error": self.last_error, **self.stats}

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "config": dict(self.config)}

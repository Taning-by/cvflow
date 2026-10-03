"""西门子 S7 PLC（S7-200 Smart / 300 / 400 / 1200 / 1500），基于 python-snap7。

以 DB 块为交换区：按 ``poll_ms`` 轮询 ``db_number`` 中从 ``watch_offset`` 开始的 ``watch_bytes`` 字节，
按 16 位字（大端）检测变化并产生上升沿触发；``write_value`` 的地址是 DB 内的字节偏移。
S7-1200/1500 需要在 PLC 上允许 PUT/GET 访问，并把 DB 的“优化块访问”关闭。
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Any

from .base import CommDevice, CommError
from .modbus import _RegisterWatcher

_FMT = {"int16": ">h", "uint16": ">H", "int32": ">i", "uint32": ">I", "float32": ">f"}


class S7Device(CommDevice, _RegisterWatcher):
    kind = "s7"
    config_schema = [
        ("host", "string", "PLC 地址", "192.168.0.1", ""),
        ("rack", "int", "机架号", 0, "1200/1500 为 0"),
        ("slot", "int", "插槽号", 1, "1200/1500 为 1，300 为 2"),
        ("port", "int", "端口", 102, ""),
        ("db_number", "int", "DB 块号", 1, "作为交换区的 DB"),
        ("poll_ms", "int", "轮询间隔（毫秒）", 50, "0 表示不轮询"),
        ("watch_offset", "int", "监视起始字节", 0, ""),
        ("watch_bytes", "int", "监视字节数", 16, "按 2 字节一个字检测变化"),
        ("busy_address", "int", "忙标志字节偏移", -1, "-1 关闭；触发后写 1，流程结束写 0"),
        ("trigger_reset", "bool", "触发后自动复位触发字", False, ""),
        ("heartbeat_address", "int", "心跳字节偏移", -1, "-1 关闭；按心跳间隔自增"),
        ("heartbeat_s", "float", "心跳间隔（秒）", 0.0, "0 关闭"),
    ]

    def __init__(self, name, config=None, bus=None):
        CommDevice.__init__(self, name, config, bus)
        _RegisterWatcher.__init__(self)
        self._client = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last: list[int] | None = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import snap7
            except ImportError as e:
                raise CommError("西门子 S7 支持需要 `pip install python-snap7`") from e
            c = snap7.client.Client()
            c.connect(str(self.config["host"]), int(self.config["rack"]), int(self.config["slot"]), int(self.config.get("port", 102)))
            self._client = c
        return self._client

    def _db(self) -> int:
        return int(self.config["db_number"])

    # ---- 读写 ----
    def read_bytes(self, offset: int, size: int) -> bytes:
        with self._lock:
            data = bytes(self._ensure_client().db_read(self._db(), offset, size))
            self.stats["rx"] += 1
            return data

    def write_bytes(self, offset: int, data: bytes) -> None:
        with self._lock:
            self._ensure_client().db_write(self._db(), offset, bytearray(data))
            self.stats["tx"] += 1

    def read_registers(self, address: int, count: int = 1) -> list[int]:
        raw = self.read_bytes(address, count * 2)
        return list(struct.unpack(f">{count}H", raw))

    def write_registers(self, address: int, values: list[int]) -> None:
        self.write_bytes(address, struct.pack(f">{len(values)}H", *[v & 0xFFFF for v in values]))

    def write_value(self, address: int, value: Any, kind: str = "int16") -> None:
        if kind == "bool":
            self.write_bytes(int(address), bytes([1 if value else 0]))
            return
        if kind in ("int16", "uint16", "int32", "uint32"):
            value = int(round(float(value)))
        self.write_bytes(int(address), struct.pack(_FMT[kind], value))

    def read_value(self, address: int, kind: str = "int16") -> Any:
        if kind == "bool":
            return bool(self.read_bytes(int(address), 1)[0])
        return struct.unpack(_FMT[kind], self.read_bytes(int(address), struct.calcsize(_FMT[kind])))[0]

    # ---- 生命周期 ----
    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"s7-{self.name}", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3.0)
            self._thread = None
        self._close()
        self._set_connected(False, "已关闭")

    def _close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.disconnect()
                    self._client.destroy()
                except Exception:
                    pass
                self._client = None

    def _send_bytes(self, data: bytes) -> None:
        raise CommError("S7 设备不发送原始报文，请使用 write_value()")

    def test_connection(self) -> tuple[bool, str]:
        t0 = time.perf_counter()
        try:
            with self._lock:
                c = self._ensure_client()
                state = c.get_cpu_state()
            v = self.read_registers(int(self.config.get("watch_offset", 0)), 1)[0]
            return True, (f"{self.config['host']} 机架 {self.config['rack']} 插槽 {self.config['slot']}：CPU {state}，"
                          f"DB{self._db()}.DBW{self.config.get('watch_offset', 0)}={v}，耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")
        except Exception as e:
            self._close()
            return False, f"{self.config['host']} 连接失败：{e}"

    def _loop(self) -> None:
        poll = int(self.config.get("poll_ms", 50)) / 1000.0
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._ensure_client()
                if poll <= 0:
                    self._set_connected(True)
                    self._stop.wait(0.5)
                    continue
                off, nbytes = int(self.config["watch_offset"]), int(self.config["watch_bytes"])
                regs = self.read_registers(off, max(1, nbytes // 2))
                if self._last is not None:
                    for i, (o, n) in enumerate(zip(self._last, regs)):
                        if o != n:
                            self._notify_reg(off + 2 * i, o, n)
                self._last = regs
                self._set_connected(True)   # 基线快照建立之后才对外显示“已连接”
                self._stop.wait(poll)
            except Exception as e:
                if self.connected or not self.last_error:
                    self._error(str(e))
                self._set_connected(False, str(e))
                self._last = None
                self._close()
                self._stop.wait(1.0)

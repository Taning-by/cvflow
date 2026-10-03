"""RS-232/485 serial device via pyserial."""
from __future__ import annotations

import threading

from .base import CommDevice, CommError


class SerialDevice(CommDevice):
    kind = "serial"
    config_schema = [
        ("port", "string", "Port", "/dev/ttyUSB0", "e.g. COM3 or /dev/ttyUSB0"),
        ("baudrate", "int", "Baud rate", 9600, ""),
        ("bytesize", "int", "Data bits", 8, ""),
        ("parity", "enum:N,E,O", "Parity", "N", ""),
        ("stopbits", "float", "Stop bits", 1.0, ""),
        ("terminator", "string", "Terminator", "\\r\\n", ""),
        ("encoding", "string", "Encoding", "utf-8", ""),
    ]

    def __init__(self, name, config=None, bus=None):
        super().__init__(name, config, bus)
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def connect(self) -> None:
        if self._ser:
            return
        try:
            import serial
        except ImportError as e:  # pragma: no cover
            raise CommError("未安装 pyserial") from e
        try:
            self._ser = serial.Serial(port=self.config["port"], baudrate=int(self.config["baudrate"]),
                                      bytesize=int(self.config["bytesize"]), parity=str(self.config["parity"]),
                                      stopbits=float(self.config["stopbits"]), timeout=0.2)
        except Exception as e:
            self._error(str(e))
            raise CommError(str(e)) from e
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"serial-{self.name}", daemon=True)
        self._thread.start()
        self._set_connected(True)

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
            self._thread = None
        if self._ser:
            try:
                self._ser.close()
            finally:
                self._ser = None
        self._set_connected(False, "已关闭")

    def _loop(self) -> None:
        while not self._stop.is_set() and self._ser:
            try:
                n = self._ser.in_waiting
                data = self._ser.read(n if n else 1)
            except Exception as e:
                self._error(f"读取失败：{e}")
                self._set_connected(False, str(e))
                break
            if data:
                self._feed(data)

    def _send_bytes(self, data: bytes) -> None:
        if not self._ser:
            raise CommError("未打开")
        self._ser.write(data)

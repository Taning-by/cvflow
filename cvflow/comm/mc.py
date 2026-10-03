"""三菱 MC 协议（QnA 兼容 3E 帧，二进制），用于三菱 Q/L/iQ-R/FX5 以及汇川、基恩士等兼容 PLC。

* ``McDevice``：主站。按 ``poll_ms`` 轮询一段字软元件（默认 D）检测变化，支持上升沿触发；
  ``write_value`` 支持 int16/uint16/int32/uint32/float32/bool（位软元件）。地址写法 ``D100``、``M20``，
  纯数字按监视软元件解释。
* ``McSimulatorServer``：一个最小的 MC 3E 二进制从站，用来在没有 PLC 时做联调和自动化测试。
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any

from .base import CommDevice, CommError
from .modbus import _RegisterWatcher, registers_to_value, value_to_registers

DEVICE_CODES = {"D": 0xA8, "W": 0xB4, "R": 0xAF, "ZR": 0xB0, "SD": 0xA9, "TN": 0xC2, "CN": 0xC5,
                "M": 0x90, "X": 0x9C, "Y": 0x9D, "B": 0xA0, "L": 0x92, "F": 0x93, "V": 0x94, "SM": 0x91, "SB": 0xA1}
BIT_DEVICES = {"M", "X", "Y", "B", "L", "F", "V", "SM", "SB"}
_HEX_DEVICES = {"X", "Y", "B", "W", "SB", "ZR"}


def parse_address(text: str | int, default_device: str = "D") -> tuple[str, int]:
    """``"D100"`` -> ("D", 100)；``100`` -> (default_device, 100)。X/Y/B/W 的编号按十六进制解析。"""
    if isinstance(text, int):
        return default_device.upper(), text
    t = str(text).strip().upper()
    i = 0
    while i < len(t) and t[i].isalpha():
        i += 1
    dev, num = t[:i], t[i:]
    if not dev:
        dev = default_device.upper()
    if dev not in DEVICE_CODES:
        raise CommError(f"未知的软元件 {dev!r}")
    if not num:
        raise CommError(f"地址 {text!r} 缺少编号")
    return dev, int(num, 16 if dev in _HEX_DEVICES else 10)


def _frame(network: int, pc: int, io: int, station: int, command: int, sub: int, body: bytes, timer: int = 0x0010) -> bytes:
    data = struct.pack("<HHH", timer, command, sub) + body
    return struct.pack("<BBBBHBH", 0x50, 0x00, network, pc, io, station, len(data)) + data


def _device_spec(dev: str, number: int, points: int) -> bytes:
    return struct.pack("<I", number)[:3] + bytes([DEVICE_CODES[dev]]) + struct.pack("<H", points)


class McDevice(CommDevice, _RegisterWatcher):
    kind = "mc"
    config_schema = [
        ("host", "string", "PLC 地址", "192.168.3.39", ""),
        ("port", "int", "端口", 5000, "三菱内置以太网常用 5000/5002（需在 PLC 上开放 MC 二进制协议）"),
        ("network", "int", "网络号", 0, ""),
        ("pc", "int", "PC 号", 255, ""),
        ("io", "int", "请求目标模块 I/O", 1023, "0x3FF = 本站 CPU"),
        ("station", "int", "请求目标模块站号", 0, ""),
        ("timeout_s", "float", "超时（秒）", 1.0, ""),
        ("poll_ms", "int", "轮询间隔（毫秒）", 50, "0 表示不轮询"),
        ("watch_device", "string", "监视软元件", "D", "D / W / R / ZR"),
        ("watch_address", "int", "监视起始地址", 0, ""),
        ("watch_count", "int", "监视字数", 8, ""),
        ("busy_address", "int", "忙标志地址", -1, "-1 关闭；触发后写 1，流程结束写 0（监视软元件）"),
        ("trigger_reset", "bool", "触发后自动复位触发字", False, ""),
        ("heartbeat_address", "int", "心跳地址", -1, "-1 关闭；按心跳间隔自增（监视软元件）"),
        ("heartbeat_s", "float", "心跳间隔（秒）", 0.0, "0 关闭"),
    ]

    def __init__(self, name, config=None, bus=None):
        CommDevice.__init__(self, name, config, bus)
        _RegisterWatcher.__init__(self)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last: list[int] | None = None

    # ---- 底层收发 ----
    def _ensure_sock(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection((self.config["host"], int(self.config["port"])), timeout=float(self.config.get("timeout_s", 1.0)))
            s.settimeout(float(self.config.get("timeout_s", 1.0)))
            self._sock = s
        return self._sock

    def _request(self, command: int, sub: int, body: bytes) -> bytes:
        with self._lock:
            s = self._ensure_sock()
            frame = _frame(int(self.config["network"]), int(self.config["pc"]), int(self.config["io"]),
                           int(self.config["station"]), command, sub, body)
            s.sendall(frame)
            head = self._recv_exact(s, 9)
            if head[0] != 0xD0:
                raise CommError("MC：应答子头错误")
            length = struct.unpack("<H", head[7:9])[0]
            rest = self._recv_exact(s, length)
            end_code = struct.unpack("<H", rest[:2])[0]
            if end_code != 0:
                raise CommError(f"MC：PLC 返回结束代码 0x{end_code:04X}")
            self.stats["rx"] += 1
            return rest[2:]

    @staticmethod
    def _recv_exact(s: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise CommError("MC：连接已关闭")
            buf += chunk
        return buf

    # ---- 读写 ----
    def read_words(self, device: str, address: int, count: int) -> list[int]:
        data = self._request(0x0401, 0x0000, _device_spec(device, address, count))
        return list(struct.unpack(f"<{count}H", data[: count * 2]))

    def write_words(self, device: str, address: int, values: list[int]) -> None:
        body = _device_spec(device, address, len(values)) + struct.pack(f"<{len(values)}H", *[v & 0xFFFF for v in values])
        self._request(0x1401, 0x0000, body)
        self.stats["tx"] += 1

    def read_bits(self, device: str, address: int, count: int) -> list[bool]:
        data = self._request(0x0401, 0x0001, _device_spec(device, address, count))
        bits = []
        for i in range(count):
            b = data[i // 2]
            bits.append(bool((b >> 4) if i % 2 == 0 else (b & 0x0F)))
        return bits

    def write_bits(self, device: str, address: int, values: list[bool]) -> None:
        packed = bytearray((len(values) + 1) // 2)
        for i, v in enumerate(values):
            if v:
                packed[i // 2] |= 0x10 if i % 2 == 0 else 0x01
        self._request(0x1401, 0x0001, _device_spec(device, address, len(values)) + bytes(packed))
        self.stats["tx"] += 1

    def _watch_dev(self) -> str:
        return str(self.config.get("watch_device", "D")).upper()

    def read_registers(self, address: int, count: int = 1) -> list[int]:
        return self.read_words(self._watch_dev(), address, count)

    def write_registers(self, address: int, values: list[int]) -> None:
        self.write_words(self._watch_dev(), address, values)

    def write_value(self, address: str | int, value: Any, kind: str = "int16") -> None:
        dev, num = parse_address(address, self._watch_dev())
        if dev in BIT_DEVICES or kind == "bool":
            if dev in BIT_DEVICES:
                self.write_bits(dev, num, [bool(value)])
            else:
                self.write_words(dev, num, [1 if value else 0])
            return
        regs = value_to_registers(value, kind)
        if len(regs) == 2:  # MC 的 32 位值低字在前
            regs = [regs[1], regs[0]]
        self.write_words(dev, num, regs)

    def read_value(self, address: str | int, kind: str = "int16") -> Any:
        dev, num = parse_address(address, self._watch_dev())
        if dev in BIT_DEVICES:
            return self.read_bits(dev, num, 1)[0]
        if kind == "bool":
            return bool(self.read_words(dev, num, 1)[0])
        n = 2 if kind in ("int32", "uint32", "float32") else 1
        regs = self.read_words(dev, num, n)
        if n == 2:
            regs = [regs[1], regs[0]]
        return registers_to_value(regs, kind)

    # ---- 生命周期 ----
    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"mc-{self.name}", daemon=True)
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
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def _send_bytes(self, data: bytes) -> None:
        raise CommError("MC 设备不发送原始报文，请使用 write_value()")

    def test_connection(self) -> tuple[bool, str]:
        t0 = time.perf_counter()
        try:
            v = self.read_registers(int(self.config.get("watch_address", 0)), 1)[0]
            return True, f"{self.config['host']}:{self.config['port']} 可用，{self._watch_dev()}{self.config.get('watch_address', 0)}={v}，耗时 {(time.perf_counter() - t0) * 1000:.0f} ms"
        except Exception as e:
            self._close()
            return False, f"{self.config['host']}:{self.config['port']} 连接失败：{e}"

    def _loop(self) -> None:
        poll = int(self.config.get("poll_ms", 50)) / 1000.0
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._ensure_sock()
                if poll <= 0:
                    self._set_connected(True)
                    self._stop.wait(0.5)
                    continue
                addr, count = int(self.config["watch_address"]), int(self.config["watch_count"])
                regs = self.read_registers(addr, count)
                if self._last is not None:
                    for i, (o, n) in enumerate(zip(self._last, regs)):
                        if o != n:
                            self._notify_reg(addr + i, o, n)
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


class McSimulatorServer:
    """最小 MC 3E 二进制从站：字软元件与位软元件各一张表，支持批量读/写（0x0401 / 0x1401）。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host, self.port = host, port
        self.words: dict[tuple[str, int], int] = {}
        self.bits: dict[tuple[str, int], bool] = {}
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lock = threading.RLock()

    @property
    def bound_port(self) -> int:
        return self._srv.getsockname()[1] if self._srv else self.port

    def start(self) -> "McSimulatorServer":
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(4)
        s.settimeout(0.5)
        self._srv = s
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._srv:
            self._srv.close()
            self._srv = None
        if self._thread:
            self._thread.join(2.0)

    def set_word(self, device: str, address: int, value: int) -> None:
        with self.lock:
            self.words[(device.upper(), address)] = value & 0xFFFF

    def get_word(self, device: str, address: int) -> int:
        with self.lock:
            return self.words.get((device.upper(), address), 0)

    def _accept(self) -> None:
        while not self._stop.is_set() and self._srv:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while len(buf) >= 9:
                    length = struct.unpack("<H", buf[7:9])[0]
                    if len(buf) < 9 + length:
                        break
                    req, buf = buf[: 9 + length], buf[9 + length:]
                    conn.sendall(self._handle(req))
        except OSError:
            pass
        finally:
            conn.close()

    def _handle(self, req: bytes) -> bytes:
        network, pc, io, station = req[2], req[3], struct.unpack("<H", req[4:6])[0], req[6]
        timer, command, sub = struct.unpack("<HHH", req[9:15])
        body = req[15:]
        number = struct.unpack("<I", body[0:3] + b"\x00")[0]
        code = body[3]
        points = struct.unpack("<H", body[4:6])[0]
        dev = next((k for k, v in DEVICE_CODES.items() if v == code), None)
        end, data = 0, b""
        if dev is None:
            end = 0xC059
        elif command == 0x0401 and sub == 0x0000:
            with self.lock:
                data = struct.pack(f"<{points}H", *[self.words.get((dev, number + i), 0) for i in range(points)])
        elif command == 0x0401 and sub == 0x0001:
            out = bytearray((points + 1) // 2)
            with self.lock:
                for i in range(points):
                    if self.bits.get((dev, number + i), False):
                        out[i // 2] |= 0x10 if i % 2 == 0 else 0x01
            data = bytes(out)
        elif command == 0x1401 and sub == 0x0000:
            vals = struct.unpack(f"<{points}H", body[6:6 + points * 2])
            with self.lock:
                for i, v in enumerate(vals):
                    self.words[(dev, number + i)] = v
        elif command == 0x1401 and sub == 0x0001:
            raw = body[6:6 + (points + 1) // 2]
            with self.lock:
                for i in range(points):
                    b = raw[i // 2]
                    self.bits[(dev, number + i)] = bool((b >> 4) if i % 2 == 0 else (b & 0x0F))
        else:
            end = 0xC059  # 不支持的命令
        payload = struct.pack("<H", end) + data
        return struct.pack("<BBBBHBH", 0xD0, 0x00, network, pc, io, station, len(payload)) + payload

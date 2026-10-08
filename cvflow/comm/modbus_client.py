"""Modbus 主站协议层：报文拼装/解析 + TCP 与 RTU 两种传输。

为什么自己实现而不用 pymodbus：
* **统一调度**。轮询和流程里的主动读写必须排在同一条队列上，否则 RTU 半双工会撞车、TCP 会
  出现"拿到上一次请求的应答"这种错配。自己掌握传输层才能把这件事做干净。
* **可预期的依赖**。pymodbus 的客户端 API 在大版本间变过几次，产线软件不该跟着它改。
  pymodbus 现在只作为**测试对端**（第三方独立实现）使用，不是运行时依赖。

支持的功能码：01 读线圈、02 读离散输入、03 读保持寄存器、04 读输入寄存器、
05 写单个线圈、06 写单个寄存器、15 写多个线圈、16 写多个寄存器。
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from abc import ABC, abstractmethod

from .base import CommError

EXCEPTION_TEXT = {
    0x01: "非法功能码（从站不支持该功能码）",
    0x02: "非法数据地址（地址或数量超出从站的数据区）",
    0x03: "非法数据值",
    0x04: "从站设备故障",
    0x05: "已确认（从站正在处理，需稍后轮询）",
    0x06: "从站忙",
    0x08: "存储奇偶校验错",
    0x0A: "网关路径不可用",
    0x0B: "网关目标设备无响应",
}

MAX_READ_BITS = 2000
MAX_READ_WORDS = 125
MAX_WRITE_BITS = 1968
MAX_WRITE_WORDS = 123


class ModbusException(CommError):
    """从站回了异常响应（功能码最高位置 1）。"""

    def __init__(self, fc: int, code: int) -> None:
        self.fc = fc
        self.code = code
        super().__init__(f"Modbus 异常响应：功能码 {fc}，异常码 {code}（{EXCEPTION_TEXT.get(code, '未知')}）")


class ModbusTimeout(CommError):
    pass


# --------------------------------------------------------------------- CRC
_CRC_TABLE: list[int] = []


def _build_crc_table() -> None:
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        _CRC_TABLE.append(crc)


_build_crc_table()


def crc16(data: bytes) -> int:
    """Modbus RTU 的 CRC-16（多项式 0xA001，初值 0xFFFF，低字节先发）。"""
    crc = 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


# --------------------------------------------------------------------- PDU
def pdu_read(fc: int, address: int, count: int) -> bytes:
    limit = MAX_READ_BITS if fc in (1, 2) else MAX_READ_WORDS
    if not 1 <= count <= limit:
        raise CommError(f"功能码 {fc} 单次最多读 {limit} 个，请求了 {count} 个")
    _check_address(address, count)
    return struct.pack(">BHH", fc, address, count)


def pdu_write_coil(address: int, value: bool) -> bytes:
    _check_address(address, 1)
    return struct.pack(">BHH", 5, address, 0xFF00 if value else 0x0000)


def pdu_write_register(address: int, value: int) -> bytes:
    _check_address(address, 1)
    return struct.pack(">BHH", 6, address, int(value) & 0xFFFF)


def pdu_write_coils(address: int, bits: list[bool]) -> bytes:
    if not 1 <= len(bits) <= MAX_WRITE_BITS:
        raise CommError(f"功能码 15 单次最多写 {MAX_WRITE_BITS} 个线圈，请求了 {len(bits)} 个")
    _check_address(address, len(bits))
    nbytes = (len(bits) + 7) // 8
    raw = bytearray(nbytes)
    for i, b in enumerate(bits):
        if b:
            raw[i // 8] |= 1 << (i % 8)
    return struct.pack(">BHHB", 15, address, len(bits), nbytes) + bytes(raw)


def pdu_write_registers(address: int, values: list[int]) -> bytes:
    if not 1 <= len(values) <= MAX_WRITE_WORDS:
        raise CommError(f"功能码 16 单次最多写 {MAX_WRITE_WORDS} 个寄存器，请求了 {len(values)} 个")
    _check_address(address, len(values))
    return (struct.pack(">BHHB", 16, address, len(values), len(values) * 2)
            + struct.pack(f">{len(values)}H", *[int(v) & 0xFFFF for v in values]))


def _check_address(address: int, count: int) -> None:
    if address < 0 or address > 0xFFFF:
        raise CommError(f"地址 {address} 超出 0…65535")
    if address + count - 1 > 0xFFFF:
        raise CommError(f"地址 {address} 起 {count} 个越过 65535")


def check_response(fc: int, pdu: bytes) -> bytes:
    """校验应答的功能码，返回功能码之后的数据部分。异常响应抛 ModbusException。"""
    if not pdu:
        raise CommError("应答为空")
    got = pdu[0]
    if got == (fc | 0x80):
        if len(pdu) < 2:
            raise CommError("异常响应缺少异常码")
        raise ModbusException(fc, pdu[1])
    if got != fc:
        raise CommError(f"应答功能码不符：请求 {fc}，应答 {got}")
    return pdu[1:]


def parse_bits(pdu: bytes, count: int) -> list[bool]:
    body = pdu[1:]
    if not body:
        raise CommError("读位应答缺少字节数")
    nbytes = body[0]
    raw = body[1:1 + nbytes]
    if len(raw) < nbytes:
        raise CommError(f"读位应答声明 {nbytes} 字节，实际 {len(raw)} 字节")
    return [bool(raw[i // 8] >> (i % 8) & 1) for i in range(count)]


def parse_words(pdu: bytes, count: int) -> list[int]:
    body = pdu[1:]
    if not body:
        raise CommError("读寄存器应答缺少字节数")
    nbytes = body[0]
    raw = body[1:1 + nbytes]
    if nbytes != count * 2 or len(raw) < nbytes:
        raise CommError(f"读寄存器应答声明 {nbytes} 字节（期望 {count * 2}），实际 {len(raw)} 字节")
    return list(struct.unpack(f">{count}H", raw))


def expected_rtu_length(fc: int, head: bytes) -> int:
    """RTU 没有长度字段：根据功能码和已读到的头部算出整条应答该有多少字节。"""
    if fc & 0x80:
        return 5                                  # 站号 + 功能码 + 异常码 + CRC2
    if fc in (1, 2, 3, 4):
        if len(head) < 3:
            return -1                             # 还需要字节数那一字节
        return 3 + head[2] + 2
    if fc in (5, 6, 15, 16):
        return 8
    return -2                                     # 不支持的功能码


# --------------------------------------------------------------------- 传输
class Transport(ABC):
    """一条 Modbus 主站链路。``request`` 必须是**阻塞且串行**的（调用方持锁）。"""

    name = ""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def request(self, unit: int, pdu: bytes, timeout: float) -> bytes: ...

    @property
    @abstractmethod
    def opened(self) -> bool: ...

    def describe(self) -> str:
        return self.name


class TcpTransport(Transport):
    name = "Modbus TCP"

    def __init__(self, host: str, port: int = 502, connect_timeout: float = 3.0) -> None:
        self.host, self.port = host, int(port)
        self.connect_timeout = float(connect_timeout)
        self._sock: socket.socket | None = None
        self._tid = 0

    @property
    def opened(self) -> bool:
        return self._sock is not None

    def open(self) -> None:
        if self._sock is not None:
            return
        s = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def describe(self) -> str:
        return f"{self.host}:{self.port}"

    def request(self, unit: int, pdu: bytes, timeout: float) -> bytes:
        self.open()
        assert self._sock is not None
        s = self._sock
        s.settimeout(max(0.05, float(timeout)))
        self._tid = (self._tid + 1) & 0xFFFF
        tid = self._tid
        s.sendall(struct.pack(">HHHB", tid, 0, len(pdu) + 1, unit & 0xFF) + pdu)
        deadline = time.monotonic() + max(0.05, float(timeout))
        while True:
            head = self._recv_exact(7, deadline)
            rtid, proto, length, runit = struct.unpack(">HHHB", head)
            if length < 1 or length > 260:
                raise CommError(f"MBAP 长度字段非法：{length}")
            body = self._recv_exact(length - 1, deadline)
            if rtid == tid and proto == 0:
                return body
            # 事务号不符：这是上一次超时请求的迟到应答，丢掉继续等自己的，绝不能当成本次结果
            if time.monotonic() >= deadline:
                raise ModbusTimeout(f"等待事务 {tid} 的应答超时（收到的是事务 {rtid}）")

    def _recv_exact(self, n: int, deadline: float) -> bytes:
        assert self._sock is not None
        buf = b""
        while len(buf) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ModbusTimeout(f"读取应答超时（已收到 {len(buf)}/{n} 字节）")
            self._sock.settimeout(left)
            try:
                chunk = self._sock.recv(n - len(buf))
            except socket.timeout:
                raise ModbusTimeout(f"读取应答超时（已收到 {len(buf)}/{n} 字节）") from None
            if not chunk:
                raise CommError("对端关闭了连接")
            buf += chunk
        return buf


class RtuTransport(Transport):
    """串口 RTU。每次请求前清空输入缓冲（丢掉迟到的上一次应答），按 CRC 校验整条报文。"""

    name = "Modbus RTU"

    def __init__(self, port: str, baudrate: int = 9600, bytesize: int = 8, parity: str = "N",
                 stopbits: float = 1.0, open_timeout: float = 2.0) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self.bytesize = int(bytesize)
        self.parity = str(parity or "N")
        self.stopbits = float(stopbits)
        self.open_timeout = float(open_timeout)
        self._ser = None

    @property
    def opened(self) -> bool:
        return self._ser is not None

    def describe(self) -> str:
        return f"{self.port}@{self.baudrate}"

    @property
    def char_time(self) -> float:
        """一个字符的时间（起始位 + 数据位 + 校验位 + 停止位）。RTU 的帧间静默按 3.5 个字符算。"""
        bits = 1 + self.bytesize + (0 if self.parity in ("N", "n") else 1) + max(1, int(self.stopbits))
        return bits / max(1, self.baudrate)

    def open(self) -> None:
        if self._ser is not None:
            return
        try:
            import serial
        except ImportError as e:
            raise CommError("Modbus RTU 需要 pyserial：pip install pyserial") from e
        try:
            self._ser = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize,
                                      parity=self.parity, stopbits=self.stopbits, timeout=0.1,
                                      write_timeout=self.open_timeout)
        except Exception as e:
            raise CommError(f"打开串口 {self.port} 失败：{e}") from e

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def request(self, unit: int, pdu: bytes, timeout: float) -> bytes:
        self.open()
        ser = self._ser
        assert ser is not None
        frame = bytes([unit & 0xFF]) + pdu
        frame += struct.pack("<H", crc16(frame))
        try:
            ser.reset_input_buffer()              # 丢掉上一次超时请求的迟到应答
        except Exception:
            pass
        ser.write(frame)
        try:
            ser.flush()
        except Exception:
            pass
        deadline = time.monotonic() + max(0.05, float(timeout))
        buf = bytearray()
        total = -1
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ModbusTimeout(f"RTU 应答超时（已收到 {len(buf)} 字节）")
            ser.timeout = min(0.1, left)
            chunk = ser.read(max(1, (total - len(buf)) if total > 0 else 1))
            if chunk:
                buf += chunk
            if len(buf) >= 2 and total < 0:
                total = expected_rtu_length(buf[1], bytes(buf))
                if total == -2:
                    raise CommError(f"应答功能码 {buf[1]} 不支持")
            if total > 0 and len(buf) >= total:
                break
        resp = bytes(buf[:total])
        if resp[0] != (unit & 0xFF):
            raise CommError(f"应答站号不符：请求 {unit}，应答 {resp[0]}")
        if crc16(resp[:-2]) != struct.unpack("<H", resp[-2:])[0]:
            raise CommError("应答 CRC 校验失败（线路干扰或波特率不符）")
        return resp[1:-2]


# --------------------------------------------------------------------- 客户端
class ModbusClient:
    """串行化的 Modbus 主站。所有访问经过一把锁，轮询与主动读写共用同一条队列。"""

    def __init__(self, transport: Transport, unit_id: int = 1, timeout: float = 1.0,
                 read_retries: int = 1, write_retries: int = 0) -> None:
        self.transport = transport
        self.unit_id = int(unit_id)
        self.timeout = float(timeout)
        self.read_retries = max(0, int(read_retries))
        #: 写操作默认**不重试**：一次写可能对应一个物理动作，超时后盲目重发会重复执行。
        self.write_retries = max(0, int(write_retries))
        self._lock = threading.RLock()
        self.requests = 0
        self.failures = 0

    # ---- 生命周期 ----
    def open(self) -> None:
        with self._lock:
            self.transport.open()

    def close(self) -> None:
        with self._lock:
            self.transport.close()

    @property
    def opened(self) -> bool:
        return self.transport.opened

    def describe(self) -> str:
        return self.transport.describe()

    # ---- 底层 ----
    def _exec(self, pdu: bytes, retries: int) -> bytes:
        last: Exception | None = None
        with self._lock:
            for attempt in range(retries + 1):
                try:
                    self.requests += 1
                    return self.transport.request(self.unit_id, pdu, self.timeout)
                except ModbusException:
                    self.failures += 1
                    raise                      # 从站明确拒绝，重试没有意义
                except Exception as e:
                    self.failures += 1
                    last = e
                    self.transport.close()     # 链路状态未知，重开一条干净的
                    if attempt >= retries:
                        break
            raise last if last else CommError("请求失败")

    # ---- 读 ----
    def read_bits(self, area: str, address: int, count: int) -> list[bool]:
        fc = 1 if area == "coil" else 2
        resp = self._exec(pdu_read(fc, address, count), self.read_retries)
        check_response(fc, resp)
        return parse_bits(resp, count)

    def read_words(self, area: str, address: int, count: int) -> list[int]:
        fc = 3 if area == "holding" else 4
        resp = self._exec(pdu_read(fc, address, count), self.read_retries)
        check_response(fc, resp)
        return parse_words(resp, count)

    # ---- 写 ----
    def write_coil(self, address: int, value: bool) -> None:
        resp = self._exec(pdu_write_coil(address, bool(value)), self.write_retries)
        check_response(5, resp)

    def write_register(self, address: int, value: int) -> None:
        resp = self._exec(pdu_write_register(address, value), self.write_retries)
        check_response(6, resp)

    def write_coils(self, address: int, bits: list[bool]) -> None:
        if len(bits) == 1:
            return self.write_coil(address, bits[0])
        resp = self._exec(pdu_write_coils(address, bits), self.write_retries)
        check_response(15, resp)

    def write_registers(self, address: int, values: list[int], force_multi: bool = False) -> None:
        if len(values) == 1 and not force_multi:
            return self.write_register(address, values[0])
        resp = self._exec(pdu_write_registers(address, values), self.write_retries)
        check_response(16, resp)

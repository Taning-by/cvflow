"""Modbus TCP devices.

* ``ModbusTcpClientDevice`` - we are the master: poll registers of a PLC (rising-edge
  trigger detection) and write results into it. Uses pymodbus.
* ``ModbusTcpServerDevice`` - we are the slave: the PLC writes a trigger register in
  our table and reads results back. Implemented natively (FC 1/2/3/4/5/6/15/16) so
  every write is observed immediately without polling and without depending on the
  pymodbus server API, which is in transition between major versions.

Register "kinds" for reading/writing typed values: int16, uint16, int32, uint32,
float32 (two registers, big-endian word order), bool (coil).
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any, Callable

from .base import CommDevice, CommError

RegCallback = Callable[[CommDevice, int, int, int], None]  # device, address, old, new

_KIND_WORDS = {"int16": 1, "uint16": 1, "int32": 2, "uint32": 2, "float32": 2, "bool": 1}


def value_to_registers(value: Any, kind: str) -> list[int]:
    if kind == "bool":
        return [1 if value else 0]
    if kind in ("int16", "uint16"):
        v = int(round(float(value)))
        if kind == "int16":
            v = max(-32768, min(32767, v)) & 0xFFFF
        else:
            v = max(0, min(65535, v))
        return [v]
    if kind in ("int32", "uint32"):
        v = int(round(float(value)))
        packed = struct.pack(">i" if kind == "int32" else ">I", v)
    elif kind == "float32":
        packed = struct.pack(">f", float(value))
    else:
        raise CommError(f"未知的寄存器类型 {kind!r}")
    return list(struct.unpack(">HH", packed))


def registers_to_value(regs: list[int], kind: str) -> Any:
    if kind == "bool":
        return bool(regs[0])
    if kind == "uint16":
        return int(regs[0])
    if kind == "int16":
        return struct.unpack(">h", struct.pack(">H", regs[0]))[0]
    packed = struct.pack(">HH", regs[0], regs[1])
    return struct.unpack({"int32": ">i", "uint32": ">I", "float32": ">f"}[kind], packed)[0]


class _RegisterWatcher:
    """Mixin: register change callbacks plus the PLC handshake helpers shared by register devices."""

    def __init__(self) -> None:
        self._reg_callbacks: list[RegCallback] = []

    # 子类需提供 read_registers / write_registers / config / connected
    def on_trigger(self, address: int) -> None:
        """收到触发后：置忙标志，按需清零触发寄存器。"""
        busy = int(self.config.get("busy_address", -1))  # type: ignore[attr-defined]
        if busy >= 0:
            self.write_registers(busy, [1])  # type: ignore[attr-defined]
        if self.config.get("trigger_reset"):  # type: ignore[attr-defined]
            self.write_registers(int(address), [0])  # type: ignore[attr-defined]

    def on_done(self, result=None) -> None:
        """流程结束（结果已写入）后：清忙标志。"""
        busy = int(self.config.get("busy_address", -1))  # type: ignore[attr-defined]
        if busy >= 0:
            self.write_registers(busy, [0])  # type: ignore[attr-defined]

    def on_register_change(self, cb: RegCallback) -> None:
        self._reg_callbacks.append(cb)

    def _notify_reg(self, address: int, old: int, new: int) -> None:
        for cb in list(self._reg_callbacks):
            try:
                cb(self, address, old, new)  # type: ignore[arg-type]
            except Exception:
                pass


# --------------------------------------------------------------------------- server
class ModbusTcpServerDevice(CommDevice, _RegisterWatcher):
    kind = "modbus_tcp_server"
    config_schema = [
        ("host", "string", "Bind address", "0.0.0.0", ""),
        ("port", "int", "Port", 502, "502 needs root on Linux; use e.g. 5020"),
        ("unit_id", "int", "Unit id", 1, "0 = accept any"),
        ("register_count", "int", "Holding registers", 256, ""),
        ("coil_count", "int", "Coils", 64, ""),
        ("busy_address", "int", "忙标志地址", -1, "-1 关闭；触发后写 1，流程结束写 0"),
        ("trigger_reset", "bool", "触发后自动复位触发寄存器", False, "PLC 写 1 触发后由视觉清零，便于下次产生上升沿"),
        ("heartbeat_address", "int", "心跳寄存器地址", -1, "-1 关闭；按心跳间隔自增，PLC 据此判断视觉在线"),
        ("heartbeat_s", "float", "心跳间隔（秒）", 0.0, "0 关闭"),
    ]

    def __init__(self, name, config=None, bus=None):
        CommDevice.__init__(self, name, config, bus)
        _RegisterWatcher.__init__(self)
        n = int(self.config["register_count"])
        self.registers: list[int] = [0] * n
        self.coils: list[bool] = [False] * int(self.config["coil_count"])
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: set[socket.socket] = set()

    # ---- public register API (used by rules / nodes) ----
    def read_registers(self, address: int, count: int = 1) -> list[int]:
        with self._lock:
            return self.registers[address:address + count]

    def write_registers(self, address: int, values: list[int]) -> None:
        with self._lock:
            if address < 0 or address + len(values) > len(self.registers):
                raise CommError("寄存器地址越界")
            olds = self.registers[address:address + len(values)]
            self.registers[address:address + len(values)] = [v & 0xFFFF for v in values]
        for i, (o, n) in enumerate(zip(olds, values)):
            if o != (n & 0xFFFF):
                self._notify_reg(address + i, o, n & 0xFFFF)

    def write_value(self, address: int, value: Any, kind: str = "int16") -> None:
        if kind == "bool":
            self._write_coils(address, [bool(value)])
        else:
            self.write_registers(address, value_to_registers(value, kind))

    def read_value(self, address: int, kind: str = "int16") -> Any:
        if kind == "bool":
            return self.coils[address]
        return registers_to_value(self.read_registers(address, _KIND_WORDS[kind]), kind)

    def _write_coils(self, address: int, values: list[bool]) -> None:
        with self._lock:
            if address < 0 or address + len(values) > len(self.coils):
                raise CommError("线圈地址越界")
            olds = self.coils[address:address + len(values)]
            self.coils[address:address + len(values)] = values
        for i, (o, n) in enumerate(zip(olds, values)):
            if o != n:
                self._notify_reg(10000 + address + i, int(o), int(n))  # coils reported at 10000+addr

    @property
    def bound_port(self) -> int:
        return self._server.getsockname()[1] if self._server else int(self.config["port"])

    # ---- lifecycle ----
    def connect(self) -> None:
        if self._server:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.config["host"], int(self.config["port"])))
        srv.listen(8)
        srv.settimeout(0.5)
        self._server = srv
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name=f"mbs-{self.name}", daemon=True)
        self._thread.start()
        self._set_connected(True)

    def disconnect(self) -> None:
        self._stop.set()
        for c in list(self._clients):
            try:
                c.close()
            except OSError:
                pass
        self._clients.clear()
        if self._server:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        if self._thread:
            self._thread.join(2.0)
            self._thread = None
        self._set_connected(False, "已关闭")

    def _send_bytes(self, data: bytes) -> None:
        raise CommError("Modbus 从站不发送原始报文，请使用 write_value()")

    def test_connection(self) -> tuple[bool, str]:
        if self._server is not None:
            return True, f"Modbus 从站正在监听 {self.config['host']}:{self.bound_port}，当前 {len(self._clients)} 个客户端"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.config["host"], int(self.config["port"])))
            s.close()
            return True, f"端口 {self.config['port']} 可用（从站尚未启动）"
        except OSError as e:
            return False, f"无法监听 {self.config['host']}:{self.config['port']}：{e}"

    # ---- protocol ----
    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(0.5)
            self._clients.add(conn)
            threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()

    def _client_loop(self, conn: socket.socket) -> None:
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
                while len(buf) >= 7:
                    tid, pid, length, unit = struct.unpack(">HHHB", buf[:7])
                    if len(buf) < 6 + length:
                        break
                    pdu, buf = buf[7:6 + length], buf[6 + length:]
                    uid = int(self.config.get("unit_id", 1))
                    if uid and unit not in (uid, 0):
                        continue
                    resp = self._handle_pdu(pdu)
                    self.stats["rx"] += 1
                    conn.sendall(struct.pack(">HHHB", tid, pid, len(resp) + 1, unit) + resp)
        except OSError:
            pass
        finally:
            self._clients.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def _handle_pdu(self, pdu: bytes) -> bytes:
        fc = pdu[0]
        try:
            if fc in (1, 2):   # read coils / discrete inputs
                addr, count = struct.unpack(">HH", pdu[1:5])
                if count < 1 or count > 2000 or addr + count > len(self.coils):
                    return bytes([fc | 0x80, 2])
                bits = self.coils[addr:addr + count]
                nbytes = (count + 7) // 8
                out = bytearray(nbytes)
                for i, b in enumerate(bits):
                    if b:
                        out[i // 8] |= 1 << (i % 8)
                return bytes([fc, nbytes]) + bytes(out)
            if fc in (3, 4):   # read holding / input registers
                addr, count = struct.unpack(">HH", pdu[1:5])
                if count < 1 or count > 125 or addr + count > len(self.registers):
                    return bytes([fc | 0x80, 2])
                regs = self.read_registers(addr, count)
                return bytes([fc, count * 2]) + struct.pack(f">{count}H", *regs)
            if fc == 5:        # write single coil
                addr, val = struct.unpack(">HH", pdu[1:5])
                if addr >= len(self.coils):
                    return bytes([fc | 0x80, 2])
                self._write_coils(addr, [val == 0xFF00])
                return pdu[:5]
            if fc == 6:        # write single register
                addr, val = struct.unpack(">HH", pdu[1:5])
                if addr >= len(self.registers):
                    return bytes([fc | 0x80, 2])
                self.write_registers(addr, [val])
                return pdu[:5]
            if fc == 15:       # write multiple coils
                addr, count, nbytes = struct.unpack(">HHB", pdu[1:6])
                if addr + count > len(self.coils):
                    return bytes([fc | 0x80, 2])
                raw = pdu[6:6 + nbytes]
                vals = [bool(raw[i // 8] >> (i % 8) & 1) for i in range(count)]
                self._write_coils(addr, vals)
                return pdu[:5]
            if fc == 16:       # write multiple registers
                addr, count, nbytes = struct.unpack(">HHB", pdu[1:6])
                if addr + count > len(self.registers) or nbytes != count * 2:
                    return bytes([fc | 0x80, 2])
                vals = list(struct.unpack(f">{count}H", pdu[6:6 + nbytes]))
                self.write_registers(addr, vals)
                return pdu[:5]
            return bytes([fc | 0x80, 1])  # illegal function
        except (struct.error, IndexError):
            return bytes([fc | 0x80, 3])  # illegal data value

    def info(self):
        d = super().info()
        d.update({"clients": len(self._clients), "registers": len(self.registers)})
        return d


# --------------------------------------------------------------------------- client
class ModbusTcpClientDevice(CommDevice, _RegisterWatcher):
    kind = "modbus_tcp_client"
    config_schema = [
        ("host", "string", "PLC host", "192.168.0.10", ""),
        ("port", "int", "Port", 502, ""),
        ("unit_id", "int", "Unit id", 1, ""),
        ("poll_ms", "int", "Poll interval (ms)", 50, "0 disables polling"),
        ("watch_address", "int", "Watch start address", 0, "Holding registers polled for changes"),
        ("watch_count", "int", "Watch count", 8, ""),
        ("timeout_s", "float", "Timeout (s)", 1.0, ""),
        ("busy_address", "int", "忙标志地址", -1, "-1 关闭；触发后写 1，流程结束写 0"),
        ("trigger_reset", "bool", "触发后自动复位触发寄存器", False, ""),
        ("heartbeat_address", "int", "心跳寄存器地址", -1, "-1 关闭；按心跳间隔自增"),
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
                from pymodbus.client import ModbusTcpClient
            except ImportError as e:  # pragma: no cover
                raise CommError("未安装 pymodbus") from e
            self._client = ModbusTcpClient(self.config["host"], port=int(self.config["port"]),
                                           timeout=float(self.config.get("timeout_s", 1.0)), retries=1)
        return self._client

    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"mbc-{self.name}", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3.0)
            self._thread = None
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
        self._set_connected(False, "已关闭")

    def _send_bytes(self, data: bytes) -> None:
        raise CommError("Modbus 主站不发送原始报文，请使用 write_value()")

    def test_connection(self) -> tuple[bool, str]:
        t0 = time.perf_counter()
        try:
            with self._lock:
                c = self._ensure_client()
                if not c.connected and not c.connect():
                    raise CommError("TCP 连接失败")
            addr = int(self.config.get("watch_address", 0))
            v = self.read_registers(addr, 1)[0]
            return True, f"{self.config['host']}:{self.config['port']} 单元 {self.config['unit_id']} 可用，寄存器 {addr}={v}，耗时 {(time.perf_counter() - t0) * 1000:.0f} ms"
        except Exception as e:
            return False, f"{self.config['host']}:{self.config['port']} 连接失败：{e}"

    def _check(self, rr):
        if rr is None or rr.isError():
            raise CommError(f"Modbus 错误：{rr}")
        return rr

    def read_registers(self, address: int, count: int = 1) -> list[int]:
        with self._lock:
            c = self._ensure_client()
            rr = self._check(c.read_holding_registers(address, count=count, device_id=int(self.config["unit_id"])))
            return list(rr.registers)

    def write_registers(self, address: int, values: list[int]) -> None:
        with self._lock:
            c = self._ensure_client()
            uid = int(self.config["unit_id"])
            if len(values) == 1:
                self._check(c.write_register(address, values[0] & 0xFFFF, device_id=uid))
            else:
                self._check(c.write_registers(address, [v & 0xFFFF for v in values], device_id=uid))
            self.stats["tx"] += 1

    def write_value(self, address: int, value: Any, kind: str = "int16") -> None:
        if kind == "bool":
            with self._lock:
                c = self._ensure_client()
                self._check(c.write_coil(address, bool(value), device_id=int(self.config["unit_id"])))
                self.stats["tx"] += 1
            return
        self.write_registers(address, value_to_registers(value, kind))

    def read_value(self, address: int, kind: str = "int16") -> Any:
        if kind == "bool":
            with self._lock:
                c = self._ensure_client()
                rr = self._check(c.read_coils(address, count=1, device_id=int(self.config["unit_id"])))
                return bool(rr.bits[0])
        return registers_to_value(self.read_registers(address, _KIND_WORDS[kind]), kind)

    def _loop(self) -> None:
        poll = int(self.config.get("poll_ms", 50)) / 1000.0
        while not self._stop.is_set():
            try:
                with self._lock:
                    c = self._ensure_client()
                    if not c.connected:
                        if not c.connect():
                            raise CommError("连接失败")
                if poll <= 0:
                    self._set_connected(True)
                    self._stop.wait(0.5)
                    continue
                addr, count = int(self.config["watch_address"]), int(self.config["watch_count"])
                regs = self.read_registers(addr, count)
                self.stats["rx"] += 1
                if self._last is not None:
                    for i, (o, n) in enumerate(zip(self._last, regs)):
                        if o != n:
                            self._notify_reg(addr + i, o, n)
                self._last = regs
                self._set_connected(True)   # 基线快照建立之后才对外显示“已连接”，避免漏掉首次变化
                self._stop.wait(poll)
            except Exception as e:
                if self.connected or not self.last_error:
                    self._error(str(e))
                self._set_connected(False, str(e))
                self._last = None
                with self._lock:
                    if self._client is not None:
                        try:
                            self._client.close()
                        except Exception:
                            pass
                self._stop.wait(1.0)

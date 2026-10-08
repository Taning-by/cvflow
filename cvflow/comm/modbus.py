"""Modbus 设备：主站（TCP / RTU）与从站（TCP）。

* ``ModbusTcpClientDevice`` / ``ModbusRtuClientDevice``：**我们是主站**，主动读写外部设备。
  协议层见 ``modbus_client``（自带实现，不依赖 pymodbus）。轮询与流程里的主动读写共用同一把
  传输锁，所以不会出现"拿到上一次请求的应答"这种错配，RTU 半双工也不会撞车。
* ``ModbusTcpServerDevice``：**我们是从站**，PLC 直接读写本机的数据区。自带实现
  （功能码 01/02/03/04/05/06/15/16），PLC 的每次写入都立即被观察到，不需要轮询。
  离散输入与输入寄存器按协议本身就没有写功能码，所以远程写不进去；另有
  ``allow_remote_write`` 可以把整张表锁成只读。

两种角色都支持**数据点表**（见 ``modbus_map``）：给地址起名字、声明类型与字节序、绑定全局
变量、设置轮询周期。流程、触发规则和检测握手都按**名字**引用数据点，改地址不用改流程。
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from typing import Any, Callable

from .base import CommDevice, CommError, Peer
from .modbus_client import (ModbusClient, ModbusException, RtuTransport, TcpTransport,
                            MAX_READ_BITS, MAX_READ_WORDS)
from .modbus_map import (AREA_INFO, DataPoint, DataPointTable, MapError, PointValue,
                         decode_value, encode_value)

log = logging.getLogger("cvflow.comm")

RegCallback = Callable[..., None]        # (device, address, old, new)
PointCallback = Callable[..., None]      # (device, point, old, new)

_KIND_WORDS = {"int16": 1, "uint16": 1, "int32": 2, "uint32": 2, "float32": 2, "bool": 1}


# ------------------------------------------------------------------ 兼容旧接口
def value_to_registers(value: Any, kind: str) -> list[int]:
    """值 → 寄存器（大端 ABCD）。老接口，内部转调 modbus_map。"""
    if kind not in _KIND_WORDS:
        raise CommError(f"未知的寄存器类型 {kind!r}")
    return encode_value(value, kind, "ABCD")


def registers_to_value(regs: list[int], kind: str) -> Any:
    if kind not in _KIND_WORDS:
        raise CommError(f"未知的寄存器类型 {kind!r}")
    return decode_value(list(regs), kind, "ABCD")


# ------------------------------------------------------------------ 数据点宿主
class _PointHost:
    """给设备加上数据点表：命名读写、类型与字节序换算、绑定全局变量、断线失效。"""

    def __init__(self) -> None:
        self.points = DataPointTable()
        self.values: dict[str, PointValue] = {}
        self.variables = None                      # GlobalVariables，由管理器注入
        self._point_callbacks: list[tuple[PointCallback, int]] = []
        self._point_lock = threading.RLock()

    # ---- 配置 ----
    def load_points(self, items: list[dict] | None) -> list[str]:
        """载入数据点表，返回出错信息（坏的那一条被跳过，其它照常载入）。"""
        errors: list[str] = []
        table = DataPointTable()
        for d in items or []:
            try:
                table.add(DataPoint.from_dict(d))
            except (MapError, TypeError) as e:
                errors.append(str(e))
        with self._point_lock:
            self.points = table
            self.values = {p.name: PointValue() for p in table}
        return errors

    def points_to_list(self) -> list[dict]:
        return self.points.to_list()

    def point(self, name: str) -> DataPoint:
        p = self.points.get(name)
        if p is None:
            raise CommError(f"没有数据点 {name!r}（已配置：{', '.join(self.points.names()) or '无'}）")
        return p

    def has_point(self, name: str) -> bool:
        return self.points.get(name) is not None

    def point_value(self, name: str) -> PointValue:
        with self._point_lock:
            return self.values.get(name, PointValue())

    def point_snapshot(self) -> dict[str, dict]:
        with self._point_lock:
            return {k: v.to_dict() for k, v in self.values.items()}

    # ---- 变化通知 ----
    def on_point_change(self, cb: PointCallback) -> None:
        import inspect
        arity = 4
        try:
            arity = min(4, len([p for p in inspect.signature(cb).parameters.values()
                                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]))
        except (TypeError, ValueError):
            pass
        self._point_callbacks.append((cb, arity))

    def _notify_point(self, point: DataPoint, old: Any, new: Any) -> None:
        for cb, arity in list(self._point_callbacks):
            try:
                cb(self, point, old, new) if arity >= 4 else cb(self, point)
            except Exception:
                log.exception("数据点回调出错：%s", point.name)

    # ---- 值记录 ----
    def _store(self, point: DataPoint, words: list[int] | list[bool], now: float | None = None,
               defer: list | None = None) -> Any:
        """解码、写入缓存、绑定变量、必要时发变化通知。返回解码后的值。

        ``defer`` 给定时**不立即派发**变化通知，而是把 ``(数据点, 旧值, 新值)`` 追加进去，
        由调用方在整批数据都落地之后统一派发。对端常常用一条请求同时写触发位和请求编号，
        逐个派发会让触发规则读到上一次的请求编号。
        """
        now = time.time() if now is None else now
        try:
            value = point.decode(list(words))
        except Exception as e:
            with self._point_lock:
                pv = self.values.setdefault(point.name, PointValue())
                pv.valid, pv.error, pv.updated = False, f"解码失败：{e}", now
            return None
        with self._point_lock:
            pv = self.values.setdefault(point.name, PointValue())
            old, had = pv.value, pv.valid
            pv.value, pv.words, pv.updated = value, [int(w) for w in words], now
            pv.valid, pv.stale, pv.error = True, False, ""
        if point.variable and self.variables is not None and (not had or old != value):
            try:
                self.variables.set(point.variable, value)
            except Exception:
                log.exception("数据点 %s 绑定变量 %s 失败", point.name, point.variable)
        # 第一次读到只建立基线，不当成变化——否则一上线就会误触发一次
        if had and old != value:
            if defer is None:
                self._notify_point(point, old, value)
            else:
                defer.append((point, old, value))
        return value

    def invalidate_points(self, reason: str = "连接断开") -> None:
        with self._point_lock:
            for pv in self.values.values():
                if pv.valid:
                    pv.stale, pv.error = True, reason

    # ---- 子类实现 ----
    def read_point(self, name: str) -> Any:
        raise NotImplementedError

    def write_point(self, name: str, value: Any) -> None:
        raise NotImplementedError


# ------------------------------------------------------------------ 寄存器握手
class _RegisterWatcher:
    """寄存器类设备共用：地址变化回调 + PLC 握手的忙标志/触发复位。

    ``busy_address`` / ``trigger_reset`` 是**简版握手**，保留给老方案；完整的
    Ready/Busy/Done/请求编号闭环见 ``handshake.py``。
    """

    def __init__(self) -> None:
        self._reg_callbacks: list[RegCallback] = []

    def on_trigger(self, address: int) -> None:
        busy = int(self.config.get("busy_address", -1))        # type: ignore[attr-defined]
        if busy >= 0:
            self.write_registers(busy, [1])                    # type: ignore[attr-defined]
        if self.config.get("trigger_reset"):                   # type: ignore[attr-defined]
            self.write_registers(int(address), [0])            # type: ignore[attr-defined]

    def on_done(self, result=None) -> None:
        busy = int(self.config.get("busy_address", -1))        # type: ignore[attr-defined]
        if busy >= 0:
            self.write_registers(busy, [0])                    # type: ignore[attr-defined]

    def on_register_change(self, cb: RegCallback) -> None:
        self._reg_callbacks.append(cb)

    def _notify_reg(self, address: int, old: int, new: int) -> None:
        for cb in list(self._reg_callbacks):
            try:
                cb(self, address, old, new)                    # type: ignore[arg-type]
            except Exception:
                log.exception("寄存器回调出错：地址 %s", address)


# =================================================================== 从站
class ModbusTcpServerDevice(CommDevice, _RegisterWatcher, _PointHost):
    kind = "modbus_tcp_server"
    role = "slave"
    byte_stream = False
    supports_peers = True
    supports_registers = True
    config_schema = [
        ("host", "string", "监听地址", "0.0.0.0", ""),
        ("port", "int", "监听端口", 502, "Linux 上 502 需要 root，调试建议用 5020"),
        ("unit_id", "int", "从站站号", 1, "0 表示接受任意站号"),
        ("register_count", "int", "保持寄存器数量 (4x)", 256, ""),
        ("input_count", "int", "输入寄存器数量 (3x)", 64, ""),
        ("coil_count", "int", "线圈数量 (0x)", 64, ""),
        ("discrete_count", "int", "离散输入数量 (1x)", 64, ""),
        ("allow_remote_write", "bool", "允许对端写入", True,
         "关掉后对所有写功能码回异常码 02，整张表只读（离散输入与输入寄存器本来就不可写）"),
        ("max_clients", "int", "最大客户端数", 8, ""),
        ("busy_address", "int", "忙标志地址（简版握手）", -1, "-1 关闭；触发后写 1，流程结束写 0"),
        ("trigger_reset", "bool", "触发后自动复位触发寄存器", False, ""),
        ("heartbeat_address", "int", "心跳寄存器地址", -1, "-1 关闭；按心跳间隔自增"),
    ]

    def __init__(self, name, config=None, bus=None, device_id=None, enabled=True):
        CommDevice.__init__(self, name, config, bus, device_id, enabled)
        _RegisterWatcher.__init__(self)
        _PointHost.__init__(self)
        self.holding: list[int] = [0] * max(1, self.cfg_int("register_count", 256))
        self.input_registers: list[int] = [0] * max(1, self.cfg_int("input_count", 64))
        self.coils: list[bool] = [False] * max(1, self.cfg_int("coil_count", 64))
        self.discrete: list[bool] = [False] * max(1, self.cfg_int("discrete_count", 64))
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clients: dict[str, socket.socket] = {}

    #: 兼容旧代码：``registers`` 就是保持寄存器
    @property
    def registers(self) -> list[int]:
        return self.holding

    @registers.setter
    def registers(self, value: list[int]) -> None:
        self.holding = value

    def _area_list(self, area: str):
        return {"coil": self.coils, "discrete": self.discrete,
                "holding": self.holding, "input": self.input_registers}[area]

    # ---- 本地数据区读写（供规则、流程、握手使用）----
    def read_area(self, area: str, address: int, count: int = 1) -> list:
        table = self._area_list(area)
        with self._lock:
            if address < 0 or address + count > len(table):
                raise CommError(f"{AREA_INFO[area]['label']} 地址 {address} 起 {count} 个越界"
                                f"（本机共 {len(table)} 个）")
            return list(table[address:address + count])

    def write_area(self, area: str, address: int, values: list) -> None:
        table = self._area_list(area)
        bits = AREA_INFO[area]["bits"]
        with self._lock:
            if address < 0 or address + len(values) > len(table):
                raise CommError(f"{AREA_INFO[area]['label']} 地址 {address} 起 {len(values)} 个越界"
                                f"（本机共 {len(table)} 个）")
            olds = list(table[address:address + len(values)])
            new = [bool(v) for v in values] if bits else [int(v) & 0xFFFF for v in values]
            table[address:address + len(values)] = new
        changed = [(i, o, n) for i, (o, n) in enumerate(zip(olds, new)) if o != n]
        if not changed:
            return
        # 先把这次写入覆盖到的数据点全部重新落地，再统一派发变化通知。
        # 对端常用一条 FC16 同时写触发位和请求编号，逐个派发会让触发规则读到旧的请求编号。
        deferred: list = []
        self._touch_range(area, address, len(values), deferred)
        for i, o, n in changed:
            # 旧接口的地址语义：保持寄存器用原地址，线圈用 10000+地址
            if area in ("holding", "coil"):
                legacy = address + i + (10000 if area == "coil" else 0)
                self._notify_reg(legacy, int(o), int(n))
        for point, old, new_value in deferred:
            self._notify_point(point, old, new_value)

    def _touch_range(self, area: str, address: int, count: int, defer: list | None = None) -> None:
        """本机数据区的 [address, address+count) 被改动后，重新解码覆盖到这一段的数据点。"""
        for p in self.points:
            if not p.enabled or p.area != area:
                continue
            if int(p.address) < address + count and address < int(p.address) + p.words:
                try:
                    words = self.read_area(area, int(p.address), p.words)
                except CommError:
                    continue
                self._store(p, words, defer=defer)

    # ---- 旧接口 ----
    def read_registers(self, address: int, count: int = 1) -> list[int]:
        return self.read_area("holding", address, count)

    def write_registers(self, address: int, values: list[int]) -> None:
        self.write_area("holding", address, list(values))

    def write_value(self, address: int, value: Any, kind: str = "int16") -> None:
        if kind == "bool":
            self.write_area("coil", int(address), [bool(value)])
        else:
            self.write_area("holding", int(address), value_to_registers(value, kind))

    def read_value(self, address: int, kind: str = "int16") -> Any:
        if kind == "bool":
            return bool(self.read_area("coil", int(address), 1)[0])
        return registers_to_value(self.read_area("holding", int(address), _KIND_WORDS[kind]), kind)

    # ---- 数据点 ----
    def read_point(self, name: str) -> Any:
        p = self.point(name)
        return self._store(p, self.read_area(p.area, int(p.address), p.words))

    def write_point(self, name: str, value: Any) -> None:
        p = self.point(name)
        self.write_area(p.area, int(p.address), p.encode(value))

    # ---- 状态 ----
    @property
    def bound_port(self) -> int:
        return self._server.getsockname()[1] if self._server else self.cfg_int("port", 502)

    def peer_text(self) -> str:
        if self._server is None:
            return f"未监听 {self.config.get('host')}:{self.config.get('port')}"
        if not self._clients:
            return f"监听 {self.config.get('host')}:{self.bound_port}，无主站"
        return (next(iter(self._clients)) if len(self._clients) == 1
                else f"{len(self._clients)} 个主站")

    def test_connection(self) -> tuple[bool, str]:
        areas = (f"4x×{len(self.holding)} 3x×{len(self.input_registers)} "
                 f"0x×{len(self.coils)} 1x×{len(self.discrete)}")
        if self._server is not None:
            return True, (f"Modbus 从站正在监听 {self.config['host']}:{self.bound_port}"
                          f"（站号 {self.config['unit_id']}，{areas}），"
                          f"当前 {len(self._clients)} 个主站"
                          + ("" if self.config.get("allow_remote_write", True) else "，已设为只读"))
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.config["host"], self.cfg_int("port", 502)))
            s.close()
            return True, f"端口 {self.config['port']} 可用（从站尚未启动，{areas}）"
        except OSError as e:
            extra = "（Linux 下 502 端口需要 root，调试请改 5020）" if self.cfg_int("port", 502) < 1024 else ""
            return False, f"无法监听 {self.config['host']}:{self.config['port']}：{e}{extra}"

    # ---- 生命周期 ----
    def connect(self) -> None:
        if self._server:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.config["host"], self.cfg_int("port", 502)))
        srv.listen(max(1, self.cfg_int("max_clients", 8)))
        srv.settimeout(0.5)
        self._server = srv
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name=f"mbs-{self.name}", daemon=True)
        self._thread.start()
        self._set_connected(True)
        for p in self.points:                     # 从站的数据点一上线就有值（本机表）
            try:
                self.read_point(p.name)
            except CommError:
                pass

    def disconnect(self) -> None:
        self._stop.set()
        for c in list(self._clients.values()):
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
        self.invalidate_points("从站已停止")
        self._set_connected(False, "已关闭")

    def _send_bytes(self, data: bytes, peer: Peer | None = None) -> None:
        raise CommError("Modbus 从站不发送原始报文，请写数据点或用 write_value()")

    # ---- 协议 ----
    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if len(self._clients) >= max(1, self.cfg_int("max_clients", 8)):
                try:
                    conn.close()
                except OSError:
                    pass
                self._error(f"主站数已达上限，拒绝 {addr[0]}:{addr[1]}")
                continue
            pid = f"{addr[0]}:{addr[1]}"
            conn.settimeout(0.5)
            self._clients[pid] = conn
            self._add_peer(pid, pid)
            threading.Thread(target=self._client_loop, args=(pid, conn), daemon=True,
                             name=f"mbs-{self.name}-{pid}").start()

    def _client_loop(self, pid: str, conn: socket.socket) -> None:
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
                    tid, proto, length, unit = struct.unpack(">HHHB", buf[:7])
                    if length < 1 or length > 260:
                        buf = b""                  # MBAP 头不可信，丢掉重新同步
                        self._error(f"{pid}：MBAP 长度字段非法（{length}），已重置接收缓冲")
                        break
                    if len(buf) < 6 + length:
                        break
                    pdu, buf = buf[7:6 + length], buf[6 + length:]
                    uid = self.cfg_int("unit_id", 1)
                    if uid and unit not in (uid, 0):
                        continue                   # 站号不符：按规范不应答
                    resp = self._handle_pdu(pdu)
                    self.stats["rx"] += 1
                    self.last_rx = time.time()
                    peer = self.peers.get(pid)
                    if peer is not None:
                        peer.rx += 1
                        peer.last_rx = self.last_rx
                    conn.sendall(struct.pack(">HHHB", tid, proto, len(resp) + 1, unit) + resp)
                    self.stats["tx"] += 1
                    self.last_tx = time.time()
        except OSError:
            pass
        finally:
            self._clients.pop(pid, None)
            self._drop_peer(pid)
            try:
                conn.close()
            except OSError:
                pass

    def _exc(self, fc: int, code: int) -> bytes:
        return bytes([fc | 0x80, code])

    def _handle_pdu(self, pdu: bytes) -> bytes:
        if not pdu:
            return b""
        fc = pdu[0]
        writable = bool(self.config.get("allow_remote_write", True))
        try:
            if fc in (1, 2):
                area = "coil" if fc == 1 else "discrete"
                addr, count = struct.unpack(">HH", pdu[1:5])
                if not 1 <= count <= MAX_READ_BITS or addr + count > len(self._area_list(area)):
                    return self._exc(fc, 2)
                bits = self.read_area(area, addr, count)
                nbytes = (count + 7) // 8
                out = bytearray(nbytes)
                for i, b in enumerate(bits):
                    if b:
                        out[i // 8] |= 1 << (i % 8)
                return bytes([fc, nbytes]) + bytes(out)
            if fc in (3, 4):
                area = "holding" if fc == 3 else "input"
                addr, count = struct.unpack(">HH", pdu[1:5])
                if not 1 <= count <= MAX_READ_WORDS or addr + count > len(self._area_list(area)):
                    return self._exc(fc, 2)
                regs = self.read_area(area, addr, count)
                return bytes([fc, count * 2]) + struct.pack(f">{count}H", *regs)
            if fc == 5:
                addr, val = struct.unpack(">HH", pdu[1:5])
                if not writable:
                    return self._exc(fc, 2)
                if addr >= len(self.coils):
                    return self._exc(fc, 2)
                if val not in (0x0000, 0xFF00):
                    return self._exc(fc, 3)
                self.write_area("coil", addr, [val == 0xFF00])
                return pdu[:5]
            if fc == 6:
                addr, val = struct.unpack(">HH", pdu[1:5])
                if not writable:
                    return self._exc(fc, 2)
                if addr >= len(self.holding):
                    return self._exc(fc, 2)
                self.write_area("holding", addr, [val])
                return pdu[:5]
            if fc == 15:
                addr, count, nbytes = struct.unpack(">HHB", pdu[1:6])
                if not writable:
                    return self._exc(fc, 2)
                if addr + count > len(self.coils) or count < 1:
                    return self._exc(fc, 2)
                if nbytes != (count + 7) // 8 or len(pdu) < 6 + nbytes:
                    return self._exc(fc, 3)
                raw = pdu[6:6 + nbytes]
                self.write_area("coil", addr, [bool(raw[i // 8] >> (i % 8) & 1) for i in range(count)])
                return pdu[:5]
            if fc == 16:
                addr, count, nbytes = struct.unpack(">HHB", pdu[1:6])
                if not writable:
                    return self._exc(fc, 2)
                if addr + count > len(self.holding) or count < 1:
                    return self._exc(fc, 2)
                if nbytes != count * 2 or len(pdu) < 6 + nbytes:
                    return self._exc(fc, 3)
                self.write_area("holding", addr, list(struct.unpack(f">{count}H", pdu[6:6 + nbytes])))
                return pdu[:5]
            return self._exc(fc, 1)
        except CommError:
            return self._exc(fc, 2)
        except (struct.error, IndexError):
            return self._exc(fc, 3)

    def info(self):
        d = super().info()
        d.update({"clients": len(self._clients), "registers": len(self.holding),
                  "points": len(self.points)})
        return d

    def to_dict(self) -> dict:
        d = super().to_dict()
        if len(self.points):
            d["points"] = self.points_to_list()
        return d


# =================================================================== 主站
class _ModbusClientBase(CommDevice, _RegisterWatcher, _PointHost):
    """Modbus 主站的公共部分：连接循环、数据点轮询、兼容旧的监视窗口。"""

    role = "master"
    byte_stream = False
    supports_registers = True

    def __init__(self, name, config=None, bus=None, device_id=None, enabled=True):
        CommDevice.__init__(self, name, config, bus, device_id, enabled)
        _RegisterWatcher.__init__(self)
        _PointHost.__init__(self)
        self._client: ModbusClient | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last: list[int] | None = None        # 旧的监视窗口基线
        self._next_poll: dict[str, float] = {}

    # ---- 子类提供传输 ----
    def _make_transport(self):
        raise NotImplementedError

    def _client_or_open(self) -> ModbusClient:
        with self._lock:
            if self._client is None:
                self._client = ModbusClient(
                    self._make_transport(), unit_id=self.cfg_int("unit_id", 1),
                    timeout=self.request_timeout,
                    read_retries=self.cfg_int("read_retries", 1),
                    write_retries=self.cfg_int("write_retries", 0))
            self._client.unit_id = self.cfg_int("unit_id", 1)
            self._client.timeout = self.request_timeout
            self._client.open()
            return self._client

    def _close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None

    # ---- 读写 ----
    def read_area(self, area: str, address: int, count: int = 1) -> list:
        c = self._client_or_open()
        if AREA_INFO[area]["bits"]:
            return c.read_bits(area, int(address), int(count))
        return c.read_words(area, int(address), int(count))

    def write_area(self, area: str, address: int, values: list) -> None:
        if not AREA_INFO[area]["writable"]:
            raise CommError(f"{AREA_INFO[area]['label']} 按 Modbus 协议不可写")
        c = self._client_or_open()
        if AREA_INFO[area]["bits"]:
            c.write_coils(int(address), [bool(v) for v in values])
        else:
            c.write_registers(int(address), [int(v) & 0xFFFF for v in values])
        self.stats["tx"] += 1
        self.last_tx = time.time()

    def read_registers(self, address: int, count: int = 1) -> list[int]:
        return self.read_area("holding", address, count)

    def write_registers(self, address: int, values: list[int]) -> None:
        self.write_area("holding", address, list(values))

    def write_value(self, address: int, value: Any, kind: str = "int16") -> None:
        if kind == "bool":
            self.write_area("coil", int(address), [bool(value)])
            return
        self.write_area("holding", int(address), value_to_registers(value, kind))

    def read_value(self, address: int, kind: str = "int16") -> Any:
        if kind == "bool":
            return bool(self.read_area("coil", int(address), 1)[0])
        return registers_to_value(self.read_area("holding", int(address), _KIND_WORDS[kind]), kind)

    # ---- 数据点 ----
    def read_point(self, name: str) -> Any:
        p = self.point(name)
        words = self.read_area(p.area, int(p.address), p.words)
        self.stats["rx"] += 1
        self.last_rx = time.time()
        return self._store(p, words)

    def write_point(self, name: str, value: Any) -> None:
        p = self.point(name)
        if not p.writes:
            raise CommError(f"数据点 {p.name!r} 的方向是“{p.direction}”，不允许写")
        self.write_area(p.area, int(p.address), p.encode(value))

    # ---- 生命周期 ----
    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"mb-{self.name}", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3.0)
            self._thread = None
        self._close()
        self.invalidate_points()
        self._last = None
        self._set_connected(False, "已关闭")

    def _send_bytes(self, data: bytes, peer: Peer | None = None) -> None:
        raise CommError("Modbus 主站不发送原始报文，请写数据点或用 write_value()")

    # ---- 轮询 ----
    def _loop(self) -> None:
        legacy_poll = self.cfg_int("poll_ms", 50) / 1000.0
        while not self._stop.is_set():
            try:
                self._client_or_open()
                did = self._poll_points()
                did = self._poll_legacy_window(legacy_poll) or did
                self._set_connected(True)       # 基线建立之后才显示"已连接"，避免漏掉首次变化
                if not did:
                    self._stop.wait(0.02)
            except Exception as e:
                if self.connected or not self.last_error:
                    self._error(str(e))
                self._set_connected(False, str(e))
                self.invalidate_points(str(e))
                self._last = None
                self._next_poll.clear()
                self._close()
                if not self.auto_reconnect:
                    return
                self._stop.wait(self.reconnect_interval)

    def _poll_points(self) -> bool:
        """到点的数据点按块合并后读一次。返回这一轮有没有真的发请求。"""
        blocks = self.points.read_blocks(only_polled=True)
        if not blocks:
            return False
        now = time.monotonic()
        did = False
        deferred: list = []
        for block in blocks:
            key = f"{block.area}:{block.address}:{block.count}"
            due = min((p.poll_ms for p, _ in block.points if p.poll_ms > 0), default=0) / 1000.0
            if now < self._next_poll.get(key, 0.0):
                continue
            self._next_poll[key] = now + max(0.005, due)
            words = self.read_area(block.area, block.address, block.count)
            self.stats["rx"] += 1
            self.last_rx = time.time()
            did = True
            for p, offset in block.points:
                self._store(p, list(words[offset:offset + p.words]), defer=deferred)
        # 一次轮询读到的是一份快照：全部落地之后再派发变化通知，触发规则才能读到
        # 同一快照里的请求编号（对端通常先写编号、再置触发位，两者在同一个块里）
        for point, old, new_value in deferred:
            self._notify_point(point, old, new_value)
        return did

    def _poll_legacy_window(self, interval: float) -> bool:
        """旧方案的"监视起始地址 + 字数"窗口，继续支持按原地址的触发规则。"""
        if interval <= 0:
            return False
        count = self.cfg_int("watch_count", 0)
        if count <= 0:
            return False
        now = time.monotonic()
        if now < self._next_poll.get("__legacy__", 0.0):
            return False
        self._next_poll["__legacy__"] = now + interval
        addr = self.cfg_int("watch_address", 0)
        regs = self.read_area("holding", addr, count)
        self.stats["rx"] += 1
        self.last_rx = time.time()
        if self._last is not None:
            for i, (o, n) in enumerate(zip(self._last, regs)):
                if o != n:
                    self._notify_reg(addr + i, o, n)
        self._last = list(regs)
        return True

    def info(self):
        d = super().info()
        if self._client is not None:
            d.update({"requests": self._client.requests, "failures": self._client.failures})
        d["points"] = len(self.points)
        return d

    def to_dict(self) -> dict:
        d = super().to_dict()
        if len(self.points):
            d["points"] = self.points_to_list()
        return d


class ModbusTcpClientDevice(_ModbusClientBase):
    kind = "modbus_tcp_client"
    config_schema = [
        ("host", "string", "从站地址", "192.168.0.10", "PLC 或仪表的 IP"),
        ("port", "int", "端口", 502, ""),
        ("unit_id", "int", "从站站号", 1, ""),
        ("poll_ms", "int", "监视窗口轮询间隔（毫秒）", 50, "0 关闭；数据点有各自的轮询周期"),
        ("watch_address", "int", "监视起始地址", 0, "兼容老方案的按地址触发；0 字时关闭"),
        ("watch_count", "int", "监视字数", 8, "0 表示不用监视窗口，只用数据点"),
        ("read_retries", "int", "读重试次数", 1, "读超时后重试几次"),
        ("write_retries", "int", "写重试次数", 0,
         "默认 0：一次写可能对应一个物理动作，超时后盲目重发会重复执行"),
        ("busy_address", "int", "忙标志地址（简版握手）", -1, "-1 关闭"),
        ("trigger_reset", "bool", "触发后自动复位触发寄存器", False, ""),
        ("heartbeat_address", "int", "心跳寄存器地址", -1, "-1 关闭；按心跳间隔自增"),
    ]

    def _make_transport(self):
        return TcpTransport(str(self.config["host"]), self.cfg_int("port", 502), self.connect_timeout)

    def peer_text(self) -> str:
        return f"{self.config.get('host')}:{self.config.get('port')} 站号 {self.config.get('unit_id')}"

    def test_connection(self) -> tuple[bool, str]:
        t0 = time.perf_counter()
        try:
            addr = self.cfg_int("watch_address", 0)
            probe = next((p for p in self.points if p.reads), None)
            if probe is not None:
                v = self.read_point(probe.name)
                what = f"数据点 {probe.name}（{probe.reference}）= {v}"
            else:
                v = self.read_area("holding", addr, 1)[0]
                what = f"保持寄存器 {addr} = {v}"
            return True, (f"{self.peer_text()} 可用，{what}，"
                          f"耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")
        except ModbusException as e:
            return False, f"{self.peer_text()} 已连接但请求被拒绝：{e}"
        except Exception as e:
            self._close()
            return False, f"{self.peer_text()} 连接失败：{e}"


class ModbusRtuClientDevice(_ModbusClientBase):
    """串口上的 Modbus 主站（RTU）。从站角色见本文件末尾的说明。"""

    kind = "modbus_rtu_client"
    config_schema = [
        ("port", "string", "串口号", "/dev/ttyUSB0", "Windows 形如 COM3"),
        ("baudrate", "int", "波特率", 9600, ""),
        ("bytesize", "enum:7,8", "数据位", "8", "RTU 必须是 8 位"),
        ("parity", "enum:N,E,O", "校验位", "N", ""),
        ("stopbits", "enum:1,1.5,2", "停止位", "1", ""),
        ("unit_id", "int", "从站站号", 1, "1…247"),
        ("poll_ms", "int", "监视窗口轮询间隔（毫秒）", 100, "0 关闭"),
        ("watch_address", "int", "监视起始地址", 0, ""),
        ("watch_count", "int", "监视字数", 0, "0 表示只用数据点"),
        ("read_retries", "int", "读重试次数", 2, "串口受干扰时重试很常见"),
        ("write_retries", "int", "写重试次数", 0, "默认 0：避免重复执行物理动作"),
        ("busy_address", "int", "忙标志地址（简版握手）", -1, ""),
        ("trigger_reset", "bool", "触发后自动复位触发寄存器", False, ""),
        ("heartbeat_address", "int", "心跳寄存器地址", -1, ""),
    ]

    def _make_transport(self):
        return RtuTransport(str(self.config["port"]), self.cfg_int("baudrate", 9600),
                            int(float(self.config.get("bytesize", 8) or 8)),
                            str(self.config.get("parity", "N") or "N")[:1].upper(),
                            float(self.config.get("stopbits", 1) or 1), self.connect_timeout)

    def peer_text(self) -> str:
        return (f"{self.config.get('port')}@{self.config.get('baudrate')} "
                f"站号 {self.config.get('unit_id')}")

    def test_connection(self) -> tuple[bool, str]:
        # 直接试一次真实请求；端口枚举只在失败时作为提示（见 serial_port._port_hint）
        t0 = time.perf_counter()
        try:
            probe = next((p for p in self.points if p.reads), None)
            if probe is not None:
                v = self.read_point(probe.name)
                what = f"数据点 {probe.name}（{probe.reference}）= {v}"
            else:
                v = self.read_area("holding", self.cfg_int("watch_address", 0), 1)[0]
                what = f"保持寄存器 {self.cfg_int('watch_address', 0)} = {v}"
            return True, f"{self.peer_text()} 可用，{what}，耗时 {(time.perf_counter() - t0) * 1000:.0f} ms"
        except ModbusException as e:
            return False, f"{self.peer_text()} 已连接但请求被拒绝：{e}"
        except Exception as e:
            self._close()
            from .serial_port import _port_hint
            return False, f"{self.peer_text()} 失败：{e}{_port_hint(str(self.config.get('port', '')), e)}"

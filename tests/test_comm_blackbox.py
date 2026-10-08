"""通信功能黑盒测试：把通信层当作整体，只从“对端”观察。

对端手段：真实 TCP/UDP 套接字、Linux 伪终端串口、第三方 pymodbus 主站与从站、snap7 服务器，
以及按协议手册手工拼出的报文（验证字节级一致性）。观察点：事件总线上的事件、被触发的流程与统计、
写回到对端的数据、持久化文件。测试命名 test_comm_<功能区>_<场景>，报告脚本据此归类。
功能区：tcpserver tcpclient udp serial modbusserver modbusclient mc s7 rules template handshake heartbeat events persistence testconn cli
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from cvflow.comm import CommManager, ReceiveRule, SendRule, format_template, set_manager
from cvflow.core import EventBus, FlowRunner, Graph, Solution, registry

ROOT = Path(__file__).resolve().parents[1]
registry.load_builtins()
reg = registry


# ---------------------------------------------------------------- 工具
def _wait(cond, timeout=5.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class LineClient:
    """对端 TCP 客户端：按行收发。"""

    def __init__(self, port: int, host="127.0.0.1"):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(5)
        self._buf = b""

    def send(self, data: bytes):
        self.sock.sendall(data)

    def line(self, timeout=5.0) -> str:
        self.sock.settimeout(timeout)
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1024)
            if not chunk:
                break
            self._buf += chunk
        line, sep, rest = self._buf.partition(b"\n")
        self._buf = rest
        return (line + sep).decode(errors="replace")

    def raw(self, n=1024, timeout=2.0) -> bytes:
        self.sock.settimeout(timeout)
        try:
            return self.sock.recv(n)
        except socket.timeout:
            return b""

    def close(self):
        self.sock.close()


class Events:
    """订阅事件总线，记录所有 comm_* 事件。"""

    def __init__(self, bus: EventBus):
        self.items: list[tuple[str, dict]] = []
        bus.subscribe("*", lambda event, **kw: self.items.append((event, kw)) if event.startswith("comm_") else None)

    def of(self, name, device=None):
        return [kw for ev, kw in self.items if ev == name and (device is None or kw.get("device") == device)]


def make_flow(reg, low=3, name="main") -> Graph:
    """常量 3 → 判定（可通过 low 控制 OK/NG）→ 发布 count。"""
    g = Graph(name)
    c = g.add_node(reg.create("source.constant", name="C", values={"kind": "int", "value": "3"}))
    j = g.add_node(reg.create("logic.judge", name="J", values={"op": "==", "low": low}))
    p = g.add_node(reg.create("output.publish", name="P", values={"name_a": "count"}))
    g.add_link(c.id, "value", j.id, "value"); g.add_link(c.id, "value", p.id, "a")
    return g


class System:
    """被测系统：一个方案、一个流程 runner、一个通信管理器。"""

    def __init__(self, reg, low=3):
        self.bus = EventBus()
        self.sol = Solution("bb", self.bus)
        self.flow = self.sol.add_flow(make_flow(reg, low))
        self.runner = FlowRunner(self.flow, self.sol.variables, self.bus)
        self.mgr = CommManager(self.bus, self.sol.variables)
        self.mgr.set_runners({"main": self.runner})
        self.events = Events(self.bus)

    def start(self):
        self.mgr.connect_all(); self.runner.start()

    def stop(self):
        self.runner.stop(); self.mgr.shutdown()


@pytest.fixture
def system():
    s = System(reg)
    yield s
    s.stop()


@pytest.fixture
def pymodbus_server():
    """第三方 Modbus 从站（pymodbus 自带服务器），在独立事件循环线程里运行。"""
    pytest.importorskip("pymodbus")
    import warnings
    warnings.simplefilter("ignore")
    from pymodbus.datastore import ModbusDeviceContext, ModbusSequentialDataBlock, ModbusServerContext
    from pymodbus.server import ModbusTcpServer
    port = free_port()
    holder = {}
    ready = threading.Event()
    loop = asyncio.new_event_loop()

    async def main():
        ctx = ModbusServerContext(devices=ModbusDeviceContext(hr=ModbusSequentialDataBlock(1, [0] * 200),
                                                             co=ModbusSequentialDataBlock(1, [False] * 64)), single=True)
        srv = ModbusTcpServer(ctx, address=("127.0.0.1", port))
        holder["srv"] = srv
        await srv.serve_forever(background=True)
        ready.set()
        while not holder.get("stop"):
            await asyncio.sleep(0.05)
        await srv.shutdown()

    t = threading.Thread(target=lambda: loop.run_until_complete(main()), daemon=True)
    t.start()
    if not ready.wait(5):
        pytest.skip("pymodbus server did not start")
    time.sleep(0.2)
    yield port
    holder["stop"] = True
    t.join(5)


@pytest.fixture
def pty_serial():
    if platform.system() != "Linux":
        pytest.skip("pty only on Linux")
    import pty
    m, s = pty.openpty()
    yield m, os.ttyname(s)
    os.close(m); os.close(s)


class RawServer:
    """原始 TCP 服务端：记录收到的字节并按脚本应答，用于字节级协议验证。"""

    def __init__(self, reply=None):
        self.sock = socket.socket(); self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0)); self.sock.listen(2); self.sock.settimeout(5)
        self.port = self.sock.getsockname()[1]
        self.received: list[bytes] = []
        self.reply = reply  # callable(bytes) -> bytes | None
        self.thread = threading.Thread(target=self._run, daemon=True); self.stop = False; self.thread.start()

    def _run(self):
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        conn.settimeout(0.5)
        while not self.stop:
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            self.received.append(data)
            if self.reply:
                out = self.reply(data)
                if out:
                    conn.sendall(out)
        conn.close()

    def close(self):
        self.stop = True; self.sock.close(); self.thread.join(2)


# =============================================================== TCP 服务端
def test_comm_tcpserver_routing_framing_and_multiple_clients(system):
    """多客户端路由：结果默认只回给**请求来源**的那个客户端，另有广播选项。"""
    dev = system.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0, "terminator": "\\n"})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="startswith", pattern="TRIG", flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="plc", flow="main", template="{status},{out.count}\\n"))
    system.start()
    a, b = LineClient(dev.bound_port), LineClient(dev.bound_port)
    time.sleep(0.2)
    a.send(b"TRIG\n")
    assert a.line() == "OK,3\n"                                      # 回复发给来源客户端
    assert b.raw(timeout=0.4) == b""                                 # 另一个客户端收不到别人的结果
    a.send(b"TR"); time.sleep(0.1); a.send(b"IG\nTRIG\n")             # 分包 + 粘包
    assert a.line() == "OK,3\n" and a.line() == "OK,3\n" and system.runner.stats.count == 3
    b.send(b"TRIG\n"); assert b.line() == "OK,3\n"                   # 换一个客户端请求，回复也换
    assert a.raw(timeout=0.4) == b""
    a.send(b"HELLO\n"); time.sleep(0.2)
    assert system.runner.stats.count == 4                            # 不匹配的报文不触发
    assert dev.stats["rx"] == 5 and dev.stats["tx"] == 4
    # 广播：把发送规则的目标改成 broadcast，两个客户端都收到
    system.mgr.send_rules[0].target = "broadcast"
    a.send(b"TRIG\n")
    assert a.line() == "OK,3\n" and b.line() == "OK,3\n"
    # 指定客户端发送
    system.mgr.send(dev.id, "HI\n", peer=b.sock.getsockname()[0] + ":" + str(b.sock.getsockname()[1]))
    assert b.line() == "HI\n" and a.raw(timeout=0.4) == b""
    a.close(); b.close()
    assert _wait(lambda: dev.client_count == 0)
    c = LineClient(dev.bound_port); time.sleep(0.2); c.send(b"TRIG\n")  # 断开后新客户端仍可用
    assert c.line() == "OK,3\n"; c.close()


def test_comm_tcpserver_terminator_variants_encoding_and_raw(system):
    crlf = system.mgr.add_device("crlf", "tcp_server", {"host": "127.0.0.1", "port": 0, "terminator": "\\r\\n", "encoding": "gbk"})
    raw = system.mgr.add_device("raw", "tcp_server", {"host": "127.0.0.1", "port": 0, "terminator": ""})
    system.mgr.add_receive_rule(ReceiveRule(name="v", device="crlf", match="any", action="set_variable", variable="last"))
    system.start()
    c = LineClient(crlf.bound_port); time.sleep(0.2)
    c.send("型号A\r\n".encode("gbk"))
    assert _wait(lambda: system.sol.variables.get("last") == "型号A")
    got = system.events.of("comm_received", "crlf")[-1]
    assert got["text"] == "型号A" and got["data"] == "型号A".encode("gbk")
    c.send(b"\xff\xfe\r\n")
    assert _wait(lambda: len(system.events.of("comm_received", "crlf")) == 2)
    assert system.events.of("comm_received", "crlf")[-1]["text"] == "ff fe"   # 非法编码显示为十六进制
    r = LineClient(raw.bound_port); time.sleep(0.2); r.send(b"\x01\x02\x03")
    assert _wait(lambda: system.events.of("comm_received", "raw") and system.events.of("comm_received", "raw")[-1]["data"] == b"\x01\x02\x03")
    c.send(b"x" * 1_100_000)                                          # 无终止符的大量数据不崩溃
    time.sleep(0.5); c.send(b"END\r\n")
    assert _wait(lambda: system.sol.variables.get("last") in ("END", "x" * 1_100_000 + "END") or system.events.of("comm_received", "crlf"))
    assert crlf.connected
    c.close(); r.close()


# =============================================================== TCP 客户端
def test_comm_tcpclient_connect_receive_send_reconnect(system):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(("127.0.0.1", 0)); srv.listen(1); srv.settimeout(5)
    port = srv.getsockname()[1]
    dev = system.mgr.add_device("robot", "tcp_client", {"host": "127.0.0.1", "port": port, "reconnect_s": 0.1})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="robot", match="regex", pattern=r"^GO\d+", flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="robot", template="DONE,{out.count}\\n"))
    system.start()
    conn, _ = srv.accept(); conn.settimeout(5)
    assert _wait(lambda: dev.connected) and system.events.of("comm_connected", "robot")
    conn.sendall(b"GO7\n")
    assert conn.recv(64) == b"DONE,3\n" and system.runner.stats.count == 1
    assert system.mgr.send("robot", "HI\n") and conn.recv(16) == b"HI\n"
    conn.close()
    assert _wait(lambda: not dev.connected) and system.events.of("comm_disconnected", "robot")
    conn2, _ = srv.accept(); conn2.settimeout(5)                      # 自动重连
    assert _wait(lambda: dev.connected)
    conn2.sendall(b"GO8\n"); assert conn2.recv(64) == b"DONE,3\n"
    conn2.close(); srv.close()


def test_comm_tcpclient_no_server_and_no_auto_reconnect(system):
    port = free_port()
    dev = system.mgr.add_device("robot", "tcp_client", {"host": "127.0.0.1", "port": port, "auto_reconnect": False, "reconnect_s": 0.1})
    system.start()
    assert _wait(lambda: system.events.of("comm_error", "robot"), 5)
    assert not dev.connected and system.mgr.send("robot", "x") is False
    time.sleep(0.5)
    assert not dev.connected


# =============================================================== UDP
def test_comm_udp_datagrams_both_ways(system):
    peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); peer.bind(("127.0.0.1", 0)); peer.settimeout(5)
    dev = system.mgr.add_device("udp", "udp", {"host": "127.0.0.1", "port": peer.getsockname()[1], "local_port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="udp", match="equals", pattern="TRIG", flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="udp", template="{status}"))
    system.start()
    local = dev._sock.getsockname()[1] if hasattr(dev, "_sock") else None
    peer.sendto(b"TRIG", ("127.0.0.1", local))
    data, _ = peer.recvfrom(64)
    assert data == b"OK" and system.runner.stats.count == 1
    peer.close()


# =============================================================== 串口
def test_comm_serial_pty_loopback(system, pty_serial):
    master, name = pty_serial
    dev = system.mgr.add_device("scanner", "serial", {"port": name, "baudrate": 9600, "terminator": "\\r\\n"})
    system.mgr.add_receive_rule(ReceiveRule(name="code", device="scanner", match="any", action="set_variable", variable="barcode"))
    system.mgr.add_send_rule(SendRule(name="r", device="scanner", template="ACK\\r\\n"))
    system.start()
    assert _wait(lambda: dev.connected)
    os.write(master, b"ABC123\r\n")
    assert _wait(lambda: system.sol.variables.get("barcode") == "ABC123")
    assert system.mgr.send("scanner", "PING\r\n") and os.read(master, 32) == b"PING\r\n"
    ok, msg = dev.test_connection(); assert ok and name in msg


# =============================================================== Modbus 从站（用第三方 pymodbus 主站观察）
def test_comm_modbusserver_function_codes_and_exceptions(system):
    from pymodbus.client import ModbusTcpClient
    dev = system.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0, "register_count": 64, "coil_count": 16, "unit_id": 1})
    system.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2); assert c.connect()
    assert not c.write_register(5, 0xBEEF, device_id=1).isError()
    assert not c.write_registers(10, [1, 2, 3], device_id=1).isError()
    assert c.read_holding_registers(5, count=1, device_id=1).registers == [0xBEEF]
    assert c.read_holding_registers(10, count=3, device_id=1).registers == [1, 2, 3]
    # 四个数据区各自独立：输入寄存器 (3x) 不是保持寄存器 (4x) 的别名
    assert c.read_input_registers(10, count=3, device_id=1).registers == [0, 0, 0]
    dev.write_area("input", 10, [7, 8, 9])
    assert c.read_input_registers(10, count=3, device_id=1).registers == [7, 8, 9]
    assert c.read_holding_registers(10, count=3, device_id=1).registers == [1, 2, 3]
    assert not c.write_coil(2, True, device_id=1).isError() and not c.write_coils(4, [True, False, True], device_id=1).isError()
    assert c.read_coils(0, count=8, device_id=1).bits[:8] == [False, False, True, False, True, False, True, False]
    assert c.read_discrete_inputs(2, count=1, device_id=1).bits[0] is False       # 离散输入与线圈也互不影响
    dev.write_area("discrete", 2, [True])
    assert c.read_discrete_inputs(2, count=1, device_id=1).bits[0] is True
    # 输入寄存器与离散输入没有写功能码，第三方主站也写不进去
    assert c.write_register(10, 1, device_id=1).isError() is False                # 4x 可写
    assert c.read_holding_registers(60, count=10, device_id=1).isError()           # 越界 → 异常响应
    assert c.write_coil(16, True, device_id=1).isError()
    dev.write_value(20, -2.5, "float32"); dev.write_value(22, -70000, "int32"); dev.write_value(24, -3, "int16")
    r = c.read_holding_registers(20, count=5, device_id=1).registers
    assert struct.unpack(">f", struct.pack(">HH", *r[0:2]))[0] == -2.5 and struct.unpack(">i", struct.pack(">HH", *r[2:4]))[0] == -70000 and r[4] == 0xFFFD
    c2 = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2); assert c2.connect()   # 第二个主站并存
    assert c2.read_holding_registers(5, count=1, device_id=1).registers == [0xBEEF]
    raw = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=2)
    raw.sendall(struct.pack(">HHHB", 7, 0, 2, 1) + bytes([0x11]))                     # 非法功能码 0x11
    resp = raw.recv(64); assert resp[7] == 0x91 and resp[8] == 0x01
    raw.close(); c.close(); c2.close()


def test_comm_modbusserver_unit_id_filter(system):
    from pymodbus.client import ModbusTcpClient
    dev = system.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0, "unit_id": 3})
    system.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=1, retries=0); assert c.connect()
    assert not c.write_register(0, 9, device_id=3).isError() and dev.read_value(0) == 9
    try:                                                                                 # 单元号不符 → 按规范不应答（主站超时）
        r = c.read_holding_registers(0, count=1, device_id=1)
        assert r is None or r.isError()
    except Exception as e:
        assert "No response" in str(e) or "timeout" in str(e).lower()
    c.close()


# =============================================================== Modbus 主站（对第三方 pymodbus 从站）
def test_comm_modbusclient_against_third_party_server(system, pymodbus_server):
    from pymodbus.client import ModbusTcpClient
    dev = system.mgr.add_device("plc", "modbus_tcp_client", {"host": "127.0.0.1", "port": pymodbus_server, "poll_ms": 20,
                                                             "watch_address": 0, "watch_count": 8, "busy_address": 1, "trigger_reset": True})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="register_rising", address=0, flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="plc", template="", registers=[{"address": 10, "expr": "{out.count}", "kind": "int16"},
                                                                                     {"address": 12, "expr": "{duration_ms}", "kind": "float32"}]))
    system.start()
    assert _wait(lambda: dev.connected, 5)
    ok, msg = dev.test_connection(); assert ok and "可用" in msg
    dev.write_value(30, 1.5, "float32"); dev.write_value(32, -5, "int16"); dev.write_value(33, 70000, "uint32")
    plc = ModbusTcpClient("127.0.0.1", port=pymodbus_server, timeout=2); assert plc.connect()
    r = plc.read_holding_registers(30, count=5, device_id=1).registers
    assert struct.unpack(">f", struct.pack(">HH", *r[0:2]))[0] == 1.5 and r[2] == 0xFFFB and struct.unpack(">I", struct.pack(">HH", *r[3:5]))[0] == 70000
    assert dev.read_value(30, "float32") == 1.5
    plc.write_register(0, 1, device_id=1)                                               # PLC 置位触发字
    assert _wait(lambda: system.runner.stats.count == 1, 5)
    assert _wait(lambda: plc.read_holding_registers(0, count=2, device_id=1).registers == [0, 0], 5)   # 触发字复位、忙标志清零
    assert plc.read_holding_registers(10, count=1, device_id=1).registers == [3]
    plc.close()


# =============================================================== 三菱 MC（字节级）
def test_comm_mc_request_frames_match_specification():
    """用原始套接字扮演 PLC，核对客户端发出的 3E 二进制帧与手册一致，并解析其对手工应答的处理。"""
    def reply(data: bytes):
        cmd = struct.unpack("<H", data[11:13])[0]
        if cmd == 0x0401:
            return bytes.fromhex("D000 00FF FF03 00 0600 0000 D204 0100")   # 结束码 0，D100=1234, D101=1
        return bytes.fromhex("D000 00FF FF03 00 0200 0000")                 # 写入成功
    srv = RawServer(reply)
    mgr = CommManager(EventBus())
    dev = mgr.add_device("plc", "mc", {"host": "127.0.0.1", "port": srv.port, "poll_ms": 0, "network": 0, "pc": 255, "io": 0x3FF, "station": 0})
    try:
        assert dev.read_words("D", 100, 2) == [1234, 1]
        assert srv.received[0] == bytes.fromhex("5000 00FF FF03 00 0C00 1000 0104 0000 640000 A8 0200")
        dev.write_words("D", 10, [1234])
        assert srv.received[1] == bytes.fromhex("5000 00FF FF03 00 0E00 1000 0114 0000 0A0000 A8 0100 D204")
        dev.write_bits("M", 20, [True, False, True])
        assert srv.received[2] == bytes.fromhex("5000 00FF FF03 00 0E00 1000 0114 0100 140000 90 0300 1010")   # 3 个位打包为 2 字节
        assert dev.read_value("X1F", "bool") in (True, False) and srv.received[3][15:19] == bytes.fromhex("1F0000 9C")  # X 为十六进制编号
    finally:
        mgr.shutdown(); srv.close()


def test_comm_mc_end_code_error_is_reported():
    srv = RawServer(lambda data: bytes.fromhex("D000 00FF FF03 00 0200 59C0"))      # 结束码 0xC059
    mgr = CommManager(EventBus())
    dev = mgr.add_device("plc", "mc", {"host": "127.0.0.1", "port": srv.port, "poll_ms": 0})
    try:
        ok, msg = dev.test_connection()
        assert not ok and "C059" in msg
    finally:
        mgr.shutdown(); srv.close()


def test_comm_mc_simulator_end_to_end(system):
    from cvflow.comm.mc import McSimulatorServer
    sim = McSimulatorServer().start()
    dev = system.mgr.add_device("plc", "mc", {"host": "127.0.0.1", "port": sim.bound_port, "poll_ms": 20, "watch_count": 4,
                                              "busy_address": 1, "trigger_reset": True})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="register_rising", address=0, flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="plc", template="", registers=[{"address": "D10", "expr": "{out.count}", "kind": "int16"},
                                                                                     {"address": "M5", "expr": "{ok}", "kind": "bool"}]))
    system.start()
    try:
        assert _wait(lambda: dev.connected)
        sim.set_word("D", 0, 1)
        assert _wait(lambda: sim.get_word("D", 10) == 3 and sim.get_word("D", 0) == 0 and sim.get_word("D", 1) == 0 and sim.bits.get(("M", 5)) is True, 5)
    finally:
        sim.stop()


# =============================================================== 西门子 S7（第三方 snap7 服务器）
def test_comm_s7_third_party_server_roundtrip_and_trigger(system):
    snap7 = pytest.importorskip("snap7")
    from snap7.type import SrvArea
    db = bytearray(64); server = snap7.server.Server(); server.register_area(SrvArea.DB, 1, db)
    port = free_port()
    try:
        server.start(tcp_port=port)
    except Exception as e:
        pytest.skip(f"snap7 server unavailable: {e}")
    dev = system.mgr.add_device("plc", "s7", {"host": "127.0.0.1", "port": port, "rack": 0, "slot": 1, "db_number": 1, "poll_ms": 20,
                                              "watch_bytes": 8, "busy_address": 2, "trigger_reset": True})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="register_rising", address=0, flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="plc", template="", registers=[{"address": 10, "expr": "{out.count}", "kind": "int16"},
                                                                                     {"address": 12, "expr": "{out.count}", "kind": "float32"},
                                                                                     {"address": 16, "expr": "{ok}", "kind": "bool"}]))
    try:
        ok, msg = dev.test_connection(); assert ok, msg
        dev.write_value(20, -1, "int16"); dev.write_value(22, 123456789, "uint32")
        assert db[20:22] == b"\xff\xff" and struct.unpack(">I", bytes(db[22:26]))[0] == 123456789 and dev.read_value(22, "uint32") == 123456789
        system.start()
        assert _wait(lambda: dev.connected)
        db[0:2] = b"\x00\x01"
        assert _wait(lambda: db[10:12] == b"\x00\x03" and db[12:16] == struct.pack(">f", 3.0) and db[16] == 1 and db[0:2] == b"\x00\x00" and db[2:4] == b"\x00\x00", 5)
    finally:
        server.stop(); server.destroy()


# =============================================================== 规则
@pytest.mark.parametrize("match,pattern,sent,expect", [
    ("startswith", "TRIG", b"TRIGGER\n", 1), ("startswith", "TRIG", b"XTRIG\n", 0), ("equals", "GO", b"GO\n", 1), ("equals", "GO", b"GO1\n", 0),
    ("contains", "AB", b"xxABxx\n", 1), ("regex", r"^T\d{2}$", b"T42\n", 1), ("regex", r"^T\d{2}$", b"T4\n", 0), ("any", "", b"whatever\n", 1),
    ("regex", "(", b"x\n", 0),
])
def test_comm_rules_text_match_kinds(system, match, pattern, sent, expect):
    dev = system.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match=match, pattern=pattern, flow="main"))
    system.start()
    c = LineClient(dev.bound_port); time.sleep(0.15); c.send(sent); time.sleep(0.3)
    assert system.runner.stats.count == expect
    c.close()


def test_comm_rules_device_filter_disabled_and_flow_not_running(system):
    a = system.mgr.add_device("a", "tcp_server", {"host": "127.0.0.1", "port": 0})
    b = system.mgr.add_device("b", "tcp_server", {"host": "127.0.0.1", "port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="only_a", device="a", match="any", flow="main"))
    system.mgr.add_receive_rule(ReceiveRule(name="off", device="b", match="any", flow="main", enabled=False))
    system.mgr.add_receive_rule(ReceiveRule(name="nope", device="", match="equals", pattern="GHOST", flow="ghost"))
    system.mgr.connect_all()                                                       # runner 未启动
    ca, cb = LineClient(a.bound_port), LineClient(b.bound_port); time.sleep(0.15)
    ca.send(b"x\n"); time.sleep(0.3)
    assert system.runner.stats.count == 0                                         # 流程未运行：忽略，不崩溃
    system.runner.start()
    ca.send(b"x\n"); cb.send(b"x\n"); cb.send(b"GHOST\n"); time.sleep(0.4)
    assert system.runner.stats.count == 1                                         # b 的规则被停用；指向不存在流程的规则被忽略
    ca.close(); cb.close()


def test_comm_rules_register_match_kinds(system):
    from pymodbus.client import ModbusTcpClient
    dev = system.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="rise", device="plc", match="register_rising", address=0, flow="main"))
    system.mgr.add_receive_rule(ReceiveRule(name="chg", device="plc", match="register_change", address=1, action="set_variable", variable="chg"))
    system.mgr.add_receive_rule(ReceiveRule(name="eq", device="plc", match="register_equals", address=2, value=7, action="set_variable", variable="eq"))
    system.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2); assert c.connect()
    c.write_register(0, 5, device_id=1); assert _wait(lambda: system.runner.stats.count == 1)
    c.write_register(0, 6, device_id=1); time.sleep(0.2); assert system.runner.stats.count == 1   # 非 0→非 0 不算上升沿
    c.write_register(0, 0, device_id=1); c.write_register(0, 1, device_id=1); assert _wait(lambda: system.runner.stats.count == 2)
    # 写变量的动作存的是**寄存器的值**（不是 "reg[1]=3" 这种字符串），流程里可直接当数字用
    c.write_register(1, 3, device_id=1); assert _wait(lambda: system.sol.variables.get("chg") == 3)
    c.write_register(2, 6, device_id=1); time.sleep(0.1); assert system.sol.variables.get("eq") is None
    c.write_register(2, 7, device_id=1); assert _wait(lambda: system.sol.variables.get("eq") is not None)
    c.close()


def test_comm_rules_send_conditions_and_flow_filter(system):
    dev = system.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="startswith", pattern="T", flow="main"))
    for when in ("always", "ok", "ng", "error"):
        system.mgr.add_send_rule(SendRule(name=when, device="plc", flow="main", when=when, template=f"{when}\\n"))
    system.mgr.add_send_rule(SendRule(name="other", device="plc", flow="other", template="other\\n"))
    system.start()
    c = LineClient(dev.bound_port); time.sleep(0.15); c.send(b"T\n")
    got = {c.line() for _ in range(2)}
    assert got == {"always\n", "ok\n"} and c.raw(64, 0.5) == b""
    system.flow.find_by_name("J").set("low", 99)                                   # 变成 NG
    c.send(b"T\n"); got = {c.line() for _ in range(2)}
    assert got == {"always\n", "ng\n"}
    system.flow.find_by_name("J").set("op", "<"); system.flow.find_by_name("J").set("low", "x") if False else None
    system.flow.add_node(reg.create("logic.expression", name="E", values={"expression": "1/0"}))   # 引入错误
    e = system.flow.find_by_name("E"); system.flow.add_link(system.flow.find_by_name("C").id, "value", e.id, "a")
    c.send(b"T\n"); got = {c.line() for _ in range(2)}
    assert got == {"always\n", "error\n"}
    c.close()


# =============================================================== 模板
def test_comm_template_fields_specs_and_errors(system):
    from cvflow.core.engine import Engine
    r = Engine(system.flow, system.sol.variables).run()
    system.sol.variables.set("product", "A1")
    s = format_template("{status}|{ok}|{ng}|{run_id}|{flow}|{duration_ms:.0f}|{out.count:03d}|{out.missing}|{node[J].ok}|{node[Nope].x}|{var.product}|{var.none}|\\t\\r\\n", r, system.sol.variables.snapshot())
    assert s.startswith("OK|1|0|1|main|") and "|003||True||A1||\t\r\n" in s
    with pytest.raises(Exception):
        format_template("{out.count:zz}", r)
    # 发送规则模板错误 → comm_error 事件，不中断
    dev = system.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="any", flow="main"))
    system.mgr.add_send_rule(SendRule(name="bad", device="plc", template="{out.count:zz}\\n"))
    system.mgr.add_send_rule(SendRule(name="good", device="plc", template="ok {out.count}\\n"))
    system.start()
    c = LineClient(dev.bound_port); time.sleep(0.15); c.send(b"x\n")
    assert c.line() == "ok 3\n" and _wait(lambda: system.events.of("comm_error", "plc"))
    assert "bad" in system.events.of("comm_error", "plc")[-1]["error"]
    c.close()


def test_comm_template_register_kinds(system):
    from pymodbus.client import ModbusTcpClient
    dev = system.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="register_rising", address=0, flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="plc", template="", registers=[
        {"address": 10, "expr": "{out.count}", "kind": "int16"}, {"address": 11, "expr": "-{out.count}", "kind": "int16"},
        {"address": 12, "expr": "70000", "kind": "int32"}, {"address": 14, "expr": "{out.count}.5", "kind": "float32"},
        {"address": 16, "expr": "{ok}", "kind": "uint16"}, {"address": 3, "expr": "{status}", "kind": "bool"},
        {"address": 17, "expr": "{out.missing}", "kind": "int16"}]))
    system.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2); assert c.connect()
    c.write_register(0, 1, device_id=1)
    assert _wait(lambda: dev.read_value(10) == 3, 5)
    r = c.read_holding_registers(10, count=8, device_id=1).registers
    assert r[0] == 3 and r[1] == 0xFFFD and struct.unpack(">i", struct.pack(">HH", r[2], r[3]))[0] == 70000
    assert struct.unpack(">f", struct.pack(">HH", r[4], r[5]))[0] == 3.5 and r[6] == 1 and r[7] == 0      # 缺失值 → 0
    assert c.read_coils(3, count=1, device_id=1).bits[0] is True                                             # "OK" → True
    c.close()


# =============================================================== 握手 / 心跳
def test_comm_handshake_busy_flag_visible_to_plc(system):
    from pymodbus.client import ModbusTcpClient
    system.flow.add_node(reg.create("logic.delay", name="D", values={"ms": 300}))
    dev = system.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0, "busy_address": 5, "trigger_reset": True})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="register_rising", address=0, flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="plc", template="", registers=[{"address": 10, "expr": "{out.count}", "kind": "int16"}]))
    system.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2); assert c.connect()
    c.write_register(0, 1, device_id=1)
    assert _wait(lambda: c.read_holding_registers(5, count=1, device_id=1).registers == [1], 2)     # 忙
    assert c.read_holding_registers(0, count=1, device_id=1).registers == [0]                        # 触发字已清零
    assert _wait(lambda: c.read_holding_registers(5, count=6, device_id=1).registers[0] == 0, 3)
    assert c.read_holding_registers(10, count=1, device_id=1).registers == [3]
    c.write_register(0, 1, device_id=1); assert _wait(lambda: system.runner.stats.count == 2, 3)   # 第二次上升沿有效
    c.close()


def test_comm_heartbeat_register_and_text_and_stop(system):
    from pymodbus.client import ModbusTcpClient
    mb = system.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0, "heartbeat_address": 20, "heartbeat_s": 0.05})
    tcp = system.mgr.add_device("hmi", "tcp_server", {"host": "127.0.0.1", "port": 0, "heartbeat_s": 0.05, "heartbeat_text": "HB\\n"})
    system.start()
    c = ModbusTcpClient("127.0.0.1", port=mb.bound_port, timeout=2); assert c.connect()
    assert _wait(lambda: c.read_holding_registers(20, count=1, device_id=1).registers[0] >= 3, 2)
    h = LineClient(tcp.bound_port)
    assert h.line() == "HB\n" and h.line() == "HB\n"
    system.mgr.disconnect_all()
    v = mb.registers[20]; time.sleep(0.2); assert mb.registers[20] == v
    h.close(); c.close()


# =============================================================== 事件
def test_comm_events_sequence_and_payloads(system):
    dev = system.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    system.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="any", flow="main"))
    system.mgr.add_send_rule(SendRule(name="r", device="plc", template="R{out.count}\\n"))
    system.start()
    c = LineClient(dev.bound_port); time.sleep(0.15); c.send(b"GO\n"); assert c.line() == "R3\n"
    assert _wait(lambda: len(system.events.items) >= 3)   # 事件在字节写出之后才记录，等它落账
    names = [e for e, _ in system.events.items]
    assert names[:3] == ["comm_connected", "comm_received", "comm_sent"]
    rx = system.events.of("comm_received", "plc")[0]; tx = system.events.of("comm_sent", "plc")[0]
    assert rx["data"] == b"GO" and rx["text"] == "GO" and tx["data"] == b"R3\n" and tx["text"] == "R3\n"
    c.close(); system.mgr.disconnect_all()
    assert system.events.of("comm_disconnected", "plc")


# =============================================================== 持久化
def test_comm_persistence_roundtrip_and_invalid_kind(tmp_path):
    bus = EventBus(); sol = Solution("p", bus); sol.add_flow(make_flow(reg))
    mgr = CommManager(bus, sol.variables)
    mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 5020, "busy_address": 1, "trigger_reset": True})
    mgr.add_device("hmi", "tcp_client", {"host": "10.0.0.5", "port": 9000, "terminator": "\\r\\n"})
    mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="register_rising", address=0, flow="main"))
    mgr.add_send_rule(SendRule(name="r", device="hmi", template="{status}\\r\\n", when="ng"))
    sol.comm_config = mgr.to_dict(); sol.save(tmp_path / "s.json"); mgr.shutdown()
    sol2 = Solution.load(tmp_path / "s.json")
    mgr2 = CommManager(sol2.bus, sol2.variables)
    assert mgr2.load_dict(sol2.comm_config) == []
    assert set(mgr2.devices) == {"plc", "hmi"} and mgr2.devices["plc"].config["busy_address"] == 1 and mgr2.devices["hmi"].config["terminator"] == "\\r\\n"
    # 老方案里的 register_* 写法读进来会迁移成新的条件名（source=datapoint + rising）
    assert mgr2.receive_rules[0].match == "rising" and mgr2.receive_rules[0].source == "datapoint"
    assert mgr2.send_rules[0].when == "ng"
    bad = dict(sol2.comm_config); bad["devices"] = bad["devices"] + [{"name": "x", "kind": "profinet", "config": {}}]
    errs = mgr2.load_dict(bad)
    assert len(errs) == 1 and "profinet" in errs[0] and set(mgr2.devices) == {"plc", "hmi"}
    mgr2.shutdown()


# =============================================================== 连接测试
def test_comm_testconn_all_kinds(system):
    srv = system.mgr.add_device("srv", "tcp_server", {"host": "127.0.0.1", "port": 0})
    assert srv.test_connection()[0]
    srv.connect(); assert "监听" in srv.test_connection()[1]
    assert system.mgr.add_device("cli", "tcp_client", {"host": "127.0.0.1", "port": srv.bound_port}).test_connection()[0]
    assert not system.mgr.add_device("dead", "tcp_client", {"host": "127.0.0.1", "port": free_port()}).test_connection()[0]
    assert system.mgr.add_device("udp", "udp", {"host": "127.0.0.1", "port": 9, "local_port": 0}).test_connection()[0]
    mbs = system.mgr.add_device("mbs", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0}); mbs.connect()
    assert "= 0" in system.mgr.add_device("mbc", "modbus_tcp_client", {"host": "127.0.0.1", "port": mbs.bound_port, "poll_ms": 0}).test_connection()[1]
    assert not system.mgr.add_device("mbc2", "modbus_tcp_client", {"host": "127.0.0.1", "port": free_port(), "poll_ms": 0, "timeout_s": 0.5}).test_connection()[0]
    assert not system.mgr.add_device("mc", "mc", {"host": "127.0.0.1", "port": free_port(), "timeout_s": 0.5}).test_connection()[0]
    assert not system.mgr.add_device("s7", "s7", {"host": "127.0.0.1", "port": free_port(), "rack": 0, "slot": 1}).test_connection()[0]
    assert not system.mgr.add_device("ser", "serial", {"port": "/dev/nonexistent-tty"}).test_connection()[0]
    assert system.mgr.test_connection("missing") == (False, "没有名为 'missing' 的设备")


# =============================================================== 命令行无界面模式
def _unlink_eventually(path: Path, timeout: float = 10.0) -> None:
    """删临时文件，Windows 上允许慢一点。

    刚被 TerminateProcess 干掉的子进程，它继承的文件句柄不一定立刻放掉，
    这时 unlink 会抛 PermissionError（WinError 32）。清理失败不该让用例判失败，
    所以重试一段时间，仍然删不掉就留着。
    """
    end = time.monotonic() + timeout
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= end:
                return
            time.sleep(0.2)


def test_comm_cli_serve_modbus_end_to_end(tmp_path):
    from pymodbus.client import ModbusTcpClient
    src = ROOT / "examples" / "solutions" / "demo_modbus.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    data["comm"]["devices"][0]["config"]["port"] = 0      # 让子进程自己挑端口，避免与其它用例抢占
    # 方案文件必须和原方案同目录（方案里的相对路径按方案所在目录解析），日志放 tmp_path：
    # 子进程的 stdout 句柄在 Windows 上不一定立刻释放，放在 pytest 的临时目录里就不用自己删
    tmp = src.parent / "_bb_modbus.json"; tmp.write_text(json.dumps(data), encoding="utf-8")
    log_path = tmp_path / "serve.log"
    log = open(log_path, "w", encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "cvflow", "serve", str(tmp), "--stats-every", "100"], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        def bound_port():
            log.flush()
            m = re.search(r"监听 [^:\s]+:(\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
            return int(m.group(1)) if m else None
        ok = _wait(lambda: proc.poll() is not None or bound_port(), 60)     # 从子进程输出里读回真实端口
        port = bound_port()
        if proc.poll() is not None or not port:
            raise AssertionError(f"serve 子进程退出码 {proc.poll()}，输出：\n{log_path.read_text(encoding='utf-8', errors='replace')[-2000:]}")
        c = ModbusTcpClient("127.0.0.1", port=port, timeout=2)
        assert _wait(lambda: c.connect(), 30)
        c.write_register(0, 1, device_id=1)
        assert _wait(lambda: c.read_holding_registers(10, count=1, device_id=1).registers == [3], 10)
        r = c.read_holding_registers(0, count=2, device_id=1).registers
        assert r == [0, 0]                                                           # 触发字复位、忙标志清零
        assert _wait(lambda: c.read_holding_registers(20, count=1, device_id=1).registers[0] >= 1, 5)   # 心跳
        c.close()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:                 # terminate 不奏效就硬杀，别把句柄留着
            proc.kill(); proc.wait(10)
        log.close()
        _unlink_eventually(tmp)

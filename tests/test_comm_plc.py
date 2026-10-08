import socket
import time

import pytest

from cvflow.core import EventBus, FlowRunner, Graph, Solution
from cvflow.comm import CommManager, ReceiveRule, SendRule
from cvflow.comm.mc import McDevice, McSimulatorServer, parse_address


def _wait(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def _flow(reg, delay_ms=0):
    g = Graph("main")
    c = g.add_node(reg.create("source.constant", name="C", values={"kind": "int", "value": "3"}))
    p = g.add_node(reg.create("output.publish", name="P", values={"name_a": "count"}))
    g.add_link(c.id, "value", p.id, "a")
    if delay_ms:
        d = g.add_node(reg.create("logic.delay", name="D", values={"ms": delay_ms}))
        g.add_link(c.id, "value", d.id, "value")
    return g


def test_connection_tests_for_text_and_modbus_devices():
    mgr = CommManager(EventBus())
    srv = mgr.add_device("srv", "tcp_server", {"host": "127.0.0.1", "port": 0})
    ok, msg = srv.test_connection()
    assert ok and "可用" in msg
    srv.connect()
    ok, msg = srv.test_connection()
    assert ok and "监听" in msg
    cli = mgr.add_device("cli", "tcp_client", {"host": "127.0.0.1", "port": srv.bound_port})
    ok, msg = cli.test_connection()
    assert ok and "可连接" in msg
    dead = mgr.add_device("dead", "tcp_client", {"host": "127.0.0.1", "port": 1})
    ok, msg = dead.test_connection()
    assert not ok and "失败" in msg
    udp = mgr.add_device("udp", "udp", {"host": "127.0.0.1", "port": 9, "local_port": 0})
    assert udp.test_connection()[0]
    mbs = mgr.add_device("mbs", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0})
    mbs.connect()
    mbs.write_registers(0, [42])
    mbc = mgr.add_device("mbc", "modbus_tcp_client", {"host": "127.0.0.1", "port": mbs.bound_port, "poll_ms": 0})
    ok, msg = mbc.test_connection()
    assert ok and "= 42" in msg
    assert mgr.test_connection("nope") == (False, "没有名为 'nope' 的设备")
    mgr.shutdown()


def test_modbus_handshake_busy_and_trigger_reset(reg):
    pytest.importorskip("pymodbus")
    from pymodbus.client import ModbusTcpClient
    bus = EventBus()
    sol = Solution("hs", bus)
    g = sol.add_flow(_flow(reg, delay_ms=300))
    runner = FlowRunner(g, sol.variables, bus)
    mgr = CommManager(bus, sol.variables)
    dev = mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0, "busy_address": 5, "trigger_reset": True})
    mgr.add_receive_rule(ReceiveRule(name="trig", device="plc", match="register_rising", address=0, flow="main"))
    mgr.add_send_rule(SendRule(name="res", device="plc", flow="main", template="",
                               registers=[{"address": 10, "expr": "{out.count}", "kind": "int16"}]))
    mgr.set_runners({"main": runner})
    mgr.connect_all()
    runner.start()
    try:
        c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2)
        assert c.connect()
        c.write_register(0, 1, device_id=1)
        assert _wait(lambda: dev.read_value(5) == 1, 2.0)          # busy while the flow runs
        assert dev.read_value(0) == 0                               # trigger register was reset at once
        assert _wait(lambda: dev.read_value(10) == 3 and dev.read_value(5) == 0, 3.0)  # result then busy cleared
        assert runner.stats.count == 1
        c.write_register(0, 1, device_id=1)                       # second cycle works because the edge is clean
        assert _wait(lambda: runner.stats.count == 2, 3.0)
        c.close()
    finally:
        runner.stop()
        mgr.shutdown()


def test_modbus_and_tcp_heartbeats():
    mgr = CommManager(EventBus())
    mbs = mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0, "heartbeat_address": 20, "heartbeat_s": 0.05})
    tcp = mgr.add_device("hmi", "tcp_server", {"host": "127.0.0.1", "port": 0, "heartbeat_s": 0.05, "heartbeat_text": "HB\\n"})
    mgr.connect_all()
    try:
        s = socket.create_connection(("127.0.0.1", tcp.bound_port), timeout=3)
        s.settimeout(3)
        assert _wait(lambda: mbs.read_value(20) >= 3, 2.0)
        buf = b""
        while buf.count(b"HB\n") < 2:
            buf += s.recv(64)
        s.close()
    finally:
        mgr.shutdown()
    v = mbs.read_value(20)
    time.sleep(0.2)
    assert mbs.read_value(20) == v  # heartbeat stopped with the manager


def test_mc_protocol_against_simulator(reg):
    assert parse_address("D100") == ("D", 100) and parse_address("X1F") == ("X", 31) and parse_address(7, "W") == ("W", 7)
    sim = McSimulatorServer().start()
    bus = EventBus()
    sol = Solution("mc", bus)
    g = sol.add_flow(_flow(reg))
    runner = FlowRunner(g, sol.variables, bus)
    mgr = CommManager(bus, sol.variables)
    dev: McDevice = mgr.add_device("plc", "mc", {"host": "127.0.0.1", "port": sim.bound_port, "poll_ms": 20,
                                                 "watch_address": 0, "watch_count": 4, "busy_address": 1, "trigger_reset": True})
    mgr.add_receive_rule(ReceiveRule(name="trig", device="plc", match="register_rising", address=0, flow="main"))
    mgr.add_send_rule(SendRule(name="res", device="plc", flow="main", template="",
                               registers=[{"address": "D10", "expr": "{out.count}", "kind": "int16"},
                                          {"address": "D12", "expr": "{duration_ms}", "kind": "float32"}]))
    mgr.set_runners({"main": runner})
    try:
        ok, msg = dev.test_connection()
        assert ok and "可用" in msg
        dev.write_value("D100", -5, "int16")
        assert sim.get_word("D", 100) == 0xFFFB and dev.read_value("D100", "int16") == -5
        dev.write_value("D200", 1.5, "float32")
        assert dev.read_value("D200", "float32") == 1.5
        dev.write_value("D300", 70000, "int32")
        assert dev.read_value("D300", "int32") == 70000
        dev.write_value("M10", True)
        assert dev.read_value("M10") is True and dev.read_value("M11") is False
        mgr.connect_all()
        runner.start()
        assert _wait(lambda: dev.connected)
        sim.set_word("D", 0, 1)                                     # PLC raises the trigger word
        assert _wait(lambda: runner.stats.count == 1, 3.0)
        assert _wait(lambda: sim.get_word("D", 10) == 3 and sim.get_word("D", 0) == 0 and sim.get_word("D", 1) == 0, 3.0)
        assert sim.get_word("D", 12) != 0 or sim.get_word("D", 13) != 0
    finally:
        runner.stop()
        mgr.shutdown()
        sim.stop()


def test_s7_against_snap7_server(reg):
    snap7 = pytest.importorskip("snap7")
    from snap7.type import SrvArea
    db = bytearray(64)
    server = snap7.server.Server()
    server.register_area(SrvArea.DB, 1, db)
    try:
        server.start(tcp_port=11102)
    except Exception as e:
        pytest.skip(f"snap7 server unavailable: {e}")
    bus = EventBus()
    sol = Solution("s7", bus)
    g = sol.add_flow(_flow(reg))
    runner = FlowRunner(g, sol.variables, bus)
    mgr = CommManager(bus, sol.variables)
    dev = mgr.add_device("plc", "s7", {"host": "127.0.0.1", "rack": 0, "slot": 1, "port": 11102, "db_number": 1,
                                       "poll_ms": 20, "watch_offset": 0, "watch_bytes": 8, "busy_address": 2, "trigger_reset": True})
    mgr.add_receive_rule(ReceiveRule(name="trig", device="plc", match="register_rising", address=0, flow="main"))
    mgr.add_send_rule(SendRule(name="res", device="plc", flow="main", template="",
                               registers=[{"address": 10, "expr": "{out.count}", "kind": "int16"},
                                          {"address": 20, "expr": "{duration_ms}", "kind": "float32"}]))
    mgr.set_runners({"main": runner})
    try:
        ok, msg = dev.test_connection()
        assert ok, msg
        dev.write_value(30, -1234, "int16")
        assert dev.read_value(30, "int16") == -1234 and db[30:32] == b"\xfb\x2e"
        dev.write_value(40, 2.5, "float32")
        assert dev.read_value(40, "float32") == 2.5
        mgr.connect_all()
        runner.start()
        assert _wait(lambda: dev.connected)
        db[0:2] = b"\x00\x01"                                        # PLC raises DBW0
        assert _wait(lambda: runner.stats.count == 1, 3.0)
        assert _wait(lambda: db[10:12] == b"\x00\x03" and db[0:2] == b"\x00\x00" and db[2:4] == b"\x00\x00", 3.0)
    finally:
        runner.stop()
        mgr.shutdown()
        server.stop()
        server.destroy()

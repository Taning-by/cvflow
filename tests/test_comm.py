import socket
import time

import pytest

from cvflow.core import EventBus, FlowRunner, Graph, Solution, Trigger, TriggerSource
from cvflow.comm import CommManager, ReceiveRule, SendRule, format_template
from cvflow.comm.modbus import registers_to_value, value_to_registers


def _flow(reg, name="main"):
    g = Graph(name)
    c = g.add_node(reg.create("source.constant", name="C", values={"kind": "int", "value": "3"}))
    j = g.add_node(reg.create("logic.judge", name="J", values={"op": "==", "low": 3}))
    p = g.add_node(reg.create("output.publish", name="P", values={"name_a": "count"}))
    g.add_link(c.id, "value", j.id, "value")
    g.add_link(c.id, "value", p.id, "a")
    return g


def _wait(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


class _LineReader:
    """Read one line at a time even when several replies arrive in a single TCP segment."""

    def __init__(self, sock, timeout=5.0):
        sock.settimeout(timeout)
        self._sock = sock
        self._buf = b""

    def line(self) -> str:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1024)
            if not chunk:
                break
            self._buf += chunk
        line, sep, rest = self._buf.partition(b"\n")
        self._buf = rest
        return (line + sep).decode()


def test_format_template(reg):
    from cvflow.core.engine import Engine
    g = _flow(reg)
    r = Engine(g).run()
    s = format_template("{status},{ok},{run_id},{out.count},{out.missing},{node[C].value:03d},{var.x}\\n", r, {"x": "A"})
    assert s == "OK,1,1,3,,003,A\n"


def test_tcp_server_trigger_and_reply(reg):
    bus = EventBus()
    sol = Solution("t", bus)
    g = sol.add_flow(_flow(reg))
    runner = FlowRunner(g, sol.variables, bus)
    mgr = CommManager(bus, sol.variables)
    dev = mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    mgr.add_receive_rule(ReceiveRule(name="trig", device="plc", match="startswith", pattern="TRIG", flow="main"))
    mgr.add_send_rule(SendRule(name="res", device="plc", flow="main", template="{status},{out.count}\\n"))
    mgr.set_runners({"main": runner})
    assert not mgr.connect_all()
    runner.start()
    try:
        s = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3)
        rd = _LineReader(s)
        assert _wait(lambda: dev.client_count == 1)
        s.sendall(b"TRIG,1\n")
        assert rd.line() == "OK,3\n"
        # a message that matches no rule does nothing
        s.sendall(b"HELLO\n")
        time.sleep(0.2)
        assert runner.stats.count == 1
        # two frames in one packet
        s.sendall(b"TRIG\nTRIG\n")
        assert rd.line() == "OK,3\n" and rd.line() == "OK,3\n"
        assert runner.stats.count == 3 and dev.stats["rx"] == 4 and dev.stats["tx"] == 3
        s.close()
    finally:
        runner.stop()
        mgr.shutdown()


def test_tcp_client_reconnects_and_receives(reg):
    bus = EventBus()
    got = []
    bus.subscribe("comm_received", lambda **kw: got.append(kw["text"]))
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0)); srv.listen(1); srv.settimeout(5)
    port = srv.getsockname()[1]
    mgr = CommManager(bus)
    dev = mgr.add_device("robot", "tcp_client", {"host": "127.0.0.1", "port": port, "reconnect_s": 0.1})
    mgr.add_receive_rule(ReceiveRule(name="setvar", device="robot", match="regex", pattern=r"^PROD=(\w+)",
                                     action="set_variable", variable="product"))
    try:
        dev.connect()
        conn, _ = srv.accept()
        assert _wait(lambda: dev.connected)
        conn.sendall(b"PROD=A17\r\n")
        assert _wait(lambda: mgr.variables.get("product") == "PROD=A17")
        assert mgr.send("robot", "HI\n")
        assert conn.recv(16) == b"HI\n"
        conn.close()
        assert _wait(lambda: not dev.connected)
        conn2, _ = srv.accept()  # auto reconnect
        assert _wait(lambda: dev.connected)
        conn2.close()
    finally:
        mgr.shutdown()
        srv.close()


def test_udp_loopback(reg):
    bus = EventBus()
    mgr = CommManager(bus)
    a = mgr.add_device("a", "udp", {"host": "127.0.0.1", "port": 0, "local_port": 0})
    b = mgr.add_device("b", "udp", {"host": "127.0.0.1", "port": 0, "local_port": 0})
    got = []
    b.on_receive(lambda d, data: got.append(data))
    try:
        mgr.connect_all()
        a.config["port"] = b._sock.getsockname()[1]
        assert a.send("ping")
        assert _wait(lambda: got == [b"ping"])
    finally:
        mgr.shutdown()


def test_register_conversions():
    assert value_to_registers(-2, "int16") == [0xFFFE] and registers_to_value([0xFFFE], "int16") == -2
    regs = value_to_registers(3.5, "float32")
    assert len(regs) == 2 and registers_to_value(regs, "float32") == 3.5
    regs = value_to_registers(-70000, "int32")
    assert registers_to_value(regs, "int32") == -70000
    assert value_to_registers(True, "bool") == [1]


def test_modbus_server_rising_edge_and_result(reg):
    pymodbus = pytest.importorskip("pymodbus")
    from pymodbus.client import ModbusTcpClient
    bus = EventBus()
    sol = Solution("m", bus)
    g = sol.add_flow(_flow(reg))
    runner = FlowRunner(g, sol.variables, bus)
    mgr = CommManager(bus, sol.variables)
    dev = mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0, "register_count": 64})
    mgr.add_receive_rule(ReceiveRule(name="trig", device="plc", match="register_rising", address=0, flow="main"))
    mgr.add_send_rule(SendRule(name="res", device="plc", flow="main", template="",
                               registers=[{"address": 10, "expr": "{out.count}", "kind": "int16"},
                                          {"address": 11, "expr": "{ok}", "kind": "uint16"},
                                          {"address": 12, "expr": "{duration_ms}", "kind": "float32"},
                                          {"address": 20, "expr": "{ok}", "kind": "bool"}]))
    mgr.set_runners({"main": runner})
    mgr.connect_all()
    runner.start()
    try:
        c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2)
        assert c.connect()
        c.write_register(0, 1, device_id=1)                       # rising edge 0 -> 1 triggers the flow
        assert _wait(lambda: runner.stats.count == 1)
        assert _wait(lambda: dev.read_value(10, "int16") == 3)
        rr = c.read_holding_registers(10, count=4, device_id=1)
        assert not rr.isError() and rr.registers[0] == 3 and rr.registers[1] == 1
        assert registers_to_value(rr.registers[2:4], "float32") >= 0
        assert c.read_coils(20, count=1, device_id=1).bits[0] is True
        c.write_register(0, 1, device_id=1)                       # no edge -> no trigger
        c.write_register(0, 0, device_id=1)
        time.sleep(0.2)
        assert runner.stats.count == 1
        c.write_registers(0, [1, 5, 6], device_id=1)              # FC16 edge
        assert _wait(lambda: runner.stats.count == 2)
        bad = c.read_holding_registers(1000, count=1, device_id=1)
        assert bad.isError()
        c.close()
    finally:
        runner.stop()
        mgr.shutdown()


def test_modbus_client_polls_server(reg):
    pytest.importorskip("pymodbus")
    bus = EventBus()
    mgr = CommManager(bus)
    srv = mgr.add_device("plc_sim", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0})
    srv.connect()
    cli = mgr.add_device("plc", "modbus_tcp_client", {"host": "127.0.0.1", "port": srv.bound_port, "poll_ms": 20,
                                                      "watch_address": 0, "watch_count": 4})
    changes = []
    cli.on_register_change(lambda d, a, o, n: changes.append((a, o, n)))
    try:
        cli.connect()
        assert _wait(lambda: cli.connected)
        srv.write_registers(2, [7])
        assert _wait(lambda: (2, 0, 7) in changes)
        cli.write_value(30, -1.25, "float32")
        assert srv.read_value(30, "float32") == -1.25
        cli.write_value(5, 1, "bool")
        assert srv.coils[5] is True and cli.read_value(5, "bool") is True
    finally:
        mgr.shutdown()


def test_solution_comm_config_roundtrip(reg, tmp_path):
    bus = EventBus()
    mgr = CommManager(bus)
    mgr.add_device("plc", "tcp_server", {"port": 0})
    mgr.add_receive_rule(ReceiveRule(name="t", device="plc"))
    mgr.add_send_rule(SendRule(name="s", device="plc"))
    cfg = mgr.to_dict()
    mgr2 = CommManager(EventBus())
    assert mgr2.load_dict(cfg) == []
    assert "plc" in mgr2.devices and len(mgr2.receive_rules) == 1 and mgr2.send_rules[0].name == "s"

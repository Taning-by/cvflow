"""检测握手、触发策略与通信节点的黑盒测试。

观察点全部在**对端**：真实 TCP 客户端收到的报文、第三方 pymodbus 主站读到的寄存器，
以及流程的运行计数。重点覆盖业务闭环里最容易出错的几件事：

* 先写结果、再写 Done；完成编号与请求编号一致；
* TCP 服务端的回复发给**发起请求的那个**客户端；
* 触发位一直保持为 1 时不重复执行；
* 忙时的拒绝 / 丢弃 / 有界排队；
* 超时只上报不重发；断线重连后旧任务的结果被丢弃。
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from cvflow.comm import (CommManager, FormatField, FormatRule, HandshakeConfig, ParseField,
                         ParseRule, ReceiveRule, SendRule)
from cvflow.comm.handshake import ERROR_CODES
from cvflow.core import EventBus, FlowRunner, Graph, Solution, registry as reg

reg.load_builtins()


def _wait(cond, timeout=5.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Client:
    """真实 TCP 客户端，按行收发。"""

    def __init__(self, port, host="127.0.0.1"):
        self.sock = socket.create_connection((host, port), timeout=5)
        self._buf = b""

    def send(self, text: str):
        self.sock.sendall(text.encode())

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

    def quiet(self, timeout=0.4) -> bool:
        self.sock.settimeout(timeout)
        try:
            return self.sock.recv(1024) == b""
        except socket.timeout:
            return True

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def measure_flow(name="main", delay_ms=0, fail=False, ng=False) -> Graph:
    """测量流程：发布 holes / x / y / angle，可选加延时、判 NG 或直接报错。"""
    g = Graph(name)
    holes = g.add_node(reg.create("source.constant", name="holes", values={"kind": "int", "value": "3"}))
    x = g.add_node(reg.create("source.constant", name="x", values={"kind": "float", "value": "12.345"}))
    y = g.add_node(reg.create("source.constant", name="y", values={"kind": "float", "value": "67.89"}))
    a = g.add_node(reg.create("source.constant", name="angle", values={"kind": "float", "value": "0.125"}))
    pub = g.add_node(reg.create("output.publish", name="pub", values={
        "name_a": "holes", "name_b": "x", "name_c": "y", "name_d": "angle"}))
    src = holes
    if delay_ms:
        d = g.add_node(reg.create("logic.delay", name="delay", values={"ms": delay_ms}))
        g.add_link(holes.id, "value", d.id, "value")
        src = d
    if fail:
        bad = g.add_node(reg.create("logic.expression", name="boom", values={"expression": "nonexistent_name + 1"}))
        g.add_link(holes.id, "value", bad.id, "a")
    if ng:
        j = g.add_node(reg.create("logic.judge", name="judge", values={"op": "==", "low": 99}))
        g.add_link(holes.id, "value", j.id, "value")
    g.add_link(src.id, "value", pub.id, "a")
    g.add_link(x.id, "value", pub.id, "b")
    g.add_link(y.id, "value", pub.id, "c")
    g.add_link(a.id, "value", pub.id, "d")
    return g


class System:
    def __init__(self, graph=None):
        self.bus = EventBus()
        self.sol = Solution("hs", self.bus)
        self.flow = self.sol.add_flow(graph if graph is not None else measure_flow())
        self.runner = FlowRunner(self.flow, self.sol.variables, self.bus)
        self.mgr = CommManager(self.bus, self.sol.variables)
        self.mgr.set_runners({self.flow.name: self.runner})

    def start(self):
        errors = self.mgr.connect_all()
        self.runner.start()
        return errors

    def stop(self):
        self.runner.stop()
        self.mgr.shutdown()


@pytest.fixture
def sys_text():
    """示例 A 的配置：TCP 服务端 + 文本握手。"""
    s = System()
    dev = s.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0, "terminator": "\\n"})
    s.mgr.add_parse_rule(ParseRule(name="req", kind="delimited", separator=",", to_variables=False, fields=[
        ParseField(name="cmd", index=0),
        ParseField(name="request_id", index=1, dtype="int")]))
    s.mgr.add_format_rule(FormatRule(name="result", kind="text", separator=",", terminator="\\n", fields=[
        FormatField(name="h", source="literal:RESULT"),
        FormatField(name="id", source="request_id", dtype="int"),
        FormatField(name="st", source="status"),
        FormatField(name="x", source="out.x", dtype="float", decimals=3),
        FormatField(name="y", source="out.y", dtype="float", decimals=3),
        FormatField(name="a", source="out.angle", dtype="float", decimals=3)]))
    s.mgr.add_format_rule(FormatRule(name="busy", kind="text", separator=",", terminator="\\n", fields=[
        FormatField(name="h", source="literal:BUSY"),
        FormatField(name="id", source="request_id", dtype="int")]))
    s.mgr.add_format_rule(FormatRule(name="error", kind="text", separator=",", terminator="\\n", fields=[
        FormatField(name="h", source="literal:ERROR"),
        FormatField(name="id", source="request_id", dtype="int"),
        FormatField(name="code", source="error_code", dtype="int")]))
    s.mgr.add_handshake(HandshakeConfig(
        id="hs", name="hs", device=dev.id, flow="main", mode="text",
        result_format="result", busy_format="busy", error_format="error",
        busy_policy="reject", dedup_window_s=2.0, run_timeout_s=5.0))
    s.mgr.add_receive_rule(ReceiveRule(name="trig", device=dev.id, match="startswith",
                                       pattern="TRIGGER", flow="main", parse="req",
                                       busy_policy="reject"))
    yield s, dev
    s.stop()


# ============================================================ 示例 A：TCP 文本握手
def test_text_handshake_replies_to_origin_client(sys_text):
    s, dev = sys_text
    s.start()
    a, b = Client(dev.bound_port), Client(dev.bound_port)
    time.sleep(0.2)
    a.send("TRIGGER,1001\n")
    assert a.line() == "RESULT,1001,OK,12.345,67.890,0.125\n"
    assert b.quiet()                                  # 结果只回给发起请求的客户端
    b.send("TRIGGER,1002\n")
    assert b.line() == "RESULT,1002,OK,12.345,67.890,0.125\n"
    assert a.quiet()
    a.close(); b.close()


def test_text_handshake_bad_params_and_flow_error(sys_text):
    s, dev = sys_text
    s.start()
    c = Client(dev.bound_port)
    time.sleep(0.2)
    c.send("TRIGGER,abc\n")                           # 请求编号不是数字 → 参数错误
    assert c.line() == f"ERROR,0,{ERROR_CODES['bad_params']}\n"
    assert s.runner.stats.count == 0                  # 没有触发流程
    c.send("TRIGGER,7\n")
    assert c.line().startswith("RESULT,7,OK")
    c.close()


def test_text_handshake_reports_flow_exception(sys_text):
    s, dev = sys_text
    s.flow.add_node(reg.create("logic.expression", name="boom", values={"expression": "nope + 1"}))
    s.start()
    c = Client(dev.bound_port)
    time.sleep(0.2)
    c.send("TRIGGER,5\n")
    line = c.line()
    assert line.startswith("RESULT,5,ERROR")          # 状态是 ERROR
    hs = s.mgr.handshake_by_name("hs")
    assert hs.last_error_code == ERROR_CODES["flow_error"]
    c.close()


def test_text_handshake_busy_is_rejected_not_queued():
    s = System(measure_flow(delay_ms=400))
    dev = s.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    s.mgr.add_parse_rule(ParseRule(name="req", kind="delimited", to_variables=False, fields=[
        ParseField(name="cmd", index=0), ParseField(name="request_id", index=1, dtype="int")]))
    s.mgr.add_format_rule(FormatRule(name="result", kind="text", separator=",", terminator="\\n", fields=[
        FormatField(name="h", source="literal:RESULT"), FormatField(name="id", source="request_id", dtype="int")]))
    s.mgr.add_format_rule(FormatRule(name="busy", kind="text", separator=",", terminator="\\n", fields=[
        FormatField(name="h", source="literal:BUSY"), FormatField(name="id", source="request_id", dtype="int")]))
    s.mgr.add_handshake(HandshakeConfig(id="hs", name="hs", device=dev.id, flow="main", mode="text",
                                        result_format="result", busy_format="busy",
                                        busy_policy="reject", dedup_window_s=0))
    s.mgr.add_receive_rule(ReceiveRule(name="t", device=dev.id, match="startswith", pattern="TRIGGER",
                                       flow="main", parse="req", busy_policy="reject"))
    s.start()
    try:
        c = Client(dev.bound_port)
        time.sleep(0.2)
        c.send("TRIGGER,1\n")
        time.sleep(0.1)
        c.send("TRIGGER,2\n")                          # 第一条还在跑
        assert c.line() == "BUSY,2\n"                  # 忙时明确回复，而不是静默排队
        assert c.line() == "RESULT,1\n"
        assert s.runner.stats.count == 1
        c.send("TRIGGER,3\n")
        assert c.line() == "RESULT,3\n"
        hs = s.mgr.handshake_by_name("hs")
        assert hs.counters["rejected_busy"] == 1 and hs.counters["completed"] == 2
        c.close()
    finally:
        s.stop()


def test_text_handshake_duplicate_request_resends_last_result(sys_text):
    s, dev = sys_text
    s.start()
    c = Client(dev.bound_port)
    time.sleep(0.2)
    c.send("TRIGGER,42\n")
    first = c.line()
    c.send("TRIGGER,42\n")                            # 同一个请求编号在去重窗内重复发来
    assert c.line() == first                          # 重发上次结果，**不重跑流程**
    assert s.runner.stats.count == 1
    hs = s.mgr.handshake_by_name("hs")
    assert hs.counters["duplicate"] == 1
    c.send("TRIGGER,43\n")
    assert c.line().startswith("RESULT,43") and s.runner.stats.count == 2
    c.close()


def test_text_handshake_not_running_flow_is_reported(sys_text):
    s, dev = sys_text
    s.mgr.connect_all()                               # 只连设备，不启动流程
    c = Client(dev.bound_port)
    time.sleep(0.2)
    c.send("TRIGGER,1\n")
    assert c.line() == f"ERROR,1,{ERROR_CODES['not_running']}\n"
    c.close()


def test_text_handshake_timeout_reports_once_without_retrigger():
    s = System(measure_flow(delay_ms=1200))
    dev = s.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    s.mgr.add_parse_rule(ParseRule(name="req", kind="delimited", to_variables=False, fields=[
        ParseField(name="cmd", index=0), ParseField(name="request_id", index=1, dtype="int")]))
    s.mgr.add_format_rule(FormatRule(name="error", kind="text", separator=",", terminator="\\n", fields=[
        FormatField(name="h", source="literal:ERROR"), FormatField(name="id", source="request_id", dtype="int"),
        FormatField(name="code", source="error_code", dtype="int")]))
    s.mgr.add_handshake(HandshakeConfig(id="hs", name="hs", device=dev.id, flow="main", mode="text",
                                        error_format="error", run_timeout_s=0.3, dedup_window_s=0))
    s.mgr.add_receive_rule(ReceiveRule(name="t", device=dev.id, match="startswith", pattern="TRIGGER",
                                       flow="main", parse="req"))
    s.start()
    try:
        c = Client(dev.bound_port)
        time.sleep(0.2)
        c.send("TRIGGER,9\n")
        assert c.line(timeout=3) == f"ERROR,9,{ERROR_CODES['timeout']}\n"
        hs = s.mgr.handshake_by_name("hs")
        assert hs.counters["timeout"] == 1
        assert s.runner.stats.count <= 1              # 超时不会重发触发
        time.sleep(1.4)
        assert s.runner.stats.count == 1
        c.close()
    finally:
        s.stop()


def test_text_handshake_discards_result_of_disconnected_request():
    """接受请求后对端断开：流程结果不能发给新连上来的客户端。"""
    s = System(measure_flow(delay_ms=500))
    dev = s.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    s.mgr.add_parse_rule(ParseRule(name="req", kind="delimited", to_variables=False, fields=[
        ParseField(name="cmd", index=0), ParseField(name="request_id", index=1, dtype="int")]))
    s.mgr.add_format_rule(FormatRule(name="result", kind="text", terminator="\\n", fields=[
        FormatField(name="h", source="literal:RESULT")]))
    s.mgr.add_handshake(HandshakeConfig(id="hs", name="hs", device=dev.id, flow="main", mode="text",
                                        result_format="result", dedup_window_s=0))
    s.mgr.add_receive_rule(ReceiveRule(name="t", device=dev.id, match="startswith", pattern="TRIGGER",
                                       flow="main", parse="req"))
    s.start()
    try:
        a = Client(dev.bound_port)
        time.sleep(0.2)
        a.send("TRIGGER,1\n")
        time.sleep(0.1)
        a.close()                                      # 请求者走了
        b = Client(dev.bound_port)                     # 换一个客户端连上来
        time.sleep(0.6)
        assert b.quiet(0.5)                            # 绝不能把上一次的结果发给它
        hs = s.mgr.handshake_by_name("hs")
        assert hs.counters["send_failed"] >= 1 or hs.counters["completed"] == 1
        b.close()
    finally:
        s.stop()


# ============================================================ 示例 C：Modbus 从站握手
@pytest.fixture
def sys_slave():
    """Modbus TCP 从站 + 寄存器握手：外部主站写触发，读结果，再写确认。"""
    s = System()
    dev = s.mgr.add_device("plc", "modbus_tcp_server", {
        "host": "127.0.0.1", "port": 0, "register_count": 64, "coil_count": 16}, points=[
        {"name": "trigger", "area": "holding", "address": 0, "dtype": "int16", "direction": "read_write"},
        {"name": "request_id", "area": "holding", "address": 1, "dtype": "int32", "layout": "CDAB",
         "direction": "read_write"},
        {"name": "ready", "area": "holding", "address": 3, "dtype": "int16", "direction": "write"},
        {"name": "busy", "area": "holding", "address": 4, "dtype": "int16", "direction": "write"},
        {"name": "done", "area": "holding", "address": 5, "dtype": "int16", "direction": "write"},
        {"name": "result", "area": "holding", "address": 6, "dtype": "int16", "direction": "write"},
        {"name": "error_code", "area": "holding", "address": 7, "dtype": "int16", "direction": "write"},
        {"name": "done_id", "area": "holding", "address": 8, "dtype": "int32", "layout": "CDAB",
         "direction": "write"},
        {"name": "ack", "area": "holding", "address": 10, "dtype": "int16", "direction": "read_write"},
        {"name": "x", "area": "holding", "address": 20, "dtype": "float32", "layout": "CDAB",
         "direction": "write"},
        {"name": "y", "area": "holding", "address": 22, "dtype": "float32", "layout": "CDAB",
         "direction": "write"},
        {"name": "holes", "area": "holding", "address": 24, "dtype": "int16", "direction": "write"},
    ])
    s.mgr.add_handshake(HandshakeConfig(
        id="hs", name="hs", device=dev.id, flow="main", mode="register",
        ready="ready", busy="busy", done="done", result="result", error_code="error_code",
        request_id="request_id", done_id="done_id", ack="ack",
        require_ack=True, run_timeout_s=5.0, dedup_window_s=0))
    s.mgr.add_receive_rule(ReceiveRule(name="trig", device=dev.id, source="datapoint",
                                       match="rising", point="trigger", flow="main",
                                       request_field="request_id"))
    s.mgr.add_send_rule(SendRule(name="res", device=dev.id, flow="main", template="", points=[
        {"point": "x", "source": "out.x"}, {"point": "y", "source": "out.y"},
        {"point": "holes", "source": "out.holes"}]))
    yield s, dev
    s.stop()


def test_register_handshake_full_cycle_with_ack(sys_slave):
    from pymodbus.client import ModbusTcpClient
    s, dev = sys_slave
    s.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2)
    assert c.connect()
    assert _wait(lambda: dev.read_point("ready") == 1)        # Ready 反映真实可接受能力
    c.write_registers(1, [1234, 0], device_id=1)              # request_id（CDAB：低字在前）
    c.write_register(0, 1, device_id=1)                       # 触发
    assert _wait(lambda: dev.read_point("done") == 1, 3.0)
    regs = c.read_holding_registers(0, count=30, device_id=1).registers
    assert regs[3] == 0 and regs[4] == 0                      # ready=0（等确认）、busy=0
    assert regs[5] == 1 and regs[6] == 1                      # done=1、result=OK
    assert regs[7] == ERROR_CODES["none"]
    assert struct.unpack(">i", struct.pack(">HH", regs[9], regs[8]))[0] == 1234   # done_id 与请求编号一致
    assert abs(struct.unpack(">f", struct.pack(">HH", regs[21], regs[20]))[0] - 12.345) < 1e-3
    assert abs(struct.unpack(">f", struct.pack(">HH", regs[23], regs[22]))[0] - 67.89) < 1e-3
    assert regs[24] == 3
    c.write_register(10, 1, device_id=1)                      # 对端确认
    assert _wait(lambda: dev.read_point("done") == 0 and dev.read_point("ready") == 1, 3.0)
    assert dev.read_point("done_id") == 0 and dev.read_point("ack") == 0
    hs = s.mgr.handshake_by_name("hs")
    assert hs.counters["acked"] == 1 and hs.last_acked
    c.close()


def test_register_handshake_writes_results_before_done():
    """先结果、后 Done：对端只要看到 Done 翻转，读到的一定是本次数据。"""
    from pymodbus.client import ModbusTcpClient
    s = System(measure_flow(delay_ms=150))
    dev = s.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0}, points=[
        {"name": "trigger", "area": "holding", "address": 0, "dtype": "int16", "direction": "read_write"},
        {"name": "done", "area": "holding", "address": 5, "dtype": "int16", "direction": "write"},
        {"name": "holes", "area": "holding", "address": 24, "dtype": "int16", "direction": "write"}])
    s.mgr.add_handshake(HandshakeConfig(id="hs", name="hs", device=dev.id, flow="main",
                                        mode="register", done="done", dedup_window_s=0))
    s.mgr.add_receive_rule(ReceiveRule(name="t", device=dev.id, source="datapoint", match="rising",
                                       point="trigger", flow="main"))
    s.mgr.add_send_rule(SendRule(name="r", device=dev.id, flow="main", template="",
                                 points=[{"point": "holes", "source": "out.holes"}]))
    s.start()
    try:
        c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2)
        assert c.connect()
        seen = []
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                r = c.read_holding_registers(0, count=30, device_id=1)
                if not r.isError():
                    seen.append((r.registers[5], r.registers[24]))
                time.sleep(0.01)

        t = threading.Thread(target=poll, daemon=True)
        t.start()
        c2 = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2)
        assert c2.connect()
        c2.write_register(0, 1, device_id=1)
        assert _wait(lambda: any(d == 1 for d, _ in seen), 3.0)
        stop.set(); t.join(2)
        # 任何一次观测里，done=1 的同时结果一定已经写好了
        assert all(holes == 3 for done, holes in seen if done == 1)
        c.close(); c2.close()
    finally:
        s.stop()


def test_register_trigger_held_high_does_not_refire(sys_slave):
    from pymodbus.client import ModbusTcpClient
    s, dev = sys_slave
    s.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2)
    assert c.connect()
    c.write_register(0, 1, device_id=1)
    assert _wait(lambda: s.runner.stats.count == 1)
    for _ in range(5):                                        # 触发位一直是 1
        c.write_register(0, 1, device_id=1)
        time.sleep(0.05)
    assert s.runner.stats.count == 1                          # 不重复执行
    c.write_register(10, 1, device_id=1)                      # 确认，复位
    assert _wait(lambda: dev.read_point("ready") == 1)
    c.write_register(0, 0, device_id=1)                       # 触发位回到 0 → 重新武装
    c.write_register(0, 1, device_id=1)
    assert _wait(lambda: s.runner.stats.count == 2)
    c.close()


def test_register_handshake_rejects_while_busy(sys_slave):
    from pymodbus.client import ModbusTcpClient
    s, dev = sys_slave
    s.runner.stop()
    s.flow = s.sol.flows["main"]
    s.start()
    c = ModbusTcpClient("127.0.0.1", port=dev.bound_port, timeout=2)
    assert c.connect()
    c.write_register(0, 1, device_id=1)
    assert _wait(lambda: dev.read_point("done") == 1, 3.0)
    # 还没确认（require_ack）→ Ready 为 0，再来一次触发应被拒绝并写忙错误码
    assert dev.read_point("ready") == 0
    c.write_register(0, 0, device_id=1)
    c.write_register(0, 1, device_id=1)
    assert _wait(lambda: dev.read_point("error_code") == ERROR_CODES["busy"], 2.0)
    assert s.runner.stats.count == 1
    c.close()


# ============================================================ 示例 B：Modbus 主站握手
def test_master_handshake_against_third_party_slave(pymodbus_slave):
    """cvflow 做主站轮询第三方从站的触发点，跑完把状态与结果写回去。"""
    from pymodbus.client import ModbusTcpClient
    port = pymodbus_slave
    s = System()
    dev = s.mgr.add_device("plc", "modbus_tcp_client", {
        "host": "127.0.0.1", "port": port, "unit_id": 1, "poll_ms": 0, "watch_count": 0}, points=[
        {"name": "trigger", "area": "holding", "address": 0, "dtype": "int16",
         "direction": "read_write", "poll_ms": 20},
        {"name": "request_id", "area": "holding", "address": 1, "dtype": "int32", "layout": "CDAB",
         "direction": "read_write", "poll_ms": 20},
        {"name": "busy", "area": "holding", "address": 4, "dtype": "int16", "direction": "write"},
        {"name": "done", "area": "holding", "address": 5, "dtype": "int16", "direction": "write"},
        {"name": "result", "area": "holding", "address": 6, "dtype": "int16", "direction": "write"},
        {"name": "done_id", "area": "holding", "address": 8, "dtype": "int32", "layout": "CDAB",
         "direction": "write"},
        {"name": "x", "area": "holding", "address": 20, "dtype": "float32", "layout": "CDAB",
         "direction": "write"},
        {"name": "holes", "area": "holding", "address": 24, "dtype": "int16", "direction": "write"},
    ])
    s.mgr.add_handshake(HandshakeConfig(id="hs", name="hs", device=dev.id, flow="main",
                                        mode="register", busy="busy", done="done", result="result",
                                        request_id="request_id", done_id="done_id",
                                        dedup_window_s=0, run_timeout_s=5.0))
    s.mgr.add_receive_rule(ReceiveRule(name="t", device=dev.id, source="datapoint", match="rising",
                                       point="trigger", flow="main", request_field="request_id"))
    s.mgr.add_send_rule(SendRule(name="r", device=dev.id, flow="main", template="", points=[
        {"point": "x", "source": "out.x"}, {"point": "holes", "source": "out.holes"}]))
    s.start()
    try:
        assert _wait(lambda: dev.connected, 3.0)
        plc = ModbusTcpClient("127.0.0.1", port=port, timeout=2)
        assert plc.connect()
        plc.write_registers(1, [1234, 0], device_id=1)     # request_id = 1234（CDAB：低字在前）
        plc.write_register(0, 1, device_id=1)              # 触发
        assert _wait(lambda: s.runner.stats.count == 1, 5.0)
        assert _wait(lambda: plc.read_holding_registers(5, count=1, device_id=1).registers[0] == 1, 3.0)
        regs = plc.read_holding_registers(0, count=30, device_id=1).registers
        assert regs[4] == 0 and regs[5] == 1 and regs[6] == 1 and regs[24] == 3
        assert struct.unpack(">i", struct.pack(">HH", regs[9], regs[8]))[0] == 1234
        assert abs(struct.unpack(">f", struct.pack(">HH", regs[21], regs[20]))[0] - 12.345) < 1e-3
        plc.close()
    finally:
        s.stop()


@pytest.fixture
def pymodbus_slave():
    """第三方 Modbus 从站（pymodbus），返回监听端口。"""
    pytest.importorskip("pymodbus")
    import asyncio
    from pymodbus.datastore import ModbusDeviceContext, ModbusSequentialDataBlock, ModbusServerContext
    from pymodbus.server import ModbusTcpServer
    port = free_port()
    holder = {}
    ready = threading.Event()
    loop = asyncio.new_event_loop()
    store = ModbusDeviceContext(hr=ModbusSequentialDataBlock(1, [0] * 200),
                                co=ModbusSequentialDataBlock(1, [False] * 64))

    async def main():
        srv = ModbusTcpServer(ModbusServerContext(devices=store, single=True),
                              address=("127.0.0.1", port))
        holder["srv"] = srv
        await srv.serve_forever(background=True)
        ready.set()
        while not holder.get("stop"):
            await asyncio.sleep(0.05)
        await srv.shutdown()

    def runner():
        """跑完 main() 之后**把事件循环关干净**。

        不关的话，pymodbus 服务端那些半启动的请求处理协程会一直挂着，之后被垃圾回收时抛
        ``coroutine 'ServerRequestHandler.handle_request' was never awaited``——而且这个警告会
        记在**当时恰好在执行的那一行**上，看着像是别处的代码有问题。Windows 用 Proactor
        事件循环，关闭路径和 Linux 不同，这个现象在那边尤其明显。
        """
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        finally:
            for task in asyncio.all_tasks(loop):
                task.cancel()
            leftover = asyncio.all_tasks(loop)
            if leftover:
                loop.run_until_complete(asyncio.gather(*leftover, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    if not ready.wait(5):
        pytest.skip("pymodbus server did not start")
    time.sleep(0.2)
    yield port
    holder["stop"] = True
    t.join(5)

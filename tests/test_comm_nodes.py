"""通信节点与统一设备管理的黑盒测试。

节点部分：节点只通过全局通信管理器拿设备，所以这里把真实设备连起来，从对端观察节点的效果。
管理部分：稳定 id、改名不破坏引用、复制、启用禁用、引用位置、配置保存恢复、退出后端口释放。
"""
from __future__ import annotations

import json
import socket
import time

import pytest

from cvflow.comm import (CommManager, FormatField, FormatRule, ParseField, ParseRule, ReceiveRule,
                         SendRule, set_manager)
from cvflow.core import EventBus, FlowRunner, Graph, Solution, Trigger, TriggerSource
from cvflow.core import registry as reg

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


class Harness:
    """一个方案 + 一条流程 + 一个通信管理器，并把管理器设为全局（节点要用）。"""

    def __init__(self, graph: Graph | None = None):
        self.bus = EventBus()
        self.sol = Solution("nodes", self.bus)
        self.flow = self.sol.add_flow(graph if graph is not None else Graph("main"))
        self.runner = FlowRunner(self.flow, self.sol.variables, self.bus)
        self.mgr = CommManager(self.bus, self.sol.variables)
        self.mgr.set_runners({"main": self.runner})
        set_manager(self.mgr)

    def run(self, payload: dict | None = None):
        trig = Trigger(TriggerSource.COMM, payload=payload or {})
        return self.runner.run_once(trig)

    def stop(self):
        self.runner.stop()
        self.mgr.shutdown()
        set_manager(None)


@pytest.fixture
def h():
    harness = Harness()
    yield harness
    harness.stop()


def node(h, type_id, name, **values):
    return h.flow.add_node(reg.create(type_id, name=name, values=values))


# ============================================================ 节点
def test_node_receive_takes_from_inbox_not_the_socket(h):
    """后台监听线程唯一地读套接字，节点从自己的收件箱取——两边不抢读。"""
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    rx = node(h, "comm.receive", "rx", device=dev.id, timeout_ms=2000)
    node(h, "output.publish", "pub", name_a="msg")
    h.flow.add_link(rx.id, "text", h.flow.nodes[[n.id for n in h.flow.nodes.values() if n.name == "pub"][0]].id, "a")
    h.mgr.connect_all()
    h.runner.start()
    c = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3)
    time.sleep(0.2)
    c.sendall(b"HELLO\n")
    assert _wait(lambda: len(h.mgr.inbox(f"node:{rx.id}", dev.id)) == 1)
    r = h.runner.trigger(Trigger(TriggerSource.MANUAL), wait=True, timeout=5)
    assert r is not None and r.outputs["msg"] == "HELLO"
    assert dev.stats["rx"] == 1          # 报文只被读了一次
    c.close()


def test_node_receive_empty_modes(h):
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    rx = node(h, "comm.receive", "rx", device=dev.id, timeout_ms=0, on_empty="skip")
    h.mgr.connect_all()
    r = h.run()
    assert r.node_results[rx.id].status.value == "skipped"
    rx.set("on_empty", "empty")
    r = h.run()
    assert r.node_results[rx.id].outputs["got"] is False
    rx.set("on_empty", "error")
    r = h.run()
    assert r.node_results[rx.id].status.value == "error" and "收件箱" in r.node_results[rx.id].error


def test_node_send_replies_to_origin_peer(h):
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    node(h, "comm.send", "tx", device=dev.id, template="ACK,{request_id}", append="\\n", target="origin")
    h.mgr.connect_all()
    a = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3)
    b = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3)
    time.sleep(0.25)
    peer_a = f"{a.getsockname()[0]}:{a.getsockname()[1]}"
    r = h.run({"request_id": 77, "peer": peer_a})
    assert r.ok
    a.settimeout(2); assert a.recv(100) == b"ACK,77\n"
    b.settimeout(0.4)
    with pytest.raises(socket.timeout):
        b.recv(100)                      # 另一个客户端收不到
    a.close(); b.close()


def test_node_send_hex_mode_and_dead_peer(h):
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    tx = node(h, "comm.send", "tx", device=dev.id, mode="hex", template="02 41 42 03",
              append="", target="broadcast")
    h.mgr.connect_all()
    c = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3)
    time.sleep(0.2)
    assert h.run().ok
    c.settimeout(2); assert c.recv(100) == b"\x02AB\x03"
    c.close()
    tx.set("target", "origin")
    r = h.run({"peer": "1.2.3.4:5"})
    assert r.node_results[tx.id].status.value == "error" and "已断开" in r.node_results[tx.id].error


def test_node_parse_uses_frozen_trigger_fields(h):
    h.mgr.add_parse_rule(ParseRule(name="req", kind="delimited", to_variables=False, fields=[
        ParseField(name="cmd", index=0), ParseField(name="request_id", index=1, dtype="int")]))
    p = node(h, "comm.parse", "p", rule="req", field="request_id", source="trigger")
    r = h.run({"fields": {"cmd": "TRIGGER", "request_id": 1001}})
    assert r.node_results[p.id].outputs["value"] == 1001
    assert r.node_results[p.id].outputs["fields"]["cmd"] == "TRIGGER"
    # 没有冻结字段时解析原始报文
    r = h.run({"raw": b"TRIGGER,55"})
    assert r.node_results[p.id].outputs["value"] == 55
    # 非通信触发
    r = h.run({})
    assert r.node_results[p.id].status.value == "error"


def test_node_parse_error_modes(h):
    h.mgr.add_parse_rule(ParseRule(name="req", kind="delimited", fields=[
        ParseField(name="request_id", index=1, dtype="int")]))
    p = node(h, "comm.parse", "p", rule="req", source="trigger", on_error="error")
    r = h.run({"raw": b"ONLYONE"})
    assert r.node_results[p.id].status.value == "error" and "解析失败" in r.node_results[p.id].error
    p.set("on_error", "empty")
    r = h.run({"raw": b"ONLYONE"})
    out = r.node_results[p.id].outputs
    assert out["ok"] is False and out["error"]
    p.set("rule", "missing")
    r = h.run({"raw": b"x,1"})
    assert "找不到解析规则" in r.node_results[p.id].error


def test_node_format_renders_current_run(h):
    h.mgr.add_format_rule(FormatRule(name="res", kind="text", separator=",", terminator="\\n", fields=[
        FormatField(name="id", source="request_id", dtype="int"),
        FormatField(name="st", source="status"),
        FormatField(name="n", source="out.count", dtype="int")]))
    c = node(h, "source.constant", "c", kind="int", value="7")
    pub = node(h, "output.publish", "pub", name_a="count")
    f = node(h, "comm.format", "f", rule="res")
    h.flow.add_link(c.id, "value", pub.id, "a")
    h.flow.add_link(pub.id if False else c.id, "value", f.id, "after")
    r = h.run({"request_id": 9})
    assert r.node_results[f.id].outputs["text"] == "9,OK,7\n"


def test_node_data_point_read_write(h):
    dev = h.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0}, points=[
        {"name": "x", "area": "holding", "address": 10, "dtype": "float32", "layout": "CDAB",
         "direction": "read_write"},
        {"name": "flag", "area": "coil", "address": 1, "dtype": "bool", "direction": "read_write"}])
    h.mgr.connect_all()
    c = node(h, "source.constant", "c", kind="float", value="12.5")
    w = node(h, "comm.write_point", "w", device=dev.id, point="x")
    rd = node(h, "comm.read_point", "r", device=dev.id, point="x", mode="fresh")
    h.flow.add_link(c.id, "value", w.id, "value")
    h.flow.add_link(w.id, "ok", rd.id, "after")
    r = h.run()
    assert r.node_results[w.id].outputs["ok"] is True
    assert abs(r.node_results[rd.id].outputs["value"] - 12.5) < 1e-6
    assert abs(dev.read_point("x") - 12.5) < 1e-6
    # 写一个不存在的数据点：报错信息要带上有哪些可用
    w.set("point", "nope")
    r = h.run()
    assert "没有数据点" in r.node_results[w.id].error
    w.set("on_error", "false")
    r = h.run()
    assert r.node_results[w.id].outputs["ok"] is False


def test_node_read_point_marks_stale_after_disconnect(h):
    dev = h.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0}, points=[
        {"name": "v", "area": "holding", "address": 0, "dtype": "int16", "direction": "read_write"}])
    h.mgr.connect_all()
    dev.write_point("v", 5)
    rd = node(h, "comm.read_point", "r", device=dev.id, point="v", mode="cached")
    assert h.run().node_results[rd.id].outputs["value"] == 5
    dev.disconnect()
    r = h.run()
    assert r.node_results[rd.id].status.value == "error" and "过期" in r.node_results[rd.id].error
    rd.set("on_error", "invalid")
    r = h.run()
    assert r.node_results[rd.id].outputs["valid"] is False


def test_node_trigger_data_exposes_frozen_request(h):
    """已有的 source.trigger 节点被扩展成也输出请求编号、冻结参数与来源对端。"""
    t = node(h, "source.trigger", "t", field="model", required=True)
    r = h.run({"request_id": 1001, "peer": "1.2.3.4:5", "device": "plc",
               "fields": {"model": "A1", "request_id": 1001}})
    out = r.node_results[t.id].outputs
    assert out["request_id"] == 1001 and out["value"] == "A1" and out["peer"] == "1.2.3.4:5"
    assert out["source"] == "comm"
    r = h.run({"fields": {}})
    assert "没有字段" in r.node_results[t.id].error


def test_node_comm_status(h):
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    st = node(h, "comm.status", "st", device=dev.id, require_connected=True)
    r = h.run()
    assert r.node_results[st.id].status.value == "error"      # 还没连接
    h.mgr.connect_all()
    r = h.run()
    out = r.node_results[st.id].outputs
    assert out["connected"] is True and json.loads(out["text"])["kind"] == "tcp_server"


# ============================================================ 设备管理
def test_device_ids_are_stable_across_rename(h):
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    did = dev.id
    h.mgr.add_receive_rule(ReceiveRule(name="t", device=did, match="any", flow="main"))
    h.mgr.add_send_rule(SendRule(name="s", device=did, template="x"))
    h.mgr.rename_device(did, "plc_new")
    assert h.mgr.resolve(did) is dev and dev.name == "plc_new"
    assert h.mgr.resolve("plc_new") is dev and h.mgr.resolve("plc") is None
    assert h.mgr.receive_rules[0].device == did and h.mgr.send_rules[0].device == did


def test_rename_rewrites_legacy_name_references(h):
    """老方案里规则与节点按**名字**引用设备；改名时一并改写成 id。"""
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    h.mgr.add_receive_rule(ReceiveRule(name="t", device="plc", match="any", flow="main"))
    n = node(h, "comm.send", "tx", device="plc", template="x")
    h.mgr.rename_device("plc", "line2")
    assert h.mgr.receive_rules[0].device == dev.id
    assert n.values["device"] == dev.id


def test_device_duplicate_enable_disable_and_remove(h):
    dev = h.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 5020}, points=[
        {"name": "v", "area": "holding", "address": 0, "dtype": "int16"}])
    copy = h.mgr.duplicate_device(dev.id)
    assert copy.id != dev.id and copy.name == "plc_副本" and copy.kind == dev.kind
    assert copy.config["port"] == 0                 # 服务端类设备的端口置 0，避免撞端口
    assert copy.points.names() == ["v"]
    h.mgr.set_device_enabled(copy.id, False)
    assert not copy.enabled
    assert h.mgr.connect_all() == [] and not copy.connected      # 禁用的设备不连
    h.mgr.remove_device(copy.id)
    assert h.mgr.resolve(copy.id) is None and len(h.mgr.devices) == 1


def test_device_auto_connect_can_be_turned_off(h):
    a = h.mgr.add_device("a", "tcp_server", {"host": "127.0.0.1", "port": 0})
    b = h.mgr.add_device("b", "tcp_server", {"host": "127.0.0.1", "port": 0, "auto_connect": False})
    h.mgr.connect_all()
    assert a.connected and not b.connected
    h.mgr.connect(b.id)
    assert _wait(lambda: b.connected)


def test_manual_disconnect_stops_auto_reconnect(h):
    port = free_port()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port)); srv.listen(2); srv.settimeout(0.5)
    dev = h.mgr.add_device("cli", "tcp_client", {"host": "127.0.0.1", "port": port,
                                                 "reconnect_s": 0.1})
    try:
        h.mgr.connect_all()
        assert _wait(lambda: dev.connected, 3.0)
        h.mgr.disconnect(dev.id)                     # 用户手动断开
        assert dev.manual_stop and not dev.connected
        time.sleep(0.5)
        assert not dev.connected                     # 不再自动重连
        h.mgr.connect(dev.id)                        # 手动连接会清掉这个标记
        assert _wait(lambda: dev.connected, 3.0) and not dev.manual_stop
    finally:
        srv.close()


def test_references_point_at_concrete_places(h):
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    h.mgr.add_receive_rule(ReceiveRule(name="触发", device=dev.id, match="any", flow="main"))
    h.mgr.add_send_rule(SendRule(name="回结果", device=dev.id, template="x"))
    from cvflow.comm import HandshakeConfig
    h.mgr.add_handshake(HandshakeConfig(name="握手", device=dev.id, flow="main", mode="text"))
    node(h, "comm.send", "发送节点", device=dev.id)
    refs = h.mgr.references(dev.id)
    assert "接收规则「触发」" in refs and "发送规则「回结果」" in refs
    assert "检测握手「握手」" in refs and "流程「main」的节点「发送节点」" in refs
    assert h.mgr.references("nope") == []


def test_validate_reports_dangling_references(h):
    h.mgr.add_receive_rule(ReceiveRule(name="t", device="ghost", match="any", flow="nowhere"))
    h.mgr.add_send_rule(SendRule(name="s", device="ghost", format="missing"))
    problems = h.mgr.validate()
    assert any("不存在的设备" in p for p in problems)
    assert any("不存在的流程" in p for p in problems)
    assert any("格式化规则" in p for p in problems)


def test_persistence_roundtrip_with_points_rules_and_handshake(tmp_path):
    from cvflow.comm import HandshakeConfig
    bus = EventBus()
    sol = Solution("p", bus)
    sol.add_flow(Graph("main"))
    mgr = CommManager(bus, sol.variables)
    dev = mgr.add_device("plc", "modbus_tcp_server", {"host": "0.0.0.0", "port": 5020}, points=[
        {"name": "trigger", "area": "holding", "address": 0, "dtype": "int16", "direction": "read_write"},
        {"name": "x", "area": "holding", "address": 20, "dtype": "float32", "layout": "CDAB",
         "direction": "write", "scale": 1.0, "variable": "last_x"}])
    hmi = mgr.add_device("hmi", "tcp_client", {"host": "10.0.0.5", "port": 9000,
                                               "framing": "length_prefix", "length_size": 4})
    mgr.add_parse_rule(ParseRule(name="req", kind="delimited", fields=[ParseField(name="request_id", index=1, dtype="int")]))
    mgr.add_format_rule(FormatRule(name="res", kind="json", fields=[FormatField(name="st", source="status")]))
    mgr.add_receive_rule(ReceiveRule(name="t", device=dev.id, source="datapoint", match="rising",
                                     point="trigger", flow="main", busy_policy="reject"))
    mgr.add_send_rule(SendRule(name="s", device=hmi.id, format="res", target="origin"))
    mgr.add_handshake(HandshakeConfig(name="hs", device=dev.id, flow="main", mode="register",
                                      ready="ready", require_ack=True, ack_value=1))
    sol.comm_config = mgr.to_dict()
    sol.save(tmp_path / "s.json")
    mgr.shutdown()

    sol2 = Solution.load(tmp_path / "s.json")
    mgr2 = CommManager(sol2.bus, sol2.variables)
    assert mgr2.load_dict(sol2.comm_config) == []
    assert set(mgr2.devices) == {"plc", "hmi"}
    assert mgr2.resolve(dev.id) is not None and mgr2.resolve(dev.id).id == dev.id   # id 也存下来了
    p = mgr2.resolve("plc")
    assert p.points.names() == ["trigger", "x"]
    assert p.points.get("x").layout == "CDAB" and p.points.get("x").variable == "last_x"
    assert mgr2.resolve("hmi").framing.kind == "length_prefix"
    assert mgr2.receive_rules[0].source == "datapoint" and mgr2.receive_rules[0].busy_policy == "reject"
    assert mgr2.send_rules[0].format == "res" and mgr2.send_rules[0].target == "origin"
    assert "req" in mgr2.parse_rules and "res" in mgr2.format_rules
    assert mgr2.handshakes[0].config.require_ack and mgr2.handshakes[0].config.ready == "ready"
    assert mgr2.validate() == []
    mgr2.shutdown()


def test_bad_point_config_is_skipped_not_fatal(h):
    dev = h.mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0}, points=[
        {"name": "ok", "area": "holding", "address": 0, "dtype": "int16"},
        {"name": "bad", "area": "input", "address": 0, "dtype": "int16", "direction": "write"}])
    assert dev.points.names() == ["ok"]            # 坏的那条被跳过
    assert any("bad" in e["note"] for e in h.mgr.log.entries() if e["note"])


def test_shutdown_releases_ports_and_threads():
    bus = EventBus()
    sol = Solution("r", bus)
    sol.add_flow(Graph("main"))
    mgr = CommManager(bus, sol.variables)
    port = free_port()
    dev = mgr.add_device("srv", "tcp_server", {"host": "127.0.0.1", "port": port})
    mgr.connect_all()
    assert _wait(lambda: dev.connected)
    mgr.shutdown()
    # 端口真的放掉了：同一个端口能重新绑定
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.close()
    import threading
    assert not any(t.name.startswith(("tcps-srv", "comm-maintenance")) and t.is_alive()
                   for t in threading.enumerate())


def test_comm_log_capacity_pause_and_export(tmp_path, h):
    dev = h.mgr.add_device("plc", "tcp_server", {"host": "127.0.0.1", "port": 0})
    h.mgr.log.capacity = 2000
    h.mgr.connect_all()
    c = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3)
    time.sleep(0.2)
    c.sendall(b"A\nB\n")
    assert _wait(lambda: len([e for e in h.mgr.log.entries() if e["dir"] == "rx"]) == 2)
    seq = h.mgr.log.last_seq
    h.mgr.log.paused = True
    c.sendall(b"C\n")
    time.sleep(0.3)
    assert h.mgr.log.last_seq == seq               # 暂停后不再记录
    h.mgr.log.paused = False
    path = tmp_path / "log.csv"
    n = h.mgr.log.export(str(path))
    text = path.read_text(encoding="utf-8-sig")
    assert n >= 2 and "方向" in text and "A" in text
    assert h.mgr.log.entries(since_seq=seq) == []
    h.mgr.log.clear()
    assert h.mgr.log.entries() == []
    c.close()

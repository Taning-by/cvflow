import json
import time

import pytest

from cvflow.core import (DataType, EventBus, FlowRunner, Graph, GraphError, Node, NodeStatus, Param,
                         Port, Solution, Trigger, TriggerSource, events)
from cvflow.core.engine import Engine


def test_param_coercion():
    p = Param("k", 5, "int", min=1, max=9)
    assert p.coerce("12") == 9 and p.coerce(0.4) == 1
    assert Param("b", False, "bool").coerce("true") is True
    assert Param("e", "a", "enum", choices=["a", "b"]).coerce("zzz") == "a"
    assert Param("r", None, "rect").coerce({"x": 1, "y": 2, "w": 3, "h": 4})["angle"] == 0.0


def test_node_roundtrip(reg):
    n = reg.create("preprocess.threshold", values={"thresh": 77, "method": "otsu"})
    n.position = [10, 20]
    d = json.loads(json.dumps(n.to_dict()))
    m = reg.node_from_dict(d)
    assert m.get("thresh") == 77 and m.get("method") == "otsu" and m.position == [10, 20] and m.id == n.id


def test_graph_links_and_cycles(reg):
    g = Graph("t")
    a = g.add_node(reg.create("source.constant", values={"kind": "float", "value": "3"}))
    b = g.add_node(reg.create("logic.expression", values={"expression": "a * 2"}))
    c = g.add_node(reg.create("logic.expression", values={"expression": "a + 1"}))
    g.add_link(a.id, "value", b.id, "a")
    g.add_link(b.id, "value", c.id, "a")
    with pytest.raises(GraphError):
        g.add_link(c.id, "value", b.id, "b")  # cycle
    with pytest.raises(GraphError):
        g.add_link(a.id, "value", a.id, "a")  # self link
    img = g.add_node(reg.create("preprocess.blur"))
    g.add_link(a.id, "value", img.id, "image")  # ANY -> IMAGE is allowed (checked at run time)
    order = g.topological_order()
    assert order[0] == a.id and order.index(b.id) < order.index(c.id)
    # replacing a link on the same input
    d = g.add_node(reg.create("source.constant", values={"kind": "float", "value": "10"}))
    g.add_link(d.id, "value", b.id, "a")
    assert len([l for l in g.links if l.dst_node == b.id and l.dst_port == "a"]) == 1
    g.remove_node(b.id)
    assert all(b.id not in (l.src_node, l.dst_node) for l in g.links)


def test_type_mismatch(reg):
    g = Graph()
    s = g.add_node(reg.create("logic.format"))  # outputs STRING
    t = g.add_node(reg.create("preprocess.blur"))  # input IMAGE
    with pytest.raises(GraphError):
        g.add_link(s.id, "text", t.id, "image")


def test_engine_pipeline_and_judge(reg):
    g = Graph("judge")
    a = g.add_node(reg.create("source.constant", values={"kind": "float", "value": "7"}))
    e = g.add_node(reg.create("logic.expression", values={"expression": "a * 2"}))
    j = g.add_node(reg.create("logic.judge", values={"op": "in_range", "low": 0, "high": 10}))
    g.add_link(a.id, "value", e.id, "a")
    g.add_link(e.id, "value", j.id, "value")
    r = Engine(g).run()
    assert r.ok and r.judgement is False and r.status_text == "NG"
    assert r.node_results[j.id].status == NodeStatus.NG
    j.set("high", 20)
    r = Engine(g).run()
    assert r.passed and r.status_text == "OK"


def test_engine_error_isolation(reg):
    g = Graph("iso")
    a = g.add_node(reg.create("source.constant", values={"kind": "float", "value": "1"}))
    bad = g.add_node(reg.create("logic.expression", values={"expression": "a / 0"}))
    after_bad = g.add_node(reg.create("logic.expression", values={"expression": "a + 1"}))
    good = g.add_node(reg.create("logic.expression", values={"expression": "a + 1"}))
    g.add_link(a.id, "value", bad.id, "a")
    g.add_link(bad.id, "value", after_bad.id, "a")
    g.add_link(a.id, "value", good.id, "a")
    r = Engine(g).run()
    assert not r.ok and r.status_text == "ERROR"
    assert r.node_results[bad.id].status == NodeStatus.ERROR
    assert r.node_results[after_bad.id].status == NodeStatus.SKIPPED
    assert r.node_results[good.id].status == NodeStatus.OK and r.node_results[good.id].outputs["value"] == 2


def test_gate_skips_downstream(reg):
    g = Graph("gate")
    c = g.add_node(reg.create("source.constant", values={"kind": "bool", "value": "false"}))
    v = g.add_node(reg.create("source.constant", values={"kind": "int", "value": "5"}))
    gate = g.add_node(reg.create("logic.gate"))
    down = g.add_node(reg.create("logic.expression", values={"expression": "a"}))
    g.add_link(c.id, "value", gate.id, "condition")
    g.add_link(v.id, "value", gate.id, "value")
    g.add_link(gate.id, "value", down.id, "a")
    r = Engine(g).run()
    assert r.ok and r.node_results[gate.id].status == NodeStatus.SKIPPED
    assert r.node_results[down.id].status == NodeStatus.SKIPPED
    c.set("value", "true")
    r = Engine(g).run()
    assert r.node_results[down.id].outputs["value"] == 5


def test_unconnected_required_input_is_skipped(reg):
    g = Graph()
    t = g.add_node(reg.create("preprocess.threshold"))
    r = Engine(g).run()
    assert r.ok and r.node_results[t.id].status == NodeStatus.SKIPPED
    assert "未连接" in r.node_results[t.id].error


def test_runner_threaded_and_timer(reg):
    bus = EventBus()
    seen = []
    bus.subscribe(events.RUN_FINISHED, lambda result: seen.append(result.run_id))
    g = Graph("rt")
    a = g.add_node(reg.create("source.constant", values={"kind": "int", "value": "1"}))
    cnt = g.add_node(reg.create("logic.counter"))
    g.add_link(a.id, "value", cnt.id, "count_if")
    runner = FlowRunner(g, bus=bus)
    assert runner.trigger() is None  # not started -> ignored
    runner.start()
    try:
        r = runner.trigger(Trigger(TriggerSource.COMM, message="GO"), wait=True, timeout=5)
        assert r is not None and r.node_results[cnt.id].outputs["count"] == 1
        runner.set_continuous(0.02)
        time.sleep(0.3)
        assert runner.stats.count >= 3
        runner.set_continuous(None)
    finally:
        runner.stop()
    assert not runner.running and len(seen) == runner.stats.count
    assert cnt.state["count"] == runner.stats.count  # state persisted across runs


def test_event_bus_wildcard():
    bus = EventBus()
    got = []
    bus.subscribe("*", lambda event, **kw: got.append((event, kw)))
    bus.subscribe("x", lambda **kw: got.append(("direct", kw)))
    bus.emit("x", a=1)
    assert ("x", {"a": 1}) in got and ("direct", {"a": 1}) in got


def test_solution_roundtrip(tmp_path, reg, plugins_dir):
    sol = Solution("demo")
    g = sol.add_flow(name="main")
    a = g.add_node(reg.create("source.constant", values={"kind": "float", "value": "2"}))
    e = g.add_node(reg.create("logic.expression", values={"expression": "a ** 2"}))
    g.add_link(a.id, "value", e.id, "a")
    sol.variables.define("product", "A1", "string")
    sol.comm_config = {"devices": [{"name": "plc", "kind": "tcp_server", "config": {"port": 0}}],
                       "receive_rules": [], "send_rules": []}
    sol.plugin_dirs = [str(plugins_dir)]
    p = tmp_path / "s.json"
    sol.save(p)
    s2 = Solution.load(p)
    assert s2.name == "demo" and "main" in s2.flows and s2.variables.get("product") == "A1"
    assert s2.comm_config["devices"][0]["name"] == "plc"
    r = Engine(s2.flows["main"]).run()
    assert r.node(e.name).outputs["value"] == 4
    assert reg.has("learning.bandit_threshold")  # loaded from plugin_dirs


def test_plugin_bandit_learns(reg, plugins_dir):
    reg.load_plugin_dir(plugins_dir)
    node = reg.create("learning.bandit_threshold", values={"low": 100, "high": 120, "step": 10, "epsilon": 0.0,
                                                           "learning_rate": 0.5})
    g = Graph()
    g.add_node(node)
    eng = Engine(g)
    # reward only when the agent picks 110; feed the reward from the previous action
    class RewardNode(Node):
        type_id = "test.reward"
        outputs = [Port("r", DataType.FLOAT)]
        def process(self, ctx, inputs):
            last = node.state.get("last_arm")
            return {"r": 1.0 if last == 110 else 0.0}
    rw = g.add_node(RewardNode())
    g.add_link(rw.id, "r", node.id, "reward")
    # force exploration of every arm once by rotating last_arm manually
    for arm in (100, 110, 120):
        node.state["last_arm"] = arm
        eng.run()
    for _ in range(5):
        eng.run()
    assert node.state["q"]["110"] > node.state["q"]["100"]
    assert eng.last_result.node_results[node.id].outputs["best"] == 110

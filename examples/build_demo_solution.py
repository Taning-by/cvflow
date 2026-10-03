"""Build the demo solution JSON (examples/solutions/demo_holes.json).

Flow "main"      : folder -> gray -> threshold -> morphology -> blob count -> judge, plus an edge
                   caliper measuring the part width; results published and rendered/saved on NG.
Flow "learning"  : closed loop where a bandit plugin node tunes the threshold from a reward.
Communication    : TCP server on 6000 - "TRIG" triggers "main", result string is sent back.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvflow.core import Graph, Solution, paths, registry  # noqa: E402

registry.load_builtins()
registry.load_plugin_dir(ROOT / "examples" / "plugins")

OUT = ROOT / "examples" / "solutions" / "demo_holes.json"
paths.set_base_dir(OUT.parent)   # relative paths below are relative to the solution file
IMAGES = "../images"


def place(node, col: int, row: int = 0):
    node.position = [40 + col * 210, 60 + row * 160]
    return node


def build_main() -> Graph:
    g = Graph("main")
    g.description = "统计工件上的三个孔并测量工件宽度。"
    src = place(g.add_node(registry.create("source.image_folder", name="取图",
                                           values={"directory": IMAGES, "mode": "next"})), 0)
    gray = place(g.add_node(registry.create("preprocess.color", name="灰度", values={"mode": "gray"})), 1)
    th = place(g.add_node(registry.create("preprocess.threshold", name="阈值",
                                          values={"method": "binary_inv", "thresh": 120})), 2)
    mo = place(g.add_node(registry.create("preprocess.morphology", name="开运算",
                                          values={"op": "open", "ksize": 3})), 3)
    blob = place(g.add_node(registry.create("analysis.blob", name="孔",
                                            values={"min_area": 500, "max_area": 5000})), 4)
    jc = place(g.add_node(registry.create("logic.judge", name="孔数判定",
                                          values={"op": "==", "low": 3, "name": "holes"})), 5)
    cal = place(g.add_node(registry.create("analysis.caliper", name="宽度",
                                           values={"roi": {"x": 60, "y": 150, "w": 520, "h": 24},
                                                   "direction": "horizontal", "select": "first_last",
                                                   "min_contrast": 15})), 4, 1)
    jw = place(g.add_node(registry.create("logic.judge", name="宽度判定",
                                          values={"op": "in_range", "low": 390, "high": 410, "name": "width"})), 5, 1)
    pub = place(g.add_node(registry.create("output.publish", name="发布",
                                           values={"name_a": "holes", "name_b": "width"})), 6)
    rnd = place(g.add_node(registry.create("output.render", name="渲染")), 6, 1)
    sav = place(g.add_node(registry.create("output.save_image", name="NG 存图",
                                           values={"directory": "../../captures", "when": "ng_only"})), 7, 1)
    g.add_link(src.id, "image", gray.id, "image")
    g.add_link(gray.id, "image", th.id, "image")
    g.add_link(th.id, "image", mo.id, "image")
    g.add_link(mo.id, "image", blob.id, "image")
    g.add_link(blob.id, "count", jc.id, "value")
    g.add_link(gray.id, "image", cal.id, "image")
    g.add_link(cal.id, "width", jw.id, "value")
    g.add_link(blob.id, "count", pub.id, "a")
    g.add_link(cal.id, "width", pub.id, "b")
    g.add_link(src.id, "image", rnd.id, "image")
    g.add_link(jc.id, "ok", rnd.id, "after")
    g.add_link(jw.id, "ok", sav.id, "after")
    g.add_link(rnd.id, "image", sav.id, "image")
    return g


def build_learning() -> Graph:
    g = Graph("learning")
    g.description = "Bandit 插件调节阈值；找到 3 个标称面积的孔时奖励最高。"
    rew = place(g.add_node(registry.create("logic.get_variable", name="奖励", values={"name": "reward", "default": "0"})), 0, 1)
    bandit = place(g.add_node(registry.create("learning.bandit_threshold", name="调优器",
                                              values={"low": 60, "high": 220, "step": 20, "epsilon": 0.2})), 1, 1)
    src = place(g.add_node(registry.create("source.image_folder", name="取图",
                                           values={"directory": IMAGES, "mode": "fixed", "index": 0})), 0)
    gray = place(g.add_node(registry.create("preprocess.color", name="灰度", values={"mode": "gray"})), 1)
    th = place(g.add_node(registry.create("preprocess.threshold", name="阈值", values={"method": "binary_inv"})), 2)
    blob = place(g.add_node(registry.create("analysis.blob", name="孔", values={"min_area": 500, "max_area": 5000})), 3)
    exp = place(g.add_node(registry.create("logic.expression", name="得分", values={
        "expression": "(1.0 if a == 3 else 0.0) * max(0.0, 1.0 - abs(b - 1963) / 1963)"})), 4)
    setv = place(g.add_node(registry.create("logic.set_variable", name="存奖励", values={"name": "reward"})), 5)
    g.add_link(rew.id, "value", bandit.id, "reward")
    g.add_link(bandit.id, "threshold", th.id, "thresh")
    g.add_link(src.id, "image", gray.id, "image")
    g.add_link(gray.id, "image", th.id, "image")
    g.add_link(th.id, "image", blob.id, "image")
    g.add_link(blob.id, "count", exp.id, "a")
    g.add_link(blob.id, "largest_area", exp.id, "b")
    g.add_link(exp.id, "value", setv.id, "value")
    return g


RESULT_REGS = [{"address": 10, "expr": "{out.holes}", "kind": "int16"},
               {"address": 11, "expr": "{ok}", "kind": "uint16"},
               {"address": 12, "expr": "{out.width}", "kind": "float32"}]

COMM_VARIANTS = {
    "demo_holes": {   # TCP 文本协议
        "devices": [{"name": "plc", "kind": "tcp_server", "config": {"host": "0.0.0.0", "port": 6000, "terminator": "\\n",
                                                                     "heartbeat_s": 0, "heartbeat_text": "HB\\n"}}],
        "receive_rules": [{"name": "trigger", "device": "plc", "match": "startswith", "pattern": "TRIG",
                           "action": "trigger_flow", "flow": "main", "enabled": True}],
        "send_rules": [{"name": "result", "device": "plc", "flow": "main", "when": "always",
                        "template": "{status},{out.holes},{out.width:.1f}\\n", "enabled": True}],
    },
    "demo_modbus": {  # 视觉做 Modbus 从站，PLC 做主站来读写
        "devices": [{"name": "plc", "kind": "modbus_tcp_server", "config": {"host": "0.0.0.0", "port": 5020, "unit_id": 1,
                     "register_count": 256, "coil_count": 64, "busy_address": 1, "trigger_reset": True,
                     "heartbeat_address": 20, "heartbeat_s": 1.0}}],
        "receive_rules": [{"name": "trigger", "device": "plc", "match": "register_rising", "address": 0,
                           "action": "trigger_flow", "flow": "main", "enabled": True}],
        "send_rules": [{"name": "result", "device": "plc", "flow": "main", "when": "always", "template": "",
                        "registers": RESULT_REGS, "enabled": True}],
    },
    "demo_mc": {      # 三菱 MC 协议，PLC 是从站
        "devices": [{"name": "plc", "kind": "mc", "config": {"host": "127.0.0.1", "port": 5000, "poll_ms": 50,
                     "watch_device": "D", "watch_address": 0, "watch_count": 8, "busy_address": 1, "trigger_reset": True,
                     "heartbeat_address": 20, "heartbeat_s": 1.0}}],
        "receive_rules": [{"name": "trigger", "device": "plc", "match": "register_rising", "address": 0,
                           "action": "trigger_flow", "flow": "main", "enabled": True}],
        "send_rules": [{"name": "result", "device": "plc", "flow": "main", "when": "always", "template": "",
                        "registers": [{"address": "D10", "expr": "{out.holes}", "kind": "int16"},
                                      {"address": "D11", "expr": "{ok}", "kind": "uint16"},
                                      {"address": "D12", "expr": "{out.width}", "kind": "float32"}], "enabled": True}],
    },
    "demo_s7": {      # 西门子 S7，DB1 为交换区
        "devices": [{"name": "plc", "kind": "s7", "config": {"host": "127.0.0.1", "rack": 0, "slot": 1, "port": 1102,
                     "db_number": 1, "poll_ms": 50, "watch_offset": 0, "watch_bytes": 8, "busy_address": 2,
                     "trigger_reset": True, "heartbeat_address": 20, "heartbeat_s": 1.0}}],
        "receive_rules": [{"name": "trigger", "device": "plc", "match": "register_rising", "address": 0,
                           "action": "trigger_flow", "flow": "main", "enabled": True}],
        "send_rules": [{"name": "result", "device": "plc", "flow": "main", "when": "always", "template": "",
                        "registers": RESULT_REGS, "enabled": True}],
    },
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, comm in COMM_VARIANTS.items():
        sol = Solution(name)
        sol.add_flow(build_main())
        sol.add_flow(build_learning())
        sol.plugin_dirs = ["../plugins"]
        sol.variables.define("reward", 0.0, "float", "学习流程的奖励")
        sol.variables.define("product", "PART-A", "string", "当前产品型号")
        sol.comm_config = comm
        out = OUT.parent / f"{name}.json"
        sol.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()

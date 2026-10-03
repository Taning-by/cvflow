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
    g.description = "Count the three holes of the part and measure its width."
    src = place(g.add_node(registry.create("source.image_folder", name="Folder",
                                           values={"directory": IMAGES, "mode": "next"})), 0)
    gray = place(g.add_node(registry.create("preprocess.color", name="Gray", values={"mode": "gray"})), 1)
    th = place(g.add_node(registry.create("preprocess.threshold", name="Threshold",
                                          values={"method": "binary_inv", "thresh": 120})), 2)
    mo = place(g.add_node(registry.create("preprocess.morphology", name="Open",
                                          values={"op": "open", "ksize": 3})), 3)
    blob = place(g.add_node(registry.create("analysis.blob", name="Holes",
                                            values={"min_area": 500, "max_area": 5000})), 4)
    jc = place(g.add_node(registry.create("logic.judge", name="Hole Count",
                                          values={"op": "==", "low": 3, "name": "holes"})), 5)
    cal = place(g.add_node(registry.create("analysis.caliper", name="Width",
                                           values={"roi": {"x": 60, "y": 150, "w": 520, "h": 24},
                                                   "direction": "horizontal", "select": "first_last",
                                                   "min_contrast": 15})), 4, 1)
    jw = place(g.add_node(registry.create("logic.judge", name="Width OK",
                                          values={"op": "in_range", "low": 390, "high": 410, "name": "width"})), 5, 1)
    pub = place(g.add_node(registry.create("output.publish", name="Publish",
                                           values={"name_a": "holes", "name_b": "width"})), 6)
    rnd = place(g.add_node(registry.create("output.render", name="Render")), 6, 1)
    sav = place(g.add_node(registry.create("output.save_image", name="Save NG",
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
    g.description = "Bandit plugin tunes the threshold; reward is highest when 3 holes of nominal area are found."
    rew = place(g.add_node(registry.create("logic.get_variable", name="Reward", values={"name": "reward", "default": "0"})), 0, 1)
    bandit = place(g.add_node(registry.create("learning.bandit_threshold", name="Tuner",
                                              values={"low": 60, "high": 220, "step": 20, "epsilon": 0.2})), 1, 1)
    src = place(g.add_node(registry.create("source.image_folder", name="Folder",
                                           values={"directory": IMAGES, "mode": "fixed", "index": 0})), 0)
    gray = place(g.add_node(registry.create("preprocess.color", name="Gray", values={"mode": "gray"})), 1)
    th = place(g.add_node(registry.create("preprocess.threshold", name="Threshold", values={"method": "binary_inv"})), 2)
    blob = place(g.add_node(registry.create("analysis.blob", name="Holes", values={"min_area": 500, "max_area": 5000})), 3)
    exp = place(g.add_node(registry.create("logic.expression", name="Score", values={
        "expression": "(1.0 if a == 3 else 0.0) * max(0.0, 1.0 - abs(b - 1963) / 1963)"})), 4)
    setv = place(g.add_node(registry.create("logic.set_variable", name="Store", values={"name": "reward"})), 5)
    g.add_link(rew.id, "value", bandit.id, "reward")
    g.add_link(bandit.id, "threshold", th.id, "thresh")
    g.add_link(src.id, "image", gray.id, "image")
    g.add_link(gray.id, "image", th.id, "image")
    g.add_link(th.id, "image", blob.id, "image")
    g.add_link(blob.id, "count", exp.id, "a")
    g.add_link(blob.id, "largest_area", exp.id, "b")
    g.add_link(exp.id, "value", setv.id, "value")
    return g


def main() -> None:
    sol = Solution("demo_holes")
    sol.add_flow(build_main())
    sol.add_flow(build_learning())
    sol.plugin_dirs = ["../plugins"]
    sol.variables.define("reward", 0.0, "float", "reward for the learning flow")
    sol.variables.define("product", "PART-A", "string", "current product code")
    sol.comm_config = {
        "devices": [{"name": "plc", "kind": "tcp_server", "config": {"host": "0.0.0.0", "port": 6000, "terminator": "\\n"}}],
        "receive_rules": [{"name": "trigger", "device": "plc", "match": "startswith", "pattern": "TRIG",
                           "action": "trigger_flow", "flow": "main", "enabled": True}],
        "send_rules": [{"name": "result", "device": "plc", "flow": "main", "when": "always",
                        "template": "{status},{out.holes},{out.width:.1f}\\n", "enabled": True}],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sol.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()

from pathlib import Path

import cv2
import numpy as np
import pytest

from cvflow.core import Graph, Image, NodeStatus, Rect
from cvflow.core.engine import Engine
from conftest import make_blob_image


def run_single(reg, type_id, inputs: dict, values: dict | None = None):
    """Run one node with literal inputs via a tiny graph using a feeder node."""
    from cvflow.core import DataType, Node, Port

    class Feeder(Node):
        type_id = "test.feeder"
        outputs = [Port(k, DataType.ANY) for k in inputs]

        def process(self, ctx, _):
            return dict(inputs)

    g = Graph()
    f = g.add_node(Feeder())
    n = g.add_node(reg.create(type_id, values=values))
    for k in inputs:
        if n.get_input_port(k) is not None:
            g.add_link(f.id, k, n.id, k)
    r = Engine(g).run()
    nr = r.node_results[n.id]
    assert nr.status in (NodeStatus.OK, NodeStatus.NG), f"{type_id}: {nr.error}"
    return nr, r


def test_blob_pipeline(reg, blob_image):
    g = Graph()
    from cvflow.core import DataType, Node, Port

    class Src(Node):
        type_id = "test.src"
        outputs = [Port("image", DataType.IMAGE)]
        def process(self, ctx, inputs):
            return {"image": blob_image}

    s = g.add_node(Src())
    gray = g.add_node(reg.create("preprocess.color", values={"mode": "gray"}))
    th = g.add_node(reg.create("preprocess.threshold", values={"method": "binary", "thresh": 128}))
    blob = g.add_node(reg.create("analysis.blob", values={"min_area": 100, "max_area": 10000}))
    judge = g.add_node(reg.create("logic.judge", values={"op": "==", "low": 3}))
    pub = g.add_node(reg.create("output.publish", values={"name_a": "holes"}))
    g.add_link(s.id, "image", gray.id, "image")
    g.add_link(gray.id, "image", th.id, "image")
    g.add_link(th.id, "image", blob.id, "image")
    g.add_link(blob.id, "count", judge.id, "value")
    g.add_link(blob.id, "count", pub.id, "a")
    r = Engine(g).run()
    assert r.passed, r.error
    assert r.node_results[blob.id].outputs["count"] == 3
    areas = [b["area"] for b in r.node_results[blob.id].outputs["blobs"]]
    assert all(abs(a - 41 * 41) < 90 for a in areas)
    assert r.outputs == {"holes": 3}
    assert len(r.node_results[blob.id].overlays) >= 3


def test_threshold_input_override(reg):
    img = Image(np.full((10, 10), 100, np.uint8))
    nr, _ = run_single(reg, "preprocess.threshold", {"image": img, "thresh": 50}, {"thresh": 200})
    assert nr.outputs["image"].data.max() == 255 and nr.outputs["thresh"] == 50.0


def test_caliper_subpixel(reg):
    data = np.full((100, 300), 30, np.uint8)
    data[:, 50:150] = 220
    img = Image(data)
    nr, _ = run_single(reg, "analysis.caliper", {"image": img},
                       {"roi": {"x": 10, "y": 40, "w": 280, "h": 20}, "direction": "horizontal",
                        "select": "first_last", "min_contrast": 20, "smooth": 1})
    edges = nr.outputs["edges"]
    assert len(edges) == 2
    assert abs(edges[0] - 49.5) < 1.0 and abs(edges[1] - 149.5) < 1.0
    assert abs(nr.outputs["width"] - 100) < 1.0


def test_template_match(reg, blob_image):
    tpl = Image(blob_image.data[70:130, 10:70].copy())
    nr, _ = run_single(reg, "analysis.template_match", {"image": blob_image, "template": tpl},
                       {"min_score": 0.9, "max_matches": 3, "overlap": 0.1})
    assert nr.outputs["found"] and nr.outputs["count"] == 3
    centers = [(m["cx"], m["cy"]) for m in nr.outputs["matches"]]
    assert any(abs(cx - 40) <= 1 and abs(cy - 100) <= 1 for cx, cy in centers)


def test_circle_fit(reg):
    data = np.zeros((200, 200), np.uint8)
    cv2.circle(data, (100, 90), 40, 255, -1)
    nr, _ = run_single(reg, "analysis.circle", {"image": Image(data)},
                       {"method": "contour_fit", "min_radius": 10, "max_radius": 100})
    assert nr.outputs["found"]
    assert abs(nr.outputs["cx"] - 100) < 1 and abs(nr.outputs["cy"] - 90) < 1 and abs(nr.outputs["radius"] - 40) < 1
    nr, _ = run_single(reg, "analysis.circle", {"image": Image(data)},
                       {"method": "hough", "min_radius": 30, "max_radius": 50, "param2": 20})
    assert nr.outputs["found"] and abs(nr.outputs["radius"] - 40) < 3


def test_crop_morph_resize(reg, blob_image):
    nr, _ = run_single(reg, "preprocess.crop", {"image": blob_image}, {"roi": {"x": 10, "y": 70, "w": 60, "h": 60}})
    assert nr.outputs["image"].data.shape[:2] == (60, 60) and nr.outputs["offset"].x == 10
    nr, _ = run_single(reg, "preprocess.morphology", {"image": blob_image}, {"op": "erode", "ksize": 5})
    assert nr.outputs["image"].data.sum() < blob_image.data.sum()
    nr, _ = run_single(reg, "preprocess.resize", {"image": blob_image}, {"mode": "scale", "scale": 0.5})
    assert nr.outputs["image"].data.shape[:2] == (100, 100)
    nr, _ = run_single(reg, "preprocess.canny", {"image": blob_image})
    assert nr.outputs["image"].is_gray


def test_intensity_and_contours(reg, blob_image):
    nr, _ = run_single(reg, "analysis.intensity", {"image": blob_image}, {"roi": {"x": 20, "y": 80, "w": 40, "h": 40}})
    assert nr.outputs["mean"] > 250
    nr, _ = run_single(reg, "analysis.contours", {"image": blob_image}, {"min_area": 100})
    assert nr.outputs["count"] == 3


def test_logic_nodes(reg):
    nr, _ = run_single(reg, "logic.format", {"a": 3, "b": 2.5}, {"template": "R,{a},{b:.2f}"})
    assert nr.outputs["text"] == "R,3,2.50"
    nr, _ = run_single(reg, "logic.expression", {"a": 3, "b": 4}, {"expression": "sqrt(a**2 + b**2)"})
    assert nr.outputs["value"] == 5.0
    with pytest.raises(AssertionError):
        run_single(reg, "logic.expression", {"a": 1}, {"expression": "__import__('os')"})
    nr, r = run_single(reg, "logic.set_variable", {"value": 42}, {"name": "x"})
    assert r.variables["x"] == 42
    nr, _ = run_single(reg, "logic.judge", {"value": "ABC"}, {"op": "==", "text": "ABC"})
    assert nr.outputs["ok"] is True


def test_script_node_default_code(reg, blob_image):
    nr, _ = run_single(reg, "script.python", {"image": blob_image})
    assert nr.outputs["image"].is_gray and isinstance(nr.outputs["out1"], float) and nr.outputs["out2"] == 1
    assert nr.overlays


def test_script_node_custom_code_and_error(reg):
    code = "def process(ctx, inputs, params, state):\n    return {'out1': inputs['in1'] * 2}\n"
    nr, _ = run_single(reg, "script.python", {"in1": 21}, {"code": code})
    assert nr.outputs["out1"] == 42
    g = Graph()
    n = g.add_node(reg.create("script.python", values={"code": "x = 1"}))
    r = Engine(g).run()
    assert r.node_results[n.id].status == NodeStatus.ERROR and "process" in r.node_results[n.id].error


def test_render_and_save(reg, blob_image, tmp_path):
    from cvflow.core import Overlay
    g = Graph()
    from cvflow.core import DataType, Node, Port

    class Src(Node):
        type_id = "test.src2"
        outputs = [Port("image", DataType.IMAGE)]
        def process(self, ctx, inputs):
            ctx.add_overlay(Overlay.rect(Rect(5, 5, 50, 50), "#ff0000", "box"))
            ctx.judge(False, "forced")
            return {"image": blob_image}

    s = g.add_node(Src())
    rnd = g.add_node(reg.create("output.render"))
    sv = g.add_node(reg.create("output.save_image", values={"directory": str(tmp_path), "when": "ng_only"}))
    g.add_link(s.id, "image", rnd.id, "image")
    g.add_link(rnd.id, "image", sv.id, "image")
    r = Engine(g).run()
    assert r.status_text == "NG"
    path = Path(r.node_results[sv.id].outputs["path"])
    assert path.suffix == ".png" and path.parent.name == "NG" and path.parent.parent == tmp_path and path.is_file()
    out = r.node_results[rnd.id].outputs["image"].data
    assert out.ndim == 3 and (out[5:55, 5] == (0, 0, 255)).all(axis=1).any()


def test_image_folder_iterates(reg, images_dir):
    g = Graph()
    n = g.add_node(reg.create("source.image_folder", values={"directory": str(images_dir), "mode": "next"}))
    eng = Engine(g)
    seen = [eng.run().node_results[n.id].outputs["index"] for _ in range(12)]
    assert seen[:10] == list(range(10)) and seen[10] == 0  # 10 parts, then loop


def test_camera_folder(reg, images_dir):
    g = Graph()
    n = g.add_node(reg.create("source.camera", values={"camera": "testcam", "kind": "folder", "source": str(images_dir)}))
    eng = Engine(g)
    r1, r2 = eng.run(), eng.run()
    assert r1.node_results[n.id].outputs["frame_id"] == 1 and r2.node_results[n.id].outputs["frame_id"] == 2
    eng.teardown_nodes()


def test_demo_images_hole_count(reg, images_dir):
    """The synthetic parts: 3 holes normally, 2 on part_07, extra blemish on part_08."""
    def count(path):
        img = Image(cv2.imread(str(path)))
        g = Graph()
        from cvflow.core import DataType, Node, Port
        class Src(Node):
            type_id = "test.src3"
            outputs = [Port("image", DataType.IMAGE)]
            def process(self, ctx, inputs):
                return {"image": img}
        s = g.add_node(Src())
        th = g.add_node(reg.create("preprocess.threshold", values={"method": "binary_inv", "thresh": 120}))
        bl = g.add_node(reg.create("analysis.blob", values={"min_area": 500, "max_area": 5000}))
        g.add_link(s.id, "image", th.id, "image")
        g.add_link(th.id, "image", bl.id, "image")
        return Engine(g).run().node_results[bl.id].outputs["count"]
    assert count(images_dir / "part_00.png") == 3
    assert count(images_dir / "part_07.png") == 2
    assert count(images_dir / "part_08.png") == 4


def test_blob_accepts_colour_and_gray_input(reg, blob_image, images_dir):
    # colour image straight into Blob: auto -> Otsu, bright squares are the foreground
    nr, _ = run_single(reg, "analysis.blob", {"image": blob_image}, {"min_area": 100, "max_area": 10000})
    assert nr.outputs["count"] == 3 and nr.outputs["binary"].data.max() == 255
    # dark polarity on the real demo part: the three dark holes inside the bright part
    part = Image(cv2.imread(str(images_dir / "part_00.png")))
    nr, _ = run_single(reg, "analysis.blob", {"image": part}, {"polarity": "dark", "min_area": 500, "max_area": 5000})
    assert nr.outputs["count"] == 3
    # manual range
    nr, _ = run_single(reg, "analysis.blob", {"image": part}, {"binarize": "manual", "low": 0, "high": 100, "min_area": 500, "max_area": 5000})
    assert nr.outputs["count"] == 3
    # 'none' on a colour image collapses into one huge blob -> filtered out (the old behaviour)
    nr, _ = run_single(reg, "analysis.blob", {"image": blob_image}, {"binarize": "none", "max_area": 10000})
    assert nr.outputs["count"] == 3  # white squares on black are already binary

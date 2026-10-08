"""功能黑盒测试：每个节点按公开接口（输入、参数 → 输出）验证，期望值来自可解析计算的合成图像。
测试函数命名 test_<分类>_<节点>_<场景>，报告脚本据此按节点归类。"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from cvflow.core import Image, Point, Rect, registry
from cvflow.core.node import NodeStatus
from nodes_helpers import (ROOT, disk_gray, make_channel_mean_classifier, make_constant_detector, make_identity_model,
                           noisy_gray, qr_bgr, ramp_gray, run_node, run_ok, squares_bgr, squares_gray, step_gray, unique_gray)

registry.load_builtins()
registry.load_plugin_dir(ROOT / "examples" / "plugins")
IMAGES = ROOT / "examples" / "images"
reg = registry

# 一个用例同时覆盖多个节点时，在这里声明额外覆盖的节点（报告脚本使用）
EXTRA_COVERAGE = {
    "test_output_comm_send_and_modbus_write_with_manager": ["output.modbus_write"],
    "test_logic_variables_roundtrip_and_default": ["logic.set_variable", "logic.get_variable"],
}


# =============================================================== Source
def test_source_image_file_color_gray_and_missing(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), squares_bgr().data)
    nr = run_ok(reg, "source.image_file", values={"path": str(p)})
    assert nr.outputs["image"].channels == 3 and nr.outputs["path"] == str(p)
    nr = run_ok(reg, "source.image_file", values={"path": str(p), "grayscale": True})
    assert nr.outputs["image"].is_gray
    nr, _, _ = run_node(reg, "source.image_file", values={"path": str(tmp_path / "missing.png")})
    assert nr.status == NodeStatus.ERROR and "不存在" in nr.error
    nr, _, _ = run_node(reg, "source.image_file", values={"path": ""})
    assert nr.status == NodeStatus.ERROR


def test_source_image_folder_modes(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"i{i}.png"), np.full((4, 4), i * 50, np.uint8))
    node = reg.create("source.image_folder", values={"directory": str(tmp_path), "mode": "next", "loop": False})
    from cvflow.core import Graph
    from cvflow.core.engine import Engine
    g = Graph(); g.add_node(node); eng = Engine(g)
    seen = [eng.run().node_results[node.id].outputs["index"] for _ in range(3)]
    assert seen == [0, 1, 2]
    r = eng.run()
    assert r.node_results[node.id].status == NodeStatus.ERROR and "末尾" in r.node_results[node.id].error
    node.set("loop", True)
    assert eng.run().node_results[node.id].outputs["index"] == 0
    nr = run_ok(reg, "source.image_folder", values={"directory": str(tmp_path), "mode": "fixed", "index": 2})
    assert nr.outputs["index"] == 2 and nr.outputs["path"].endswith("i2.png") and nr.outputs["image"].data.max() == 100
    nr = run_ok(reg, "source.image_folder", values={"directory": str(tmp_path), "mode": "random"})
    assert 0 <= nr.outputs["index"] <= 2
    nr, _, _ = run_node(reg, "source.image_folder", values={"directory": str(tmp_path / "none")})
    assert nr.status == NodeStatus.ERROR


def test_source_camera_folder_and_errors(tmp_path):
    nr = run_ok(reg, "source.camera", values={"camera": "bb_cam", "kind": "folder", "source": str(IMAGES)})
    assert nr.outputs["frame_id"] == 1 and nr.outputs["image"].width == 640
    nr, _, _ = run_node(reg, "source.camera", values={"camera": "bb_cam2", "kind": "folder", "source": str(tmp_path)})
    assert nr.status == NodeStatus.ERROR
    nr, _, _ = run_node(reg, "source.camera", values={"camera": "bb_cam3", "kind": "opencv", "source": "/nonexistent.avi"})
    assert nr.status == NodeStatus.ERROR
    from cvflow.camera import camera_manager
    camera_manager.close_all()


def test_source_constant_kinds():
    assert run_ok(reg, "source.constant", values={"kind": "float", "value": "1.5"}).outputs["value"] == 1.5
    assert run_ok(reg, "source.constant", values={"kind": "int", "value": "7.9"}).outputs["value"] == 7
    assert run_ok(reg, "source.constant", values={"kind": "bool", "value": "yes"}).outputs["value"] is True
    assert run_ok(reg, "source.constant", values={"kind": "string", "value": "ab"}).outputs["value"] == "ab"
    nr, _, _ = run_node(reg, "source.constant", values={"kind": "float", "value": "x"})
    assert nr.status == NodeStatus.ERROR


def test_source_trigger_data():
    from cvflow.core import FlowRunner, Graph, Trigger, TriggerSource
    g = Graph(); n = g.add_node(reg.create("source.trigger"))
    r = FlowRunner(g).run_once(Trigger(TriggerSource.COMM, device="plc", message="TRIG,7", payload={"k": 1}))
    o = r.node_results[n.id].outputs
    assert o == {"message": "TRIG,7", "source": "comm", "device": "plc", "payload": {"k": 1},
                 "request_id": 0, "fields": {}, "value": None, "peer": ""}
    # 通信层接受请求时冻结的那一份参数：请求编号、来源对端、解析出来的字段
    g2 = Graph(); n2 = g2.add_node(reg.create("source.trigger", values={"field": "model"}))
    r2 = FlowRunner(g2).run_once(Trigger(TriggerSource.COMM, device="plc", payload={
        "request_id": 1001, "peer": "10.0.0.5:5000", "fields": {"model": "A1", "lot": 7}}))
    o2 = r2.node_results[n2.id].outputs
    assert o2["request_id"] == 1001 and o2["peer"] == "10.0.0.5:5000"
    assert o2["fields"] == {"model": "A1", "lot": 7} and o2["value"] == "A1"
    # required=True 时字段缺失要明确报错
    g3 = Graph(); n3 = g3.add_node(reg.create("source.trigger", values={"field": "nope", "required": True}))
    r3 = FlowRunner(g3).run_once(Trigger(TriggerSource.COMM, payload={"fields": {"a": 1}}))
    assert r3.node_results[n3.id].status.value == "error"
    assert "没有字段" in r3.node_results[n3.id].error


# =============================================================== Preprocess
def test_preprocess_color_modes_and_channel():
    img = Image(np.zeros((2, 2, 3), np.uint8)); img.data[:, :] = (255, 0, 0)  # 纯蓝 (BGR)
    assert run_ok(reg, "preprocess.color", {"image": img}, {"mode": "gray"}).outputs["image"].is_gray
    rgb = run_ok(reg, "preprocess.color", {"image": img}, {"mode": "rgb"}).outputs["image"].data
    assert tuple(rgb[0, 0]) == (0, 0, 255)
    hsv = run_ok(reg, "preprocess.color", {"image": img}, {"mode": "hsv"}).outputs["image"].data
    assert hsv[0, 0, 0] == 120 and hsv[0, 0, 1] == 255 and hsv[0, 0, 2] == 255
    ch = run_ok(reg, "preprocess.color", {"image": img}, {"mode": "bgr", "channel": 0}).outputs["image"]
    assert ch.is_gray and ch.data[0, 0] == 255


@pytest.mark.parametrize("method", ["gaussian", "median", "box", "bilateral"])
def test_preprocess_blur_reduces_noise(method):
    img = noisy_gray()
    sigma = 75.0 if method == "bilateral" else 2.0
    out = run_ok(reg, "preprocess.blur", {"image": img}, {"method": method, "ksize": 7, "sigma": sigma}).outputs["image"]
    assert out.data.shape == img.data.shape and out.data.std() < img.data.std() * 0.8


def test_preprocess_threshold_methods():
    img = ramp_gray()
    b = run_ok(reg, "preprocess.threshold", {"image": img}, {"method": "binary", "thresh": 100}).outputs["image"].data
    assert b[0, 100] == 0 and b[0, 101] == 255 and set(np.unique(b)) <= {0, 255}
    bi = run_ok(reg, "preprocess.threshold", {"image": img}, {"method": "binary_inv", "thresh": 100}).outputs["image"].data
    assert bi[0, 100] == 255 and bi[0, 101] == 0
    nr = run_ok(reg, "preprocess.threshold", {"image": img}, {"method": "otsu"})
    assert 100 <= nr.outputs["thresh"] <= 155
    rg = run_ok(reg, "preprocess.threshold", {"image": img}, {"method": "range", "thresh": 50, "high": 60}).outputs["image"].data
    assert rg[0, 49] == 0 and rg[0, 50] == 255 and rg[0, 60] == 255 and rg[0, 61] == 0
    ad = run_ok(reg, "preprocess.threshold", {"image": squares_gray()}, {"method": "adaptive_mean", "block_size": 31, "c": 5}).outputs["image"].data
    assert ad[100, 40] == 255
    linked = run_ok(reg, "preprocess.threshold", {"image": img, "thresh": 200}, {"thresh": 10}).outputs
    assert linked["thresh"] == 200.0 and linked["image"].data[0, 200] == 0 and linked["image"].data[0, 201] == 255
    color = run_ok(reg, "preprocess.threshold", {"image": squares_bgr()}, {"thresh": 128}).outputs["image"]
    assert color.is_gray


def test_preprocess_morphology_effects():
    img = squares_gray()
    area = lambda d: int((d > 0).sum())  # noqa: E731
    base = area(img.data)
    er = run_ok(reg, "preprocess.morphology", {"image": img}, {"op": "erode", "ksize": 5, "shape": "rect"}).outputs["image"].data
    di = run_ok(reg, "preprocess.morphology", {"image": img}, {"op": "dilate", "ksize": 5, "shape": "rect"}).outputs["image"].data
    assert area(er) == 3 * 37 * 37 and area(di) == 3 * 45 * 45 and base == 3 * 41 * 41
    speck = img.copy(); speck.data[10, 10] = 255
    op = run_ok(reg, "preprocess.morphology", {"image": speck}, {"op": "open", "ksize": 3, "shape": "rect"}).outputs["image"].data
    assert op[10, 10] == 0 and area(op) == base
    op_e = run_ok(reg, "preprocess.morphology", {"image": speck}, {"op": "open", "ksize": 3, "shape": "ellipse"}).outputs["image"].data
    assert area(op_e) == base - 3 * 4          # 3×3 椭圆核是十字形，每个方块削掉 4 个角
    hole = img.copy(); hole.data[100, 40] = 0
    cl = run_ok(reg, "preprocess.morphology", {"image": hole}, {"op": "close", "ksize": 3}).outputs["image"].data
    assert cl[100, 40] == 255
    gr = run_ok(reg, "preprocess.morphology", {"image": img}, {"op": "gradient", "ksize": 3}).outputs["image"].data
    assert gr[100, 40] == 0 and gr[80, 20] == 255


def test_preprocess_resize_scale_and_size():
    img = squares_bgr()
    o = run_ok(reg, "preprocess.resize", {"image": img}, {"mode": "scale", "scale": 0.25}).outputs
    assert o["image"].data.shape[:2] == (50, 50) and o["scale_x"] == 0.25
    o = run_ok(reg, "preprocess.resize", {"image": img}, {"mode": "size", "width": 123, "height": 45, "interpolation": "nearest"}).outputs
    assert o["image"].data.shape[:2] == (45, 123) and abs(o["scale_y"] - 0.225) < 1e-9


def test_preprocess_crop_param_input_and_errors():
    img = squares_bgr()
    o = run_ok(reg, "preprocess.crop", {"image": img}, {"roi": {"x": 20, "y": 80, "w": 41, "h": 41}}).outputs
    assert o["image"].data.shape[:2] == (41, 41) and o["image"].data.min() == 255 and o["offset"] == Point(20, 80)
    o = run_ok(reg, "preprocess.crop", {"image": img, "roi": Rect(190, 190, 50, 50)}, {}).outputs   # 越界自动裁剪
    assert o["image"].data.shape[:2] == (10, 10) and o["roi"].w == 10
    nr, _, _ = run_node(reg, "preprocess.crop", {"image": img}, {})
    assert nr.status == NodeStatus.ERROR and "ROI" in nr.error
    nr, _, _ = run_node(reg, "preprocess.crop", {"image": img}, {"roi": {"x": 500, "y": 500, "w": 10, "h": 10}})
    assert nr.status == NodeStatus.ERROR


def test_preprocess_canny_edges_of_square():
    d = np.zeros((100, 100), np.uint8); d[30:70, 30:70] = 255
    e = run_ok(reg, "preprocess.canny", {"image": Image(d)}, {"low": 50, "high": 150}).outputs["image"].data
    assert e.dtype == np.uint8 and 140 <= int((e > 0).sum()) <= 170   # 周长 160 左右
    assert e[50, 50] == 0 and (e[28:33, 50] > 0).any()


def test_preprocess_enhance_methods():
    img = Image(np.full((4, 4), 100, np.uint8))
    lin = run_ok(reg, "preprocess.enhance", {"image": img}, {"method": "linear", "alpha": 1.5, "beta": -20}).outputs["image"].data
    assert lin[0, 0] == 130
    ramp = Image(np.tile(np.arange(100, 200, dtype=np.uint8), (4, 1)))
    nz = run_ok(reg, "preprocess.enhance", {"image": ramp}, {"method": "normalize"}).outputs["image"].data
    assert nz.min() == 0 and nz.max() == 255
    eq = run_ok(reg, "preprocess.enhance", {"image": ramp}, {"method": "equalize"}).outputs["image"].data
    assert eq.min() < 10 and eq.max() > 245 and np.all(np.diff(eq[0].astype(int)) >= 0)
    cl = run_ok(reg, "preprocess.enhance", {"image": noisy_gray()}, {"method": "clahe", "clip_limit": 2.0, "tile": 4}).outputs["image"]
    assert cl.is_gray and cl.data.shape == (128, 128)


def test_preprocess_transform_pixel_mapping():
    img = unique_gray(4, 6)
    r90 = run_ok(reg, "preprocess.transform", {"image": img}, {"rotate": "90"}).outputs["image"].data
    assert r90.shape == (6, 4) and r90[0, 3] == img.data[0, 0] and r90[5, 3] == img.data[0, 5]
    r180 = run_ok(reg, "preprocess.transform", {"image": img}, {"rotate": "180"}).outputs["image"].data
    assert r180[0, 0] == img.data[3, 5]
    r270 = run_ok(reg, "preprocess.transform", {"image": img}, {"rotate": "270"}).outputs["image"].data
    assert r270.shape == (6, 4) and r270[0, 0] == img.data[0, 5]
    fh = run_ok(reg, "preprocess.transform", {"image": img}, {"flip": "horizontal"}).outputs["image"].data
    assert fh[0, 0] == img.data[0, 5]
    fv = run_ok(reg, "preprocess.transform", {"image": img}, {"flip": "vertical"}).outputs["image"].data
    assert fv[0, 0] == img.data[3, 0]
    both = run_ok(reg, "preprocess.transform", {"image": img}, {"flip": "both"}).outputs["image"].data
    assert both[0, 0] == img.data[3, 5]


def test_preprocess_bitwise_ops():
    a = Image(np.array([[0b1100, 0b1010]], np.uint8)); b = Image(np.array([[0b1010, 0b0110]], np.uint8))
    got = {op: run_ok(reg, "preprocess.bitwise", {"a": a, "b": b}, {"op": op}).outputs["image"].data[0].tolist()
           for op in ("and", "or", "xor", "subtract", "absdiff")}
    assert got["and"] == [0b1000, 0b0010] and got["or"] == [0b1110, 0b1110] and got["xor"] == [0b0110, 0b1100]
    assert got["subtract"] == [2, 4] and got["absdiff"] == [2, 4]
    nt = run_ok(reg, "preprocess.bitwise", {"a": a}, {"op": "not"}).outputs["image"].data[0].tolist()
    assert nt == [255 - 0b1100, 255 - 0b1010]
    nr, _, _ = run_node(reg, "preprocess.bitwise", {"a": a, "b": Image(np.zeros((2, 2), np.uint8))}, {"op": "and"})
    assert nr.status == NodeStatus.ERROR and "尺寸" in nr.error


# =============================================================== Analysis
def test_analysis_blob_geometry_sorting_filters():
    img = squares_gray()
    o = run_ok(reg, "analysis.blob", {"image": img}, {"min_area": 100, "max_area": 10000, "sort": "left_to_right"}).outputs
    assert o["count"] == 3 and [b["area"] for b in o["blobs"]] == [1681.0] * 3
    assert [round(b["cx"]) for b in o["blobs"]] == [40, 100, 160] and all(round(b["cy"]) == 100 for b in o["blobs"])
    assert [(b["x"], b["y"], b["w"], b["h"]) for b in o["blobs"]][0] == (20, 80, 41, 41)
    assert o["total_area"] == 3 * 1681 and o["largest_area"] == 1681 and int((o["mask"].data > 0).sum()) == 3 * 1681
    assert run_ok(reg, "analysis.blob", {"image": img}, {"min_area": 2000}).outputs["count"] == 0
    assert run_ok(reg, "analysis.blob", {"image": img}, {"max_count": 2}).outputs["count"] == 2
    assert run_ok(reg, "analysis.blob", {"image": img}, {"min_width": 50}).outputs["count"] == 0
    diag = np.zeros((10, 10), np.uint8); diag[2, 2] = diag[3, 3] = 255
    assert run_ok(reg, "analysis.blob", {"image": Image(diag)}, {"min_area": 1, "connectivity": "8"}).outputs["count"] == 1
    assert run_ok(reg, "analysis.blob", {"image": Image(diag)}, {"min_area": 1, "connectivity": "4"}).outputs["count"] == 2
    big = squares_gray(); big.data[0:10, 0:10] = 255
    o = run_ok(reg, "analysis.blob", {"image": big}, {"min_area": 1, "sort": "area_asc"}).outputs
    assert o["blobs"][0]["area"] == 100.0
    o = run_ok(reg, "analysis.blob", {"image": squares_bgr()}, {"min_area": 100}).outputs      # 彩色输入，自动二值化
    assert o["count"] == 3 and o["binary"].data.max() == 255
    part = Image(cv2.imread(str(IMAGES / "part_00.png")))
    assert run_ok(reg, "analysis.blob", {"image": part}, {"polarity": "dark", "min_area": 500, "max_area": 5000}).outputs["count"] == 3
    assert run_ok(reg, "analysis.blob", {"image": part}, {"binarize": "manual", "low": 0, "high": 100, "min_area": 500, "max_area": 5000}).outputs["count"] == 3


def test_analysis_contours_count_area_approx():
    o = run_ok(reg, "analysis.contours", {"image": squares_gray()}, {"min_area": 100}).outputs
    assert o["count"] == 3 and all(abs(a - 1600) < 1 for a in o["areas"]) and len(o["largest"]) == 1
    o = run_ok(reg, "analysis.contours", {"image": squares_gray()}, {"min_area": 100, "approx_eps": 2.0}).outputs
    assert all(len(c) == 4 for c in o["contours"])
    inner = squares_gray(); inner.data[95:105, 35:45] = 0
    assert run_ok(reg, "analysis.contours", {"image": inner}, {"mode": "external", "min_area": 10}).outputs["count"] == 3
    assert run_ok(reg, "analysis.contours", {"image": inner}, {"mode": "tree", "min_area": 10}).outputs["count"] == 4


def test_analysis_template_match_position_threshold_roi_and_errors(tmp_path):
    img = squares_bgr()
    tpl = Image(img.data[70:130, 10:70].copy())
    o = run_ok(reg, "analysis.template_match", {"image": img, "template": tpl}, {"min_score": 0.9, "max_matches": 5, "overlap": 0.1}).outputs
    assert o["found"] and o["count"] == 3 and o["score"] > 0.99
    assert sorted(round(m["cx"]) for m in o["matches"]) == [40, 100, 160]
    o = run_ok(reg, "analysis.template_match", {"image": img, "template": tpl}, {"max_matches": 1, "search_roi": {"x": 120, "y": 60, "w": 80, "h": 80}}).outputs
    assert o["count"] == 1 and round(o["center"].x) == 160
    noise_tpl = Image(np.random.default_rng(1).integers(0, 255, (30, 30), np.uint8))
    o = run_ok(reg, "analysis.template_match", {"image": img, "template": noise_tpl}, {"min_score": 0.9}).outputs
    assert not o["found"] and o["count"] == 0
    p = tmp_path / "t.png"; cv2.imwrite(str(p), tpl.data)
    assert run_ok(reg, "analysis.template_match", {"image": img}, {"template_path": str(p), "min_score": 0.9}).outputs["found"]
    nr, _, _ = run_node(reg, "analysis.template_match", {"image": img, "template": Image(np.zeros((300, 300), np.uint8))}, {})
    assert nr.status == NodeStatus.ERROR
    nr, _, _ = run_node(reg, "analysis.template_match", {"image": img}, {"template_path": ""})
    assert nr.status == NodeStatus.ERROR


def test_analysis_caliper_positions_polarity_select_direction():
    img = step_gray()
    roi = {"x": 10, "y": 40, "w": 280, "h": 20}
    o = run_ok(reg, "analysis.caliper", {"image": img}, {"roi": roi, "select": "all", "min_contrast": 20, "smooth": 1}).outputs
    assert o["count"] == 2 and abs(o["edges"][0] - 49.5) < 0.6 and abs(o["edges"][1] - 149.5) < 0.6 and abs(o["width"] - 100) < 1
    assert abs(run_ok(reg, "analysis.caliper", {"image": img}, {"roi": roi, "polarity": "dark_to_light", "smooth": 1}).outputs["first"] - 49.5) < 0.6
    assert abs(run_ok(reg, "analysis.caliper", {"image": img}, {"roi": roi, "polarity": "light_to_dark", "smooth": 1}).outputs["first"] - 149.5) < 0.6
    assert run_ok(reg, "analysis.caliper", {"image": img}, {"roi": roi, "select": "first", "smooth": 1}).outputs["count"] == 1
    assert abs(run_ok(reg, "analysis.caliper", {"image": img}, {"roi": roi, "select": "last", "smooth": 1}).outputs["first"] - 149.5) < 0.6
    assert run_ok(reg, "analysis.caliper", {"image": img}, {"roi": roi, "select": "strongest", "smooth": 1}).outputs["count"] == 1
    vert = Image(np.ascontiguousarray(img.data.T))
    o = run_ok(reg, "analysis.caliper", {"image": vert}, {"roi": {"x": 40, "y": 10, "w": 20, "h": 280}, "direction": "vertical", "smooth": 1}).outputs
    assert o["count"] == 2 and abs(o["edges"][0] - 49.5) < 0.6
    assert run_ok(reg, "analysis.caliper", {"image": img}, {"roi": roi, "min_contrast": 250}).outputs["count"] == 0
    o = run_ok(reg, "analysis.caliper", {"image": img, "roi": Rect(10, 40, 280, 20)}, {}).outputs
    assert o["count"] == 2
    nr, _, _ = run_node(reg, "analysis.caliper", {"image": img}, {})
    assert nr.status == NodeStatus.ERROR


def test_analysis_circle_fit_and_hough_accuracy():
    img = disk_gray(cx=100, cy=90, r=40)
    o = run_ok(reg, "analysis.circle", {"image": img}, {"method": "contour_fit", "min_radius": 10, "max_radius": 100}).outputs
    assert o["found"] and abs(o["cx"] - 100) < 1 and abs(o["cy"] - 90) < 1 and abs(o["radius"] - 40) < 1 and o["count"] == 1
    o = run_ok(reg, "analysis.circle", {"image": img}, {"method": "hough", "min_radius": 30, "max_radius": 50, "param2": 20}).outputs
    assert o["found"] and abs(o["cx"] - 100) < 3 and abs(o["radius"] - 40) < 3
    assert not run_ok(reg, "analysis.circle", {"image": img}, {"method": "contour_fit", "min_radius": 50, "max_radius": 100}).outputs["found"]
    two = img.copy(); cv2.circle(two.data, (160, 160), 20, 255, -1)
    o = run_ok(reg, "analysis.circle", {"image": two}, {"method": "contour_fit", "min_radius": 10, "max_radius": 100, "max_count": 5}).outputs
    assert o["count"] == 2 and abs(o["radius"] - 40) < 1     # 最大轮廓优先
    o = run_ok(reg, "analysis.circle", {"image": two}, {"method": "contour_fit", "min_radius": 10, "max_radius": 100, "roi": {"x": 130, "y": 130, "w": 60, "h": 60}}).outputs
    assert o["count"] == 1 and abs(o["cx"] - 160) < 1


def test_analysis_intensity_stats_exact():
    d = np.zeros((10, 10), np.uint8); d[:, 5:] = 200
    o = run_ok(reg, "analysis.intensity", {"image": Image(d)}, {"fill_thresh": 128}).outputs
    assert o["mean"] == 100 and o["min"] == 0 and o["max"] == 200 and o["fill_ratio"] == 0.5 and abs(o["std"] - 100) < 1e-6
    o = run_ok(reg, "analysis.intensity", {"image": Image(d)}, {"roi": {"x": 5, "y": 0, "w": 5, "h": 10}}).outputs
    assert o["mean"] == 200 and o["std"] == 0
    o = run_ok(reg, "analysis.intensity", {"image": Image(d), "roi": Rect(0, 0, 5, 10)}, {}).outputs
    assert o["mean"] == 0


def test_analysis_qrcode_decode_and_absent():
    o = run_ok(reg, "analysis.qrcode", {"image": qr_bgr("CVFLOW-2026")}).outputs
    assert o["found"] and o["text"] == "CVFLOW-2026" and len(o["points"]) == 4
    o = run_ok(reg, "analysis.qrcode", {"image": squares_bgr()}).outputs
    assert not o["found"] and o["text"] == "" and o["points"] == []


# =============================================================== Deep learning / script
def test_dl_onnx_inference_identity(tmp_path):
    model = make_identity_model(tmp_path / "id.onnx", (1, 3, 8, 8))
    img = Image(np.full((8, 8, 3), 255, np.uint8))
    o = run_ok(reg, "dl.onnx", {"image": img}, {"model_path": str(model), "width": 8, "height": 8, "scale": 1 / 255}).outputs
    assert o["output0"].shape == (1, 3, 8, 8) and np.allclose(o["output0"], 1.0) and o["infer_ms"] >= 0 and len(o["outputs"]) == 1
    o = run_ok(reg, "dl.onnx", {"image": img}, {"model_path": str(model), "width": 8, "height": 8, "scale": 1 / 255, "mean": "0.5", "std": "0.5"}).outputs
    assert np.allclose(o["output0"], 1.0)
    nr, _, _ = run_node(reg, "dl.onnx", {"image": img}, {"model_path": str(tmp_path / "no.onnx")})
    assert nr.status == NodeStatus.ERROR and "不存在" in nr.error


def test_dl_onnx_classifier_picks_brightest_channel(tmp_path):
    model = make_channel_mean_classifier(tmp_path / "cls.onnx", 8)
    (tmp_path / "labels.txt").write_text("blue\ngreen\nred\n", encoding="utf-8")
    img = Image(np.zeros((8, 8, 3), np.uint8)); img.data[:, :, 1] = 255      # BGR 中的绿色通道
    o = run_ok(reg, "dl.onnx_classifier", {"image": img}, {"model_path": str(model), "width": 8, "height": 8, "color": "bgr",
                                                           "labels_path": str(tmp_path / "labels.txt")}).outputs
    assert o["class_id"] == 1 and o["label"] == "green" and abs(sum(o["probs"]) - 1) < 1e-5 and o["score"] == max(o["probs"])
    o = run_ok(reg, "dl.onnx_classifier", {"image": img}, {"model_path": str(model), "width": 8, "height": 8, "color": "rgb"}).outputs
    assert o["class_id"] == 1 and o["label"] == "1"
    img.data[:] = 0; img.data[:, :, 0] = 255                                  # 蓝色：bgr 第 0 类，rgb 第 2 类
    assert run_ok(reg, "dl.onnx_classifier", {"image": img}, {"model_path": str(model), "width": 8, "height": 8, "color": "bgr"}).outputs["class_id"] == 0
    assert run_ok(reg, "dl.onnx_classifier", {"image": img}, {"model_path": str(model), "width": 8, "height": 8, "color": "rgb"}).outputs["class_id"] == 2


def test_dl_onnx_detector_decodes_boxes_and_nms(tmp_path):
    boxes = [(320, 320, 100, 50, 0, 0.9), (322, 321, 100, 50, 0, 0.6), (100, 100, 40, 40, 2, 0.8), (500, 500, 20, 20, 5, 0.1)]
    model = make_constant_detector(tmp_path / "det.onnx", boxes)
    (tmp_path / "coco.txt").write_text("\n".join(f"c{i}" for i in range(80)), encoding="utf-8")
    img = Image(np.zeros((640, 640, 3), np.uint8))
    o = run_ok(reg, "dl.onnx_detector", {"image": img}, {"model_path": str(model), "labels_path": str(tmp_path / "coco.txt"), "conf": 0.25, "iou": 0.45}).outputs
    assert o["count"] == 2 and o["best_label"] == "c0"
    d0 = o["detections"][0]
    assert (d0["x"], d0["y"], d0["w"], d0["h"], d0["class_id"]) == (270, 295, 100, 50, 0) and abs(d0["score"] - 0.9) < 1e-6
    assert o["detections"][1]["label"] == "c2" and isinstance(o["boxes"][0], Rect)
    o = run_ok(reg, "dl.onnx_detector", {"image": Image(np.zeros((480, 640, 3), np.uint8))}, {"model_path": str(model), "conf": 0.25}).outputs
    d = o["detections"][0]                                                    # letterbox: r=1, 竖向填充 80 像素
    assert (d["x"], d["y"]) == (270, 215)


def test_script_python_template_custom_setup_state_and_errors():
    img = squares_bgr()
    o = run_ok(reg, "script.python", {"image": img}).outputs
    assert o["image"].is_gray and isinstance(o["out1"], float) and o["out2"] == 1
    code = ("def setup(state):\n    state['init'] = True\n"
            "def process(ctx, inputs, params, state):\n    state['n'] = state.get('n', 0) + 1\n    return {'out1': inputs['in1'] * 2, 'out2': state['n'], 'out3': state['init']}\n")
    nr, _, node = run_node(reg, "script.python", {"in1": 21}, {"code": code})
    assert nr.outputs["out1"] == 42 and nr.outputs["out2"] == 1 and nr.outputs["out3"] is True
    nr, _, _ = run_node(reg, "script.python", {}, {"code": "x = 1"})
    assert nr.status == NodeStatus.ERROR and "process" in nr.error
    nr, _, _ = run_node(reg, "script.python", {}, {"code": "def process(ctx, inputs, params, state):\n    raise ValueError('boom')\n"})
    assert nr.status == NodeStatus.ERROR and "boom" in nr.error
    nr, _, _ = run_node(reg, "script.python", {}, {"code": "def process(:\n"})
    assert nr.status == NodeStatus.ERROR


# =============================================================== Logic
@pytest.mark.parametrize("op,value,params,expected", [
    ("in_range", 5, {"low": 0, "high": 10}, True), ("in_range", 11, {"low": 0, "high": 10}, False),
    ("==", 3, {"low": 3}, True), ("!=", 3, {"low": 3}, False), ("<", 2, {"low": 3}, True), ("<=", 3, {"low": 3}, True),
    (">", 3, {"low": 3}, False), (">=", 3, {"low": 3}, True), ("is_true", 0, {}, False), ("is_true", "x", {}, True),
    ("contains", "ABC-1", {"text": "BC"}, True), ("==", "A1", {"text": "A1"}, True), ("!=", "A1", {"text": "A2"}, True),
])
def test_logic_judge_operators(op, value, params, expected):
    nr, r, _ = run_node(reg, "logic.judge", {"value": value}, {"op": op, "name": "t", **params})
    assert nr.outputs["ok"] is expected and nr.outputs["value"] == value
    assert r.judgement is expected and nr.status == (NodeStatus.OK if expected else NodeStatus.NG)
    assert (nr.outputs["reason"] == "") == expected


def test_logic_expression_math_vars_and_safety():
    assert run_ok(reg, "logic.expression", {"a": 3, "b": 4}, {"expression": "sqrt(a**2 + b**2)"}).outputs["value"] == 5
    assert run_ok(reg, "logic.expression", {"a": [1, 2, 3]}, {"expression": "len(a) + max(a)"}).outputs["value"] == 6
    assert run_ok(reg, "logic.expression", {"a": 1}, {"expression": "1.0 if a == 1 else 0.0"}).outputs["value"] == 1.0
    assert run_ok(reg, "logic.expression", {"a": 1}, {"expression": "round(pi, 2)"}).outputs["value"] == 3.14
    for bad in ("__import__('os')", "open('x')", "a.__class__"):
        nr, _, _ = run_node(reg, "logic.expression", {"a": 1}, {"expression": bad})
        assert nr.status == NodeStatus.ERROR, bad
    nr, _, _ = run_node(reg, "logic.expression", {"a": 1}, {"expression": "a / 0"})
    assert nr.status == NodeStatus.ERROR


def test_logic_variables_roundtrip_and_default():
    from cvflow.core import FlowRunner, Graph
    g = Graph()
    c = g.add_node(reg.create("source.constant", values={"kind": "int", "value": "42"}))
    s = g.add_node(reg.create("logic.set_variable", values={"name": "v1"}))
    gt = g.add_node(reg.create("logic.get_variable", values={"name": "v1", "default": "none"}))
    g.add_link(c.id, "value", s.id, "value"); g.add_link(s.id, "value", gt.id, "value") if gt.get_input_port("value") else None
    runner = FlowRunner(g)
    r = runner.run_once()
    assert r.variables["v1"] == 42 and r.node_results[s.id].outputs["value"] == 42
    assert runner.run_once().node_results[gt.id].outputs["value"] == 42
    assert run_ok(reg, "logic.get_variable", values={"name": "missing", "default": "d"}).outputs["value"] == "d"


def test_logic_format_specs_and_errors():
    assert run_ok(reg, "logic.format", {"a": 3, "b": 2.345, "c": "x"}, {"template": "R,{a:03d},{b:.1f},{c}"}).outputs["text"] == "R,003,2.3,x"
    assert run_ok(reg, "logic.format", {}, {"template": "run{run_id}"}).outputs["text"] == "run1"
    nr, _, _ = run_node(reg, "logic.format", {"a": 1}, {"template": "{zzz}"})
    assert nr.status == NodeStatus.ERROR and "模板" in nr.error


def test_logic_gate_pass_block_invert():
    nr, _, _ = run_node(reg, "logic.gate", {"condition": True, "value": 7})
    assert nr.status == NodeStatus.OK and nr.outputs["value"] == 7
    nr, _, _ = run_node(reg, "logic.gate", {"condition": 0, "value": 7})
    assert nr.status == NodeStatus.SKIPPED and "value" not in nr.outputs
    nr, _, _ = run_node(reg, "logic.gate", {"condition": 0, "value": 7}, {"invert": True})
    assert nr.status == NodeStatus.OK and nr.outputs["value"] == 7


def test_logic_delay_waits():
    t0 = time.perf_counter()
    nr = run_ok(reg, "logic.delay", {"value": "x"}, {"ms": 120})
    assert nr.outputs["value"] == "x" and time.perf_counter() - t0 >= 0.11 and nr.time_ms >= 110


def test_logic_counter_increments_conditionally_and_resets():
    from cvflow.core import Graph
    from cvflow.core.engine import Engine
    g = Graph(); n = g.add_node(reg.create("logic.counter")); eng = Engine(g)
    assert [eng.run().node_results[n.id].outputs["count"] for _ in range(3)] == [1, 2, 3]
    n.set("reset", True)
    assert eng.run().node_results[n.id].outputs["count"] == 1 and n.get("reset") is False
    g2 = Graph(); c = g2.add_node(reg.create("source.constant", values={"kind": "bool", "value": "false"}))
    n2 = g2.add_node(reg.create("logic.counter")); g2.add_link(c.id, "value", n2.id, "count_if"); eng2 = Engine(g2)
    assert eng2.run().node_results[n2.id].outputs["count"] == 0
    c.set("value", "true")
    assert eng2.run().node_results[n2.id].outputs["count"] == 1


# =============================================================== Output
def test_output_publish_names_linked_only():
    nr, r, _ = run_node(reg, "output.publish", {"a": 1, "c": "z"}, {"name_a": "count", "name_c": "code"})
    assert r.outputs == {"count": 1, "code": "z"} and nr.outputs == {}


def test_output_render_draws_overlays_and_stamp():
    from cvflow.core import DataType, Graph, Node, Overlay, Port
    from cvflow.core.engine import Engine

    class Src(Node):
        type_id = "bb.src"; outputs = [Port("image", DataType.IMAGE)]
        def process(self, ctx, inputs):
            ctx.add_overlay(Overlay.rect(Rect(5, 5, 50, 50), "#ff0000")); ctx.add_overlay(Overlay.circle(__import__("cvflow.core", fromlist=["Circle"]).Circle(100, 100, 20), "#00ff00"))
            ctx.judge(False, "x"); return {"image": squares_gray()}
    g = Graph(); s = g.add_node(Src()); rnd = g.add_node(reg.create("output.render", values={"thickness": 2}))
    g.add_link(s.id, "image", rnd.id, "image")
    out = Engine(g).run().node_results[rnd.id].outputs["image"].data
    assert out.ndim == 3 and (out[5:56, 5] == (0, 0, 255)).all(axis=1).sum() > 40 and tuple(out[100, 120]) == (0, 255, 0)
    assert (out[5:35, 5:60] == (0, 0, 255)).all(axis=2).any()                   # NG 印章为红色
    g2 = Graph(); s2 = g2.add_node(Src()); rnd2 = g2.add_node(reg.create("output.render", values={"stamp_result": False}))
    g2.add_link(s2.id, "image", rnd2.id, "image")
    out2 = Engine(g2).run().node_results[rnd2.id].outputs["image"].data
    assert not (out2[8:28, 8:40] == (0, 0, 255)).all(axis=2).any()


def test_output_save_image_modes_formats_and_name(tmp_path):
    img = squares_bgr()
    for fmt in ("png", "jpg", "bmp"):
        nr = run_ok(reg, "output.save_image", {"image": img}, {"directory": str(tmp_path), "format": fmt, "subfolder_by_result": False})
        assert Path(nr.outputs["path"]).suffix == f".{fmt}" and Path(nr.outputs["path"]).is_file()
    nr = run_ok(reg, "output.save_image", {"image": img, "name": "part7"}, {"directory": str(tmp_path), "prefix": "x_", "subfolder_by_result": True})
    assert nr.outputs["path"].endswith("x_part7.png") and (tmp_path / "OK" / "x_part7.png").is_file()
    assert run_ok(reg, "output.save_image", {"image": img}, {"directory": str(tmp_path), "when": "ng_only"}).outputs["path"] == ""
    assert run_ok(reg, "output.save_image", {"image": img}, {"directory": str(tmp_path), "when": "ok_only"}).outputs["path"] != ""


def test_output_log_passthrough():
    assert run_ok(reg, "output.log", {"value": 3.5}, {"prefix": "v=", "level": "warning"}).outputs["value"] == 3.5


def test_output_comm_send_and_modbus_write_with_manager():
    from cvflow.comm import CommManager, set_manager
    from cvflow.core import EventBus
    mgr = CommManager(EventBus())
    dev = mgr.add_device("plc", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0}); dev.connect()
    tcp = mgr.add_device("hmi", "tcp_server", {"host": "127.0.0.1", "port": 0}); tcp.connect()
    set_manager(mgr)
    try:
        nr = run_ok(reg, "output.modbus_write", {"value": 12.5}, {"device": "plc", "address": "30", "kind": "float32"})
        assert nr.outputs["ok"] is True and dev.read_value(30, "float32") == 12.5
        assert run_ok(reg, "output.modbus_write", {"value": 3}, {"device": "plc", "address": "5", "kind": "int16", "scale": 10}).outputs["ok"] and dev.read_value(5) == 30
        assert run_ok(reg, "output.modbus_write", {"value": 1}, {"device": "nope", "address": "0"}).outputs["ok"] is False
        assert run_ok(reg, "output.comm_send", {"text": "hi"}, {"device": "hmi"}).outputs["sent"] is True   # 无客户端时静默丢弃
        assert run_ok(reg, "output.comm_send", {"text": "hi"}, {"device": "nope"}).outputs["sent"] is False
    finally:
        set_manager(None); mgr.shutdown()
    nr, _, _ = run_node(reg, "output.comm_send", {"text": "hi"}, {"device": "hmi"})
    assert nr.status == NodeStatus.ERROR


# =============================================================== Plugins
def test_learning_bandit_threshold_converges_and_freezes():
    from cvflow.core import Graph
    from cvflow.core.engine import Engine
    node = reg.create("learning.bandit_threshold", values={"low": 100, "high": 120, "step": 10, "epsilon": 0.0, "learning_rate": 0.5})
    g = Graph(); g.add_node(node); eng = Engine(g)
    from nodes_helpers import _Feeder
    rewards = {100: 0.0, 110: 1.0, 120: 0.2}
    for arm in (100, 110, 120, 110, 110):
        node.state["last_arm"] = arm
        f = _Feeder({"reward": rewards[arm]}); g.add_node(f); g.add_link(f.id, "reward", node.id, "reward")
        eng.run(); g.remove_node(f.id)
    out = eng.run().node_results[node.id].outputs
    assert out["best"] == 110 and out["threshold"] == 110 and out["q_values"]["110"] > out["q_values"]["120"] > out["q_values"]["100"]
    node.set("frozen", True); q = dict(node.state["q"])
    f = _Feeder({"reward": 0.0}); g.add_node(f); g.add_link(f.id, "reward", node.id, "reward")
    o = eng.run().node_results[node.id].outputs
    assert o["threshold"] == 110 and o["explore"] is False and node.state["q"] == q


def test_learning_torch_template_reports_missing_torch_cleanly():
    pytest.importorskip("numpy")
    try:
        import torch  # noqa: F401
        pytest.skip("torch 已安装")
    except ImportError:
        pass
    nr, _, _ = run_node(reg, "learning.torch_template", {"image": squares_bgr()})
    assert nr.status == NodeStatus.ERROR and "torch" in nr.error.lower()

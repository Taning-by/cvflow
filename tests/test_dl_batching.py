"""深度学习节点多输入合批的黑盒测试：只通过端口、参数与输出观察。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from cvflow.core import DataType, Graph, Image, Node, Port, registry
from cvflow.core.engine import Engine
from cvflow.core.node import NodeStatus
from nodes_helpers import channel_image, make_channel_mean_classifier, make_identity_model, run_node

reg = registry


def runners() -> int:
    from cvflow.operators.dl import active_runners
    return active_runners()


@pytest.fixture
def cls_model(tmp_path):
    """动态批次维的分类模型：输出哪个通道最亮，用来验证每一路结果的归属。"""
    return str(make_channel_mean_classifier(tmp_path / "cls.onnx", 8, dynamic_batch=True))


@pytest.fixture
def fixed_model(tmp_path):
    """批次维固定为 1 的模型。"""
    return str(make_channel_mean_classifier(tmp_path / "fixed.onnx", 8, dynamic_batch=False))


def cls_values(model, **kw):
    v = {"model_path": model, "width": 8, "height": 8, "color": "bgr", "scale": 1 / 255.0}
    v.update(kw)
    return v


class _Feed(Node):
    """把若干张图分别送到被测节点的各个输入端口。"""
    type_id = "test.feed_images"

    def __init__(self, images):
        self.__class__.outputs = [Port(f"i{k}", DataType.IMAGE) for k in range(len(images))]
        super().__init__()
        self._images = list(images)

    def process(self, ctx, inputs):
        return {f"i{k}": im for k, im in enumerate(self._images) if im is not None}


def build(model, images, **values):
    """搭一个"多图输入 → 深度学习节点"的流程，返回 (engine, node)。"""
    g = Graph("batch")
    node = reg.create("dl.onnx_classifier", values=cls_values(model, input_count=len(images), **values))
    g.add_node(node)
    feed = _Feed(images)
    g.add_node(feed)
    for k, im in enumerate(images):
        if im is not None:
            g.add_link(feed.id, f"i{k}", node.id, "image" if k == 0 else f"image{k + 1}")
    return Engine(g), node


# =============================================================== 端口
def test_dl_batch_input_count_builds_ports():
    n = reg.create("dl.onnx_detector", values={"input_count": 1})
    assert [p.name for p in n.inputs] == ["image"]
    assert [p.name for p in n.outputs] == ["detections", "count", "boxes", "best_label", "infer_ms", "batch_size"]
    n.set("input_count", 3)
    assert [p.name for p in n.inputs] == ["image", "image2", "image3"]
    assert [p.name for p in n.outputs][:8] == ["detections", "count", "boxes", "best_label",
                                               "detections2", "count2", "boxes2", "best_label2"]
    assert n.outputs[-2].name == "infer_ms" and n.outputs[-1].name == "batch_size"
    assert n.inputs[0].optional is False and n.inputs[1].optional is True   # 第一路必连，其余可留空
    n.set("input_count", 1)
    assert [p.name for p in n.inputs] == ["image"] and len(n.outputs) == 6
    n.set("input_count", 99)
    assert len(n.inputs) == 8                                               # 上限被夹住


def test_dl_batch_ports_survive_save_and_load(tmp_path):
    import json
    n = reg.create("dl.onnx_classifier", values={"input_count": 4})
    back = reg.node_from_dict(json.loads(json.dumps(n.to_dict())))
    assert [p.name for p in back.inputs] == ["image", "image2", "image3", "image4"]
    assert "score4" in [p.name for p in back.outputs]


# =============================================================== 合批与归属
def test_dl_batch_multiple_inputs_run_as_one_inference(cls_model):
    imgs = [channel_image(c) for c in (0, 1, 2, 1)]        # 蓝 绿 红 绿
    eng, node = build(cls_model, imgs)
    r = eng.run()
    nr = r.node_results[node.id]
    assert nr.status == NodeStatus.OK, nr.error
    assert nr.outputs["batch_size"] == 4                    # 四路合成了一个批次
    assert [nr.outputs[f"class_id{i}" if i > 1 else "class_id"] for i in (1, 2, 3, 4)] == [0, 1, 2, 1]
    for sfx in ("", "2", "3", "4"):                          # 分数就是该路概率里的最大值
        assert nr.outputs[f"score{sfx}"] == pytest.approx(max(nr.outputs[f"probs{sfx}"]))
        assert nr.outputs[f"probs{sfx}"].index(max(nr.outputs[f"probs{sfx}"])) == nr.outputs[f"class_id{sfx}"]
    assert isinstance(nr.outputs["probs3"], list) and len(nr.outputs["probs3"]) == 3
    assert nr.outputs["infer_ms"] >= 0
    eng.teardown_nodes()


def test_dl_batch_results_match_one_by_one(cls_model):
    imgs = [channel_image(c) for c in (2, 0, 1)]
    eng_b, node_b = build(cls_model, imgs, max_batch=8)
    rb = eng_b.run().node_results[node_b.id].outputs
    assert rb["batch_size"] == 3
    singles = []
    for im in imgs:                                          # 同样的图逐张推，结果必须一致
        eng_s, node_s = build(cls_model, [im], max_batch=1)
        singles.append(eng_s.run().node_results[node_s.id].outputs)
        eng_s.teardown_nodes()
    for k, one in enumerate(singles):
        suffix = "" if k == 0 else str(k + 1)
        assert rb[f"class_id{suffix}"] == one["class_id"]
        assert np.allclose(rb[f"probs{suffix}"], one["probs"], atol=1e-6)
    eng_b.teardown_nodes()


def test_dl_batch_unconnected_inputs_are_skipped(cls_model):
    imgs = [channel_image(0), None, channel_image(2), None]
    eng, node = build(cls_model, imgs)
    nr = eng.run().node_results[node.id]
    assert nr.status == NodeStatus.OK, nr.error
    assert nr.outputs["batch_size"] == 2                     # 只有两路连了，批里就是两张
    assert nr.outputs["class_id"] == 0 and nr.outputs["class_id3"] == 2
    assert nr.outputs["class_id2"] is None and nr.outputs["score4"] is None
    eng.teardown_nodes()


def test_dl_batch_all_inputs_missing_reports_error(cls_model):
    g = Graph()
    node = reg.create("dl.onnx_classifier", values=cls_values(cls_model, input_count=2))
    g.add_node(node)
    nr = Engine(g).run().node_results[node.id]
    assert nr.status == NodeStatus.SKIPPED and "未连接" in nr.error     # 第一路必连


def test_dl_batch_respects_max_batch(cls_model):
    imgs = [channel_image(c % 3) for c in range(6)]
    eng, node = build(cls_model, imgs, max_batch=2)
    nr = eng.run().node_results[node.id]
    assert nr.outputs["batch_size"] == 2                     # 六张按上限 2 分块执行
    assert [nr.outputs["class_id" if i == 0 else f"class_id{i + 1}"] for i in range(6)] == [0, 1, 2, 0, 1, 2]
    eng.teardown_nodes()


def test_dl_batch_fixed_batch_model_falls_back_to_one_by_one(fixed_model):
    imgs = [channel_image(c) for c in (0, 1, 2)]
    eng, node = build(fixed_model, imgs, max_batch=8)
    nr = eng.run().node_results[node.id]
    assert nr.status == NodeStatus.OK, nr.error
    assert nr.outputs["batch_size"] == 1                     # 模型批次维固定，退回逐张
    assert node.supports_batching is False
    assert [nr.outputs["class_id"], nr.outputs["class_id2"], nr.outputs["class_id3"]] == [0, 1, 2]
    eng.teardown_nodes()


# =============================================================== 等待窗口
def test_dl_batch_zero_wait_does_not_add_latency(cls_model):
    eng, node = build(cls_model, [channel_image(0)], wait_ms=0.0)
    eng.run()
    t0 = time.perf_counter()
    for _ in range(5):
        eng.run()
    assert (time.perf_counter() - t0) / 5 < 0.1              # 不等待，单路推理应当很快
    eng.teardown_nodes()


def test_dl_batch_wait_window_expires_and_runs_anyway(cls_model):
    eng, node = build(cls_model, [channel_image(1)], wait_ms=150.0)
    eng.run()                                                 # 预热，排除建会话的开销
    t0 = time.perf_counter()
    nr = eng.run().node_results[node.id]
    waited = time.perf_counter() - t0
    assert nr.outputs["class_id"] == 1 and nr.outputs["batch_size"] == 1
    assert 0.1 < waited < 2.0                                 # 没人来，等到窗口到期照常执行
    eng.teardown_nodes()


def test_dl_batch_window_merges_two_concurrent_flows(cls_model):
    """两个流程在各自线程里同时推理，等待窗口内被合成一个批次。"""
    engines = [build(cls_model, [channel_image(c)], wait_ms=400.0, max_batch=4) for c in (0, 2)]
    for eng, _ in engines:
        eng.setup_nodes()
    assert runners() >= 1
    barrier = threading.Barrier(2)
    out = {}

    def go(k):
        eng, node = engines[k]
        barrier.wait(5)
        out[k] = eng.run().node_results[node.id].outputs
    threads = [threading.Thread(target=go, args=(k,)) for k in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert out[0]["batch_size"] == 2 and out[1]["batch_size"] == 2     # 两路跨线程合成一批
    assert out[0]["class_id"] == 0 and out[1]["class_id"] == 2         # 各自拿回自己的结果
    for eng, _ in engines:
        eng.teardown_nodes()


# =============================================================== 会话共享
def test_dl_batch_same_config_shares_one_session(cls_model):
    base = runners()
    engines = [build(cls_model, [channel_image(0)]) for _ in range(3)]
    for eng, _ in engines:
        eng.setup_nodes()
    assert runners() == base + 1                              # 三个节点共用一个会话
    for eng, _ in engines[:2]:
        eng.teardown_nodes()
    assert runners() == base + 1                              # 还有引用，不释放
    engines[2][0].teardown_nodes()
    assert runners() == base                                  # 最后一个释放后回收


def test_dl_batch_group_and_preprocessing_split_sessions(cls_model):
    base = runners()
    a, _ = build(cls_model, [channel_image(0)], batch_group="线A")
    b, _ = build(cls_model, [channel_image(0)], batch_group="线B")
    c, _ = build(cls_model, [channel_image(0)], batch_group="线A")
    for eng in (a, b, c):
        eng.setup_nodes()
    assert runners() == base + 2                              # 分组名不同的互不合批
    d, _ = build(cls_model, [channel_image(0)], batch_group="线A", width=8, height=8, scale=1.0)
    d.setup_nodes()
    assert runners() == base + 3                              # 预处理不同也必须分开
    for eng in (a, b, c, d):
        eng.teardown_nodes()
    assert runners() == base


def test_dl_batch_changing_model_releases_old_session(tmp_path, cls_model):
    base = runners()
    other = str(make_channel_mean_classifier(tmp_path / "other.onnx", 8, dynamic_batch=True))
    eng, node = build(cls_model, [channel_image(0)])
    eng.setup_nodes()
    assert runners() == base + 1
    node.set("model_path", other)
    eng.run()
    assert runners() == base + 1                              # 换模型后旧会话被释放，不累积
    eng.teardown_nodes()
    assert runners() == base


# =============================================================== 叠加层
def test_dl_batch_overlay_comes_from_selected_input(cls_model):
    imgs = [channel_image(0), channel_image(1), channel_image(2)]
    for pick, expect in ((1, "0"), (2, "1"), (3, "2")):
        eng, node = build(cls_model, imgs, overlay_input=pick)
        nr = eng.run().node_results[node.id]
        texts = [o.geometry.get("text", "") for o in nr.overlays if o.kind == "text"]
        assert len(texts) == 1 and texts[0].startswith(expect)    # 只画选中那一路的结果
        eng.teardown_nodes()


# =============================================================== 其它两种深度学习节点
def test_dl_batch_works_for_inference_and_detector_nodes(tmp_path):
    ident = str(make_identity_model(tmp_path / "id.onnx", (1, 3, 8, 8), dynamic_batch=True))
    g = Graph()
    node = reg.create("dl.onnx", values={"model_path": ident, "width": 8, "height": 8, "input_count": 2,
                                         "scale": 1 / 255.0, "color": "bgr"})
    g.add_node(node)
    feed = _Feed([channel_image(0), channel_image(2)])
    g.add_node(feed)
    g.add_link(feed.id, "i0", node.id, "image")
    g.add_link(feed.id, "i1", node.id, "image2")
    nr = Engine(g).run().node_results[node.id]
    assert nr.status == NodeStatus.OK, nr.error
    assert nr.outputs["batch_size"] == 2
    a, b = nr.outputs["output0"], nr.outputs["output02"]
    assert a.shape == (1, 3, 8, 8) and b.shape == (1, 3, 8, 8)     # 拆回后仍是单张的形状
    assert np.allclose(a[0, 0], 1.0) and np.allclose(a[0, 2], 0.0)  # 第一路是蓝通道
    assert np.allclose(b[0, 2], 1.0) and np.allclose(b[0, 0], 0.0)  # 第二路是红通道
    Engine(g).teardown_nodes()


def test_dl_batch_detector_splits_per_input(tmp_path):
    from nodes_helpers import make_constant_detector
    det = str(make_constant_detector(tmp_path / "det.onnx", [(320, 320, 100, 50, 0, 0.9)]))
    g = Graph()
    node = reg.create("dl.onnx_detector", values={"model_path": det, "input_count": 2, "conf": 0.25})
    g.add_node(node)
    feed = _Feed([Image(np.zeros((640, 640, 3), np.uint8)), Image(np.zeros((640, 640, 3), np.uint8))])
    g.add_node(feed)
    g.add_link(feed.id, "i0", node.id, "image")
    g.add_link(feed.id, "i1", node.id, "image2")
    nr = Engine(g).run().node_results[node.id]
    assert nr.status == NodeStatus.OK, nr.error
    assert nr.outputs["count"] == 1 and nr.outputs["count2"] == 1
    assert nr.outputs["detections"][0]["x"] == nr.outputs["detections2"][0]["x"] == 270

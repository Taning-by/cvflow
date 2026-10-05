"""深度学习节点多输入合批的黑盒测试：只通过端口、参数与输出观察。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")      # 没装推理运行时就跳过（onnxruntime / onnxruntime-gpu 二选一）

from cvflow.core import DataType, Graph, Image, Node, Port, registry
from cvflow.core.engine import Engine
from cvflow.core.node import NodeStatus
from nodes_helpers import channel_image, make_channel_mean_classifier, make_identity_model, run_node

reg = registry


@pytest.fixture(autouse=True)
def cpu_backend_only(monkeypatch):
    """所有用例默认只认 CPU 后端。

    本机有没有可用的显卡会改变默认的会话归属（GPU 上每个节点各加载一份权重），
    会话份数的断言就不稳定了。需要别的后端组合的用例自己再覆盖这个桩。
    """
    import onnxruntime as ort
    from cvflow.operators import dl
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(dl, "_PROVIDER_LOADABLE", {})
    monkeypatch.setattr(dl, "_BACKEND_CHOICE_LOGGED", False)


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
    """填了合批分组时窗口会真的等；没人来就到期照常执行，不会卡住。"""
    eng, node = build(cls_model, [channel_image(1)], wait_ms=150.0, batch_group="线A")
    eng.run()                                                 # 预热，排除建会话的开销
    t0 = time.perf_counter()
    nr = eng.run().node_results[node.id]
    waited = time.perf_counter() - t0
    assert nr.outputs["class_id"] == 1 and nr.outputs["batch_size"] == 1
    assert 0.1 < waited < 2.0                                 # 没人来，等到窗口到期照常执行
    eng.teardown_nodes()


def test_dl_batch_wait_is_ignored_without_a_batch_group(cls_model):
    """没填合批分组时等待窗口不生效：队列里不可能有第二个提交者，等下去纯粹是浪费节拍。"""
    eng, node = build(cls_model, [channel_image(1)], wait_ms=300.0)
    eng.run()                                                 # 预热
    t0 = time.perf_counter()
    nr = eng.run().node_results[node.id]
    waited = time.perf_counter() - t0
    assert nr.outputs["class_id"] == 1 and nr.outputs["batch_size"] == 1
    assert waited < 0.1                                       # 完全不等
    eng.teardown_nodes()


def test_dl_batch_window_merges_two_concurrent_flows(cls_model):
    """填了相同合批分组的两个流程，在等待窗口内被合成一个批次。"""
    engines = [build(cls_model, [channel_image(c)], wait_ms=400.0, max_batch=4, batch_group="线A") for c in (0, 2)]
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


def test_dl_batch_window_merges_flows_that_arrive_at_different_times(cls_model):
    """两路图像不同时到达（各自被 PLC/定时器触发）时，合批分组 + 等待窗口仍然把它们凑成一批。

    这是"同一流程里的输入不一定同时到达"的解法：每路一个流程，各自被触发，
    深度学习节点填相同的合批分组，先到的在窗口里等后到的。
    """
    engines = [build(cls_model, [channel_image(c)], wait_ms=400.0, max_batch=4, batch_group="线B") for c in (0, 2)]
    for eng, _ in engines:
        eng.setup_nodes()
    out = {}

    def go(k, delay):
        time.sleep(delay)                                     # 第二路晚 120 ms 才被触发
        eng, node = engines[k]
        out[k] = eng.run().node_results[node.id].outputs

    threads = [threading.Thread(target=go, args=(0, 0.0)), threading.Thread(target=go, args=(1, 0.12))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert out[0]["batch_size"] == 2 and out[1]["batch_size"] == 2     # 错开到达也合成了一批
    assert out[0]["class_id"] == 0 and out[1]["class_id"] == 2         # 各自拿回自己的结果
    for eng, _ in engines:
        eng.teardown_nodes()


def _reject_gpu_providers(monkeypatch):
    """把 onnxruntime 换成"收到 GPU 后端就只用 CPU 建会话"，复现缺运行库时的静默回退。

    不这么做的话，用例在真有显卡的机器上会走到 GPU 分支，结论就反了。
    """
    import onnxruntime as ort
    real = ort.InferenceSession

    def fake(path, sess_options=None, providers=None, **kw):
        kept = [p for p in (providers or []) if (p[0] if isinstance(p, tuple) else p) not in
                ("CUDAExecutionProvider", "TensorrtExecutionProvider")]
        return real(path, sess_options, providers=kept or ["CPUExecutionProvider"], **kw)

    monkeypatch.setattr(ort, "InferenceSession", fake)


def test_dl_preload_loads_the_model_before_any_run(cls_model):
    """选好模型文件就该把权重加载进来，而不是等到第一次运行。"""
    from cvflow.operators.dl import active_sessions
    s0 = active_sessions()
    node = reg.create("dl.onnx_classifier", values=cls_values(cls_model))
    assert active_sessions() == s0                       # 还没加载
    node.preload()
    assert active_sessions() == s0 + 1                   # 一次都没运行，权重已经在了
    node.preload()
    assert active_sessions() == s0 + 1                   # 重复调用不会再加载一份
    node.teardown()
    assert active_sessions() == s0                       # 只预加载、没跑过的会话也会被放掉


def test_dl_gpu_selfcheck_command_runs(cls_model, capsys):
    """cvflow gpu 自检命令：没有显卡也要能跑完并说清实际用的后端。"""
    from cvflow.cli import main
    assert main(["gpu", cls_model, "--provider", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "推理运行时" in out and "后端" in out
    assert "CPUExecutionProvider" in out                  # 实测那段报出了真正用的后端


def test_dl_gpu_selfcheck_survives_windows_na_memory(cls_model, monkeypatch, capsys):
    """Windows 的 WDDM 驱动按进程查询显存会返回 [N/A]，不能把自检命令跑崩（回归）。"""
    import os
    from cvflow import cli
    real = cli._nvidia_smi

    def fake(query, extra=None):
        if query.startswith("compute-apps"):
            return [f"{os.getpid()}, [N/A]"]                  # Windows 的典型输出
        if query.startswith("gpu=memory.used"):
            return ["1544"]
        return real(query, extra)

    monkeypatch.setattr(cli, "_nvidia_smi", fake)
    assert cli.main(["gpu", cls_model, "--provider", "cpu"]) == 0
    assert "Traceback" not in capsys.readouterr().out


def test_dl_disabling_a_node_unloads_its_weights(cls_model):
    """禁用节点就把权重放掉，重新启用再加载回来。"""
    from cvflow.operators.dl import active_sessions
    s0 = active_sessions()
    node = reg.create("dl.onnx_classifier", values=cls_values(cls_model))
    node.preload()
    assert active_sessions() == s0 + 1

    node.set_enabled(False)
    assert active_sessions() == s0                        # 显存/内存里不再留着它

    node.preload()
    assert active_sessions() == s0                        # 禁用状态下不会被重新加载
    node.set_enabled(True)
    node.preload()
    assert active_sessions() == s0 + 1                    # 重新启用后又能加载回来
    node.teardown()
    assert active_sessions() == s0


def test_dl_disabling_one_node_keeps_the_weights_another_node_still_uses(cls_model):
    """权重是共享的：只要还有别的节点在用，禁用其中一个不该把它从显存里删掉。"""
    from cvflow.operators.dl import active_sessions
    s0 = active_sessions()
    a = reg.create("dl.onnx_classifier", values=cls_values(cls_model))
    b = reg.create("dl.onnx_classifier", values=cls_values(cls_model))
    a.preload(); b.preload()
    assert active_sessions() == s0 + 1                     # 同一个模型，本来就只有一份
    a.set_enabled(False)
    assert active_sessions() == s0 + 1                     # b 还在用，不能放掉
    b.set_enabled(False)
    assert active_sessions() == s0                         # 都不用了才真正释放


def test_dl_preload_drops_the_previous_copy_when_the_model_changes(cls_model, tmp_path):
    """换了模型文件，上一份没人用的权重要放掉，不能越堆越多。"""
    from cvflow.operators.dl import active_sessions
    other = str(make_channel_mean_classifier(tmp_path / "other.onnx", 8, dynamic_batch=True))
    s0 = active_sessions()
    node = reg.create("dl.onnx_classifier", values=cls_values(cls_model))
    node.preload()
    node.set("model_path", other)
    node.preload()
    assert active_sessions() == s0 + 1                    # 还是只有一份
    node.teardown()
    assert active_sessions() == s0


def test_dl_auto_skips_a_backend_whose_library_cannot_load(cls_model, monkeypatch):
    """TensorRT 装不全时不能连累 CUDA。

    onnxruntime 只要列表里有一个后端加载失败，整个会话就退回 CPU，
    所以必须先把加载不了的后端剔掉再建会话。
    """
    import onnxruntime as ort
    from cvflow.operators import dl
    monkeypatch.setattr(dl, "_FAILED_PROVIDERS", set())
    monkeypatch.setattr(dl, "_PROVIDER_LOADABLE", {})
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(dl, "_why_provider_failed",
                        lambda o, p: "libnvinfer.so.10: cannot open shared object file"
                        if p == "TensorrtExecutionProvider" else "")
    node = reg.create("dl.onnx_classifier", values=cls_values(cls_model))
    providers, on_gpu = node._resolve_providers()
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"] and on_gpu


def test_dl_gpu_request_preloads_cuda_runtime(cls_model, monkeypatch):
    """请求 GPU 后端时要先预加载 CUDA/cuDNN，否则 pip 装的运行库不在库搜索路径里会加载失败。"""
    import onnxruntime as ort
    from cvflow.operators import dl
    called = []
    monkeypatch.setattr(dl, "_GPU_RUNTIME_PRELOADED", False)
    monkeypatch.setattr(dl, "_FAILED_PROVIDERS", set())
    monkeypatch.setattr(ort, "preload_dlls", lambda *a, **k: called.append(1), raising=False)
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    _reject_gpu_providers(monkeypatch)
    eng, node = build(cls_model, [channel_image(1)], provider="cuda")
    eng.run()                                                    # 会退回 CPU，但预加载必须发生过
    assert called, "请求 GPU 后端时没有调用 preload_dlls"
    eng.teardown_nodes()


def test_dl_provider_unavailable_says_why_and_how_to_fix(cls_model, monkeypatch, caplog):
    """请求 GPU 后端但包里没有时，日志要说清原因和解决办法，而不是只说"不可用"。"""
    import onnxruntime as ort
    from cvflow.operators import dl
    monkeypatch.setattr(dl, "_FAILED_PROVIDERS", set())
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["AzureExecutionProvider", "CPUExecutionProvider"])
    eng, node = build(cls_model, [channel_image(1)], provider="cuda")
    with caplog.at_level("WARNING", logger="cvflow.dl"):
        nr = eng.run().node_results[node.id]
    assert nr.status.value == "ok"                                   # 退回 CPU，照常出结果
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "没有这个后端" in text and "onnxruntime-gpu" in text        # 原因 + 解决办法
    eng.teardown_nodes()


# =============================================================== 会话共享
def test_dl_batch_nodes_share_weights_but_not_the_batch_executor(cls_model):
    """默认每个节点独占批处理器以保证并行，但模型权重只加载一份。"""
    from cvflow.operators.dl import active_sessions
    s0, r0 = active_sessions(), runners()
    engines = [build(cls_model, [channel_image(0)]) for _ in range(3)]
    for eng, _ in engines:
        eng.setup_nodes()
    assert active_sessions() == s0 + 1                        # 权重共享：一份
    assert runners() == r0 + 3                                # 批处理器各管各的：三个
    for eng, _ in engines[:2]:
        eng.teardown_nodes()
    assert active_sessions() == s0 + 1 and runners() == r0 + 1
    engines[2][0].teardown_nodes()
    assert active_sessions() == s0 and runners() == r0


def test_dl_session_scope_exclusive_loads_one_copy_per_owner(cls_model):
    """独占会话：每个节点各加载一份权重；同一合批分组的节点仍然只有一份（它们本来就要排队合批）。"""
    from cvflow.operators.dl import active_sessions
    s0, r0 = active_sessions(), runners()
    engines = [build(cls_model, [channel_image(0)], session_scope="exclusive") for _ in range(3)]
    for eng, _ in engines:
        eng.setup_nodes()
    assert active_sessions() == s0 + 3 and runners() == r0 + 3      # 权重三份，批处理器三个
    for eng, _ in engines:
        eng.teardown_nodes()
    assert active_sessions() == s0 and runners() == r0

    grouped = [build(cls_model, [channel_image(0)], session_scope="exclusive", batch_group="线A") for _ in range(3)]
    for eng, _ in grouped:
        eng.setup_nodes()
    assert active_sessions() == s0 + 1 and runners() == r0 + 1      # 同组：权重一份，批处理器一个
    for eng, _ in grouped:
        eng.teardown_nodes()
    assert active_sessions() == s0 and runners() == r0


def test_dl_session_exclusive_falls_back_to_shared_when_it_cannot_be_created(cls_model, monkeypatch):
    """独占一份权重失败（显存不够）时退回共享会话，流程照常跑完，不是直接报错。"""
    from cvflow.operators import dl
    from cvflow.operators.dl import active_sessions
    real = dl._OnnxBase._create_session
    calls = {"n": 0}

    def flaky(self, path):
        calls["n"] += 1
        if calls["n"] == 1:                                   # 第一次（独占）失败
            raise RuntimeError("CUDA out of memory（模拟）")
        return real(self, path)

    monkeypatch.setattr(dl._OnnxBase, "_create_session", flaky)
    s0 = active_sessions()
    eng, node = build(cls_model, [channel_image(1)], session_scope="exclusive")
    nr = eng.run().node_results[node.id]
    assert nr.status.value == "ok" and nr.outputs["class_id"] == 1     # 没有因此失败
    assert active_sessions() == s0 + 1 and calls["n"] == 2             # 退回到了共享的那一份
    eng.teardown_nodes()
    assert active_sessions() == s0


def test_dl_session_scope_shared_keeps_one_copy_even_across_groups(cls_model):
    """强制共享：不管分组怎么填，权重都只加载一份。"""
    from cvflow.operators.dl import active_sessions
    s0 = active_sessions()
    engines = [build(cls_model, [channel_image(0)], session_scope="shared", batch_group=g) for g in ("甲", "乙")]
    for eng, _ in engines:
        eng.setup_nodes()
    assert active_sessions() == s0 + 1 and runners() >= 2
    for eng, _ in engines:
        eng.teardown_nodes()
    assert active_sessions() == s0


def test_dl_device_id_does_not_split_cpu_sessions(cls_model):
    """显卡编号只在 GPU 后端下参与会话归属，CPU 上填什么都还是同一份权重。"""
    from cvflow.operators.dl import active_sessions
    s0 = active_sessions()
    engines = [build(cls_model, [channel_image(0)], provider="cpu", device_id=d) for d in (0, 1)]
    for eng, _ in engines:
        eng.setup_nodes()
    assert active_sessions() == s0 + 1
    for eng, _ in engines:
        eng.teardown_nodes()
    assert active_sessions() == s0


def test_dl_gpu_default_gives_each_node_its_own_weights(cls_model, monkeypatch):
    """GPU 上 auto 的默认：不同节点各加载一份权重（能用各自的流并行），同组的仍共用一份。

    本机不一定有可用的 CUDA 运行库，所以只把"落在 GPU 上"这一判断打桩，
    验证的是会话归属规则本身，不是真的去跑显卡。
    """
    from cvflow.operators import dl
    from cvflow.operators.dl import active_sessions
    monkeypatch.setattr(dl, "_FAILED_PROVIDERS", set())          # 别污染其它用例的后端可用性判断
    monkeypatch.setattr(dl._OnnxBase, "_resolve_providers",
                        lambda self: (["CUDAExecutionProvider", "CPUExecutionProvider"], True))
    s0 = active_sessions()
    engines = [build(cls_model, [channel_image(0)]) for _ in range(2)]
    for eng, _ in engines:
        eng.setup_nodes()
    assert active_sessions() == s0 + 2                              # 各一份
    for eng, _ in engines:
        eng.teardown_nodes()
    grouped = [build(cls_model, [channel_image(0)], batch_group="线A") for _ in range(2)]
    for eng, _ in grouped:
        eng.setup_nodes()
    assert active_sessions() == s0 + 1                              # 同组共用一份
    for eng, _ in grouped:
        eng.teardown_nodes()
    assert active_sessions() == s0


def test_dl_batch_group_makes_nodes_share_one_executor(cls_model):
    """填了相同分组名的节点共用一个批处理器，不同分组名的互相独立。"""
    from cvflow.operators.dl import active_sessions
    s0, r0 = active_sessions(), runners()
    same = [build(cls_model, [channel_image(0)], batch_group="线A") for _ in range(3)]
    other = build(cls_model, [channel_image(0)], batch_group="线B")
    plain = build(cls_model, [channel_image(0)])
    for eng, _ in same + [other, plain]:
        eng.setup_nodes()
    assert active_sessions() == s0 + 1                        # 仍然只有一份权重
    assert runners() == r0 + 3                                # 线A 一个、线B 一个、未分组的一个
    assert same[0][1]._runner is same[1][1]._runner is same[2][1]._runner
    assert other[1]._runner is not same[0][1]._runner
    assert plain[1]._runner is not same[0][1]._runner
    for eng, _ in same + [other, plain]:
        eng.teardown_nodes()
    assert active_sessions() == s0 and runners() == r0


def test_dl_batch_independent_nodes_infer_in_parallel(cls_model):
    """默认配置下两个节点必须能同时推理；被串行化时屏障会超时。"""
    import threading
    engines = [build(cls_model, [channel_image(0)]) for _ in range(2)]
    for eng, _ in engines:
        eng.setup_nodes()
    n1, n2 = engines[0][1], engines[1][1]
    assert n1._runner.holder.session is n2._runner.holder.session     # 共用会话
    assert n1._runner is not n2._runner                               # 不共用批处理器
    sess = n1._runner.holder.session
    orig = sess.run
    gate = threading.Barrier(2)

    def wrapped(o, f):
        try:
            gate.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return orig(o, f)
    sess.run = wrapped
    start = threading.Barrier(2)
    threads = [threading.Thread(target=lambda e=e: (start.wait(5), e[0].run())) for e in engines]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    sess.run = orig
    assert not gate.broken, "两个节点没能同时进入推理，说明被串行化了"
    for eng, _ in engines:
        eng.teardown_nodes()


def test_dl_batch_group_still_split_by_preprocessing(cls_model):
    """同一个分组名但预处理不同的节点不能合批，因为张量形状对不上。"""
    base = runners()
    a, _ = build(cls_model, [channel_image(0)], batch_group="线A")
    b, _ = build(cls_model, [channel_image(0)], batch_group="线A")
    c, _ = build(cls_model, [channel_image(0)], batch_group="线A", scale=1.0)
    for eng in (a, b, c):
        eng.setup_nodes()
    assert runners() == base + 2                              # 预处理不同的被分开
    for eng in (a, b, c):
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
def test_dl_batch_overlays_are_tagged_with_their_source_input(cls_model):
    """每一路都产生叠加层，并标记来自哪个输入端口，图像窗口据此过滤。"""
    imgs = [channel_image(0), channel_image(1), channel_image(2)]
    eng, node = build(cls_model, imgs)
    nr = eng.run().node_results[node.id]
    texts = [(o.group, o.geometry.get("text", "")) for o in nr.overlays if o.kind == "text"]
    assert len(texts) == 3
    assert [g for g, _ in texts] == ["image", "image2", "image3"]
    assert [t.split()[0] for _, t in texts] == ["0", "1", "2"]    # 每一路画的是自己的结果
    eng.teardown_nodes()


def test_dl_batch_unconnected_input_produces_no_overlay(cls_model):
    eng, node = build(cls_model, [channel_image(2), None, channel_image(0)])
    nr = eng.run().node_results[node.id]
    groups = sorted({o.group for o in nr.overlays if o.kind == "text"})
    assert groups == ["image", "image3"]
    eng.teardown_nodes()


def test_dl_batch_single_input_overlays_stay_ungrouped(cls_model):
    """单输入节点的叠加层不带分组，老流程的显示行为完全不变。"""
    eng, node = build(cls_model, [channel_image(1)])
    nr = eng.run().node_results[node.id]
    assert [o.group for o in nr.overlays if o.kind == "text"] == ["image"]
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


# =============================================================== 模型输入形状自适应
def _run_onnx(model, img, **values):
    g = Graph()
    node = reg.create("dl.onnx", values={"model_path": model, "scale": 1 / 255.0, **values})
    g.add_node(node)
    feed = _Feed([img])
    g.add_node(feed)
    g.add_link(feed.id, "i0", node.id, "image")
    eng = Engine(g)
    return eng, node, eng.run().node_results[node.id]


def test_dl_shape_fixed_size_model_corrects_width_and_height(tmp_path):
    """模型写死 512×512，节点却是 YOLO 默认的 640×640：应自动按模型调整而不是报错。"""
    from nodes_helpers import make_shaped_model
    model = str(make_shaped_model(tmp_path / "s512.onnx", ["b", 3, 512, 512]))
    eng, node, nr = _run_onnx(model, channel_image(1, 64), width=640, height=640, color="bgr")
    assert nr.status == NodeStatus.OK, nr.error
    assert node.get("width") == 512 and node.get("height") == 512      # 参数被改成模型要求的值
    assert nr.outputs["output0"].shape == (1, 3, 512, 512)
    eng.teardown_nodes()


def test_dl_shape_nhwc_model_corrects_layout(tmp_path):
    from nodes_helpers import make_shaped_model
    model = str(make_shaped_model(tmp_path / "nhwc.onnx", ["b", 32, 32, 3]))
    eng, node, nr = _run_onnx(model, channel_image(0, 64), width=640, height=640, layout="NCHW", color="bgr")
    assert nr.status == NodeStatus.OK, nr.error
    assert node.get("layout") == "NHWC" and node.get("width") == 32 and node.get("height") == 32
    assert nr.outputs["output0"].shape == (1, 32, 32, 3)
    eng.teardown_nodes()


def test_dl_shape_single_channel_model_corrects_color(tmp_path):
    from nodes_helpers import make_shaped_model
    model = str(make_shaped_model(tmp_path / "gray.onnx", ["b", 1, 16, 16]))
    eng, node, nr = _run_onnx(model, channel_image(2, 64), color="rgb", width=640, height=640)
    assert nr.status == NodeStatus.OK, nr.error
    assert node.get("color") == "gray" and nr.outputs["output0"].shape == (1, 1, 16, 16)
    eng.teardown_nodes()


def test_dl_shape_dynamic_model_keeps_user_parameters(tmp_path):
    from nodes_helpers import make_shaped_model
    model = str(make_shaped_model(tmp_path / "dyn.onnx", ["b", 3, "h", "w"]))
    eng, node, nr = _run_onnx(model, channel_image(1, 64), width=96, height=48, color="bgr")
    assert nr.status == NodeStatus.OK, nr.error
    assert node.get("width") == 96 and node.get("height") == 48        # 动态尺寸完全听节点参数
    assert nr.outputs["output0"].shape == (1, 3, 48, 96)
    eng.teardown_nodes()


def test_dl_shape_unfixable_mismatch_reports_both_shapes(tmp_path):
    """通道数不是 1/3/4，软件无法推断布局，此时必须给出说明两边形状的中文报错。"""
    from nodes_helpers import make_shaped_model
    model = str(make_shaped_model(tmp_path / "ch5.onnx", ["b", 5, 8, 8]))
    eng, node, nr = _run_onnx(model, channel_image(0, 64), width=8, height=8, color="bgr")
    assert nr.status == NodeStatus.ERROR
    assert "模型期望输入" in nr.error and "实际送入" in nr.error
    assert "5×8×8" in nr.error and "3×8×8" in nr.error                 # 两边形状都报出来
    assert "宽度、高度、颜色、张量布局" in nr.error                      # 指明该调哪些参数


def test_dl_shape_broken_model_file_reports_clearly(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model at all")
    eng, node, nr = _run_onnx(str(bad), channel_image(0, 16))
    assert nr.status == NodeStatus.ERROR and "加载模型失败" in nr.error and "bad.onnx" in nr.error


def test_dl_shape_same_model_different_preprocessing_shares_one_session(tmp_path):
    """预处理不同的两个节点不能合批，但模型权重只该加载一份。"""
    from cvflow.operators.dl import active_sessions
    from nodes_helpers import make_shaped_model
    model = str(make_shaped_model(tmp_path / "dyn2.onnx", ["b", 3, "h", "w"]))
    s0, r0 = active_sessions(), runners()
    a, _, _ = _run_onnx(model, channel_image(0, 32), width=16, height=16, color="bgr")
    b, _, _ = _run_onnx(model, channel_image(0, 32), width=32, height=32, color="bgr")
    assert active_sessions() == s0 + 1                                  # 一份权重
    assert runners() == r0 + 2                                          # 两个独立的批处理执行器
    a.teardown_nodes()
    assert active_sessions() == s0 + 1
    b.teardown_nodes()
    assert active_sessions() == s0 and runners() == r0                  # 全部释放


def test_dl_detector_unparsable_output_reports_clearly(tmp_path):
    """检测节点接到不是检测布局的输出时，要给出能看懂的提示而不是 numpy 的下标错误。"""
    from nodes_helpers import make_shaped_model
    model = str(make_shaped_model(tmp_path / "notdet.onnx", ["b", 3, 32, 32]))
    g = Graph()
    node = reg.create("dl.onnx_detector", values={"model_path": model, "color": "bgr"})
    g.add_node(node)
    feed = _Feed([Image(np.zeros((64, 64, 3), np.uint8))])
    g.add_node(feed)
    g.add_link(feed.id, "i0", node.id, "image")
    nr = Engine(g).run().node_results[node.id]
    assert nr.status == NodeStatus.ERROR
    assert "无法解析检测输出" in nr.error and "32" in nr.error
    assert "脚本节点" in nr.error                                   # 给出替代做法
    assert "IndexError" not in nr.error


def test_dl_detector_too_few_attributes_reports_clearly(tmp_path):
    """输出布局像检测但每个候选框只有 4 个属性，缺类别分数，同样要说清楚。"""
    from nodes_helpers import make_constant_detector
    model = str(make_constant_detector(tmp_path / "thin.onnx", [(1, 2, 3, 4, 0, 0.9)] * 3, n_cls=0))
    g = Graph()
    node = reg.create("dl.onnx_detector", values={"model_path": model})
    g.add_node(node)
    feed = _Feed([Image(np.zeros((640, 640, 3), np.uint8))])
    g.add_node(feed)
    g.add_link(feed.id, "i0", node.id, "image")
    nr = Engine(g).run().node_results[node.id]
    assert nr.status == NodeStatus.ERROR
    assert "无法解析检测输出" in nr.error and "4 个属性" in nr.error


# =============================================================== 预处理
@pytest.mark.parametrize("values", [
    {"color": "rgb", "layout": "NCHW", "letterbox": True, "width": 64, "height": 64},
    {"color": "bgr", "layout": "NCHW", "letterbox": False, "width": 48, "height": 32},
    {"color": "gray", "layout": "NCHW", "letterbox": True, "width": 40, "height": 40},
    {"color": "rgb", "layout": "NCHW", "letterbox": True, "width": 64, "height": 64,
     "mean": "0.5,0.5,0.5", "std": "0.25,0.25,0.25"},
    {"color": "rgb", "layout": "NCHW", "letterbox": True, "width": 64, "height": 64,
     "mean": "0.485,0.456,0.406", "std": "0.229,0.224,0.225"},
    {"color": "rgb", "layout": "NHWC", "letterbox": False, "width": 40, "height": 24},
])
def test_dl_preprocess_matches_reference_formula(values):
    """预处理有一条 OpenCV 快速路径，结果必须与 (像素*scale - 均值)/标准差 的定义完全一致。"""
    import cv2
    img = Image(np.random.default_rng(3).integers(0, 255, (97, 123, 3), dtype=np.uint8))
    node = reg.create("dl.onnx", values={"scale": 1 / 255.0, **values})
    got, meta = node._preprocess(img)
    canvas, meta2 = node._to_canvas(img)
    c = 1 if canvas.ndim == 2 else canvas.shape[2]
    scale, mean, std = node._norm_params(c)
    if values["color"] == "rgb" and canvas.ndim == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if canvas.ndim == 2:
        canvas = canvas[:, :, None]
    ref = (canvas.astype(np.float32) * scale - mean) / std
    if values["layout"] == "NCHW":
        ref = ref.transpose(2, 0, 1)
    assert got.shape == ref[None].shape and meta == meta2
    assert np.abs(got - ref[None]).max() < 1e-5


def test_dl_preprocess_letterbox_geometry_is_recoverable():
    """等比填充要记录缩放与偏移，否则检测框还原不回原图坐标。"""
    img = Image(np.zeros((300, 600, 3), np.uint8))
    node = reg.create("dl.onnx", values={"width": 640, "height": 640, "letterbox": True})
    blob, meta = node._preprocess(img)
    assert blob.shape == (1, 3, 640, 640)
    assert meta["scale"] == pytest.approx(640 / 600)
    assert meta["pad"] == (0, (640 - int(round(300 * 640 / 600))) // 2)
    assert meta["orig"] == (600, 300)


def test_dl_batch_shared_buffer_keeps_results_separate(cls_model):
    """多路输入共用一块缓冲提交，每一路拿回的必须仍是自己那一行。"""
    imgs = [channel_image(c) for c in (2, 0, 1, 2, 1, 0)]
    eng, node = build(cls_model, imgs, max_batch=8)
    nr = eng.run().node_results[node.id]
    assert nr.outputs["batch_size"] == 6
    got = [nr.outputs["class_id" if k == 0 else f"class_id{k + 1}"] for k in range(6)]
    assert got == [2, 0, 1, 2, 1, 0]
    eng.teardown_nodes()


def test_dl_batch_fixed_batch_model_warns_once(fixed_model, caplog):
    """模型不支持合批却设了批上限，要在日志里说清楚，否则用户只能靠 batch_size 猜。"""
    import logging
    caplog.set_level(logging.WARNING, logger="cvflow.dl")
    eng, node = build(fixed_model, [channel_image(0), channel_image(1)], max_batch=8)
    nr = eng.run().node_results[node.id]
    assert nr.outputs["batch_size"] == 1
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("无法合批" in m for m in msgs) == 1          # 只提示一次，不在每次运行时刷屏
    caplog.clear()
    for _ in range(3):
        eng.run()
    assert not [r for r in caplog.records if "无法合批" in r.getMessage()]
    eng.teardown_nodes()


# =============================================================== 推理后端
@pytest.fixture
def clean_failed_providers():
    from cvflow.operators import dl
    saved = set(dl._FAILED_PROVIDERS)
    dl._FAILED_PROVIDERS.clear()
    yield dl._FAILED_PROVIDERS
    dl._FAILED_PROVIDERS.clear()
    dl._FAILED_PROVIDERS.update(saved)


def test_dl_provider_load_failure_is_reported_and_not_retried(cls_model, monkeypatch, caplog, clean_failed_providers):
    """装了 GPU 版但缺运行库时，后端会加载失败并静默回退到 CPU，必须说清楚并且不重复刷屏。"""
    import logging
    import onnxruntime as ort
    from cvflow.operators import dl
    monkeypatch.setattr(ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    _reject_gpu_providers(monkeypatch)                               # 本机有没有显卡都走同一条路径
    caplog.set_level(logging.INFO, logger="cvflow.dl")
    eng, node = build(cls_model, [channel_image(0)])
    nr = eng.run().node_results[node.id]
    assert nr.status == NodeStatus.OK, nr.error                      # 回退之后照样能跑
    assert node._runner.holder.providers == ["CPUExecutionProvider"]
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("CUDAExecutionProvider 无法加载" in m for m in msgs) == 1
    assert any("使用推理后端 CPUExecutionProvider" in m for m in msgs)
    assert "CUDAExecutionProvider" in clean_failed_providers         # 记下来，本进程不再尝试
    caplog.clear()
    eng2, _ = build(cls_model, [channel_image(1)], scale=0.5)         # 另一份配置，重新建会话
    eng2.run()
    assert not [m for m in (r.getMessage() for r in caplog.records) if "无法加载" in m]
    eng.teardown_nodes(); eng2.teardown_nodes()


def test_dl_explicit_provider_unavailable_warns(cls_model, monkeypatch, caplog, clean_failed_providers):
    import logging
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    caplog.set_level(logging.INFO, logger="cvflow.dl")
    eng, node = build(cls_model, [channel_image(0)], provider="cuda")
    assert eng.run().node_results[node.id].status == NodeStatus.OK
    assert any("请求的推理后端 cuda 不可用" in r.getMessage() for r in caplog.records)
    eng.teardown_nodes()

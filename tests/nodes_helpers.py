"""黑盒测试公共工具：合成图像、节点运行器、微型 ONNX 模型。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cvflow.core import DataType, Graph, Image, Node, Port
from cvflow.core.engine import Engine

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 合成图像（真值可解析计算）
def squares_bgr(size: int = 200) -> Image:
    """黑底三个白方块，左上角 (20,80) (80,80) (140,80)，边长 41 像素（含端点）→ 面积 1681，质心 x=40/100/160, y=100。"""
    data = np.zeros((size, size), dtype=np.uint8)
    for x in (20, 80, 140):
        cv2.rectangle(data, (x, 80), (x + 40, 120), 255, -1)
    return Image(cv2.cvtColor(data, cv2.COLOR_GRAY2BGR))


def squares_gray(size: int = 200) -> Image:
    return Image(cv2.cvtColor(squares_bgr(size).data, cv2.COLOR_BGR2GRAY))


def step_gray(width: int = 300, height: int = 100, x0: int = 50, x1: int = 150, lo: int = 30, hi: int = 220) -> Image:
    """背景 lo，x0..x1-1 列为 hi：理想竖直边缘在 x0-0.5 和 x1-0.5。"""
    d = np.full((height, width), lo, np.uint8)
    d[:, x0:x1] = hi
    return Image(d)


def disk_gray(size: int = 200, cx: int = 100, cy: int = 90, r: int = 40, val: int = 255) -> Image:
    d = np.zeros((size, size), np.uint8)
    cv2.circle(d, (cx, cy), r, val, -1)
    return Image(d)


def ramp_gray(width: int = 256, height: int = 8) -> Image:
    """每列灰度 = 列号：第 i 列像素值 i。"""
    return Image(np.tile(np.arange(width, dtype=np.uint8), (height, 1)))


def unique_gray(h: int = 4, w: int = 6) -> Image:
    """每个像素唯一值 = 行*w+列，用于验证旋转/翻转的像素映射。"""
    return Image(np.arange(h * w, dtype=np.uint8).reshape(h, w))


def noisy_gray(size: int = 128, seed: int = 0, base: int = 128, sigma: float = 25) -> Image:
    rng = np.random.default_rng(seed)
    d = np.clip(rng.normal(base, sigma, (size, size)), 0, 255).astype(np.uint8)
    return Image(d)


def qr_bgr(text: str, scale: int = 8, border: int = 4) -> Image:
    enc = cv2.QRCodeEncoder.create()
    code = enc.encode(text)
    code = cv2.resize(code, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    code = cv2.copyMakeBorder(code, border * scale, border * scale, border * scale, border * scale, cv2.BORDER_CONSTANT, value=255)
    return Image(cv2.cvtColor(code, cv2.COLOR_GRAY2BGR))


# ---------------------------------------------------------------- 运行一个节点
class _Feeder(Node):
    type_id = "test.feeder"
    category = "Test"
    label = "Feeder"

    def __init__(self, values: dict):
        self.__class__.outputs = [Port(k, DataType.ANY) for k in values]
        super().__init__()
        self._vals = values

    def process(self, ctx, inputs):
        return dict(self._vals)


def run_node(reg, type_id: str, inputs: dict | None = None, values: dict | None = None, graph_hook=None):
    """按黑盒方式运行一个节点：给输入端口喂值、设参数，返回 (NodeResult, RunResult, node)。不做任何断言。"""
    inputs = inputs or {}
    g = Graph("bb")
    node = reg.create(type_id, values=values or {})
    g.add_node(node)
    if inputs:
        feeder = _Feeder(inputs)
        g.add_node(feeder)
        for k in inputs:
            if node.get_input_port(k) is not None:
                g.add_link(feeder.id, k, node.id, k)
    if graph_hook:
        graph_hook(g, node)
    eng = Engine(g)
    r = eng.run()
    return r.node_results[node.id], r, node


def run_ok(reg, type_id, inputs=None, values=None):
    nr, r, node = run_node(reg, type_id, inputs, values)
    assert nr.status.value in ("ok", "ng"), f"{type_id}: {nr.status.value} {nr.error}"
    return nr


# ---------------------------------------------------------------- 微型 ONNX 模型
def make_identity_model(path: Path, shape=(1, 3, 8, 8), dynamic_batch: bool = False) -> Path:
    import onnx
    from onnx import TensorProto, helper
    shape = (["b"] + list(shape[1:])) if dynamic_batch else list(shape)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "ident", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def make_channel_mean_classifier(path: Path, size: int = 8, dynamic_batch: bool = False) -> Path:
    """logits[c] = 通道 c 的均值 → 哪个通道最亮就是哪个类。"""
    import onnx
    from onnx import TensorProto, helper
    b = "b" if dynamic_batch else 1
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [b, 3, size, size])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [b, 3])
    node = helper.make_node("ReduceMean", ["x"], ["logits"], axes=[2, 3], keepdims=0)
    g = helper.make_graph([node], "cls", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def make_constant_detector(path: Path, boxes: list[tuple[float, float, float, float, int, float]], n_cls: int = 80) -> Path:
    """输出固定的 YOLOv8 布局 [1, 4+n_cls, N] 张量；boxes = [(cx, cy, w, h, class_id, score), ...]。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    arr = np.zeros((1, 4 + n_cls, len(boxes)), dtype=np.float32)
    for i, (cx, cy, w, h, cid, score) in enumerate(boxes):
        arr[0, 0:4, i] = (cx, cy, w, h)
        if n_cls:                     # n_cls 为 0 时只有 4 个坐标，用来构造"属性不足"的模型
            arr[0, 4 + cid, i] = score
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(arr.shape))
    const = helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(arr, "pred"))
    g = helper.make_graph([const], "det", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path


def channel_image(channel: int, size: int = 8) -> Image:
    """只有指定通道是亮的 BGR 图，用于区分批次里每一路的来源。"""
    d = np.zeros((size, size, 3), np.uint8)
    d[:, :, channel] = 255
    return Image(d)


def make_shaped_model(path: Path, shape) -> Path:
    """输入形状任意的恒等模型。shape 的元素可以是整数（固定）或字符串（动态）。"""
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    g = helper.make_graph([helper.make_node("Identity", ["images"], ["y"])], "shaped", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return path

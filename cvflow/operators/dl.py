"""Deep-learning and scripting nodes.

* OnnxInference / OnnxClassifier / OnnxDetector run any exported model with onnxruntime.
* PythonScript executes user code (PyTorch, an RL policy, anything importable) inside the flow.

For a reusable algorithm write a plugin file instead (see examples/plugins).
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

from ..core import paths
from ..core.batching import BatchExecutor
from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType, Image, Overlay, Rect
from ._util import as_bgr, parse_floats, require_image

_COLOR = "#ad1457"


def _load_labels(path: str) -> list[str]:
    path = paths.resolve(path)
    if not path or not os.path.isfile(path):
        return []
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]


# --------------------------------------------------------------------------- 共享会话与批处理
class _Runner:
    """一个模型对应的共享推理会话加批处理执行器，按键共享、引用计数释放。"""

    def __init__(self, session, input_name: str, dynamic_batch: bool, max_batch: int, wait_s: float, name: str) -> None:
        self.session = session
        self.input_name = input_name
        self.dynamic_batch = dynamic_batch
        self.refs = 0
        # 模型的批次维是固定的时候只能一张一张推，批上限强制为 1
        self.executor = BatchExecutor(self._run_batch, max_batch=max_batch if dynamic_batch else 1,
                                      wait_s=wait_s, name=name)

    def _run_batch(self, blobs: list):
        x = blobs[0] if len(blobs) == 1 else np.concatenate(blobs, axis=0)
        outs = self.session.run(None, {self.input_name: x})
        n = len(blobs)
        return [(_slice_outputs(outs, i, n), n) for i in range(n)]

    def close(self) -> None:
        self.executor.close()
        self.session = None


def _slice_outputs(outs, i: int, n: int) -> list:
    """把批量输出按第一维拆给第 i 个输入，保留批次维，使后处理与单张推理完全一致。

    第一维不等于批大小的输出（例如与批无关的常量）原样给每一个调用者。
    """
    row = []
    for o in outs:
        a = np.asarray(o)
        row.append(a[i:i + 1] if a.ndim >= 1 and a.shape[0] == n else a)
    return row


_RUNNERS: dict[tuple, _Runner] = {}
_RUNNERS_LOCK = threading.Lock()


def _acquire_runner(key: tuple, factory) -> _Runner:
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(key)
        if runner is None:
            runner = factory()
            _RUNNERS[key] = runner
        runner.refs += 1
        return runner


def _release_runner(key: tuple) -> None:
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(key)
        if runner is None:
            return
        runner.refs -= 1
        if runner.refs <= 0:
            _RUNNERS.pop(key, None)
            runner.close()


def active_runners() -> int:
    """当前存活的共享会话数量，供测试与诊断使用。"""
    with _RUNNERS_LOCK:
        return len(_RUNNERS)


MAX_INPUTS = 8


def _input_port_name(i: int) -> str:
    return "image" if i == 0 else f"image{i + 1}"


def _output_port_name(base: str, i: int) -> str:
    return base if i == 0 else f"{base}{i + 1}"


class _OnnxBase(Node):
    """共享的模型加载、预处理与批量推理。

    节点可以设置多个图像输入。一次运行里，所有已连接输入的图像会被拼成一个批次做一次推理，
    再按来源把输出拆回各自的端口。``wait_ms`` 大于 0 时还会在该窗口内等待其它线程（例如别的流程）
    提交的图像一起合批，窗口到期就按当前已有的数量执行。
    """

    #: 每个输入各自对应的输出端口，子类覆盖
    per_input_outputs: list[tuple[str, DataType]] = []

    params = [Param("model_path", "", "file", filter="ONNX models (*.onnx)"),
              Param("input_count", 1, "int", min=1, max=MAX_INPUTS,
                    description="图像输入的个数。多个输入会被合并成一个批次，一次推理完成"),
              Param("max_batch", 8, "int", min=1, max=64,
                    description="一个批次最多几张图。模型的批次维必须是动态的才能大于 1"),
              Param("wait_ms", 0.0, "float", min=0, max=1000,
                    description="等待其它线程的图像一起合批的时间，0 表示不等待。窗口到期就按当前数量执行"),
              Param("width", 224, "int", min=1, max=8192), Param("height", 224, "int", min=1, max=8192),
              Param("letterbox", False, "bool", description="Keep aspect ratio, pad to size"),
              Param("color", "rgb", "enum", choices=["rgb", "bgr", "gray"]),
              Param("scale", 1 / 255.0, "float", min=0, max=1000, description="Multiply pixels by this"),
              Param("mean", "0,0,0", "string", description="Per-channel mean (after scaling)"),
              Param("std", "1,1,1", "string", description="Per-channel std (after scaling)"),
              Param("layout", "NCHW", "enum", choices=["NCHW", "NHWC"]),
              Param("provider", "auto", "enum", choices=["auto", "cpu", "cuda", "tensorrt"], advanced=True),
              Param("batch_group", "", "string", advanced=True,
                    description="合批分组名。留空按模型与预处理自动分组；填不同的名字可以让两个节点互不合批"),
              Param("overlay_input", 1, "int", min=1, max=MAX_INPUTS, advanced=True,
                    description="图像窗口上画哪一个输入的结果"),
              Param("timeout_s", 30.0, "float", min=0.1, max=600, advanced=True,
                    description="等待批处理结果的超时时间")]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._runner: _Runner | None = None
        self._runner_key: tuple | None = None
        self._rebuild_ports()

    # ---- 动态端口 ----
    @classmethod
    def build_ports(cls, count: int) -> tuple[list[Port], list[Port]]:
        count = max(1, min(MAX_INPUTS, int(count)))
        ins = [Port(_input_port_name(i), DataType.IMAGE, optional=i > 0,
                    description="第一个输入" if i == 0 else f"第 {i + 1} 个输入，未连接时该路输出为空")
               for i in range(count)]
        outs = [Port(_output_port_name(base, i), dtype) for i in range(count) for base, dtype in cls.per_input_outputs]
        outs += [Port("infer_ms", DataType.FLOAT, description="本次推理耗时，整批共用"),
                 Port("batch_size", DataType.INT, description="本次推理实际的批大小")]
        return ins, outs

    def _rebuild_ports(self) -> None:
        self.inputs, self.outputs = self.build_ports(self.get("input_count", 1))

    def on_param_changed(self, name: str, value) -> None:
        if name == "input_count":
            self._rebuild_ports()

    @property
    def input_count(self) -> int:
        return len(self.inputs)

    # ---- 会话与批处理执行器 ----
    def _make_key(self, path: str, mtime: float) -> tuple:
        pre = (int(self.get("width")), int(self.get("height")), bool(self.get("letterbox")), self.get("color"),
               float(self.get("scale")), self.get("mean"), self.get("std"), self.get("layout"))
        return (path, mtime, self.get("provider"), pre, int(self.get("max_batch")),
                round(float(self.get("wait_ms")), 3), str(self.get("batch_group")).strip())

    def _ensure_runner(self) -> _Runner:
        path = paths.resolve(self.get("model_path"))
        if not path:
            raise NodeError("未设置模型文件")
        if not os.path.isfile(path):
            raise NodeError(f"模型文件不存在：{path}")
        key = self._make_key(path, os.path.getmtime(path))
        if self._runner_key == key and self._runner is not None:
            return self._runner
        self._release_runner()
        self._runner = _acquire_runner(key, lambda: self._create_runner(path, key))
        self._runner_key = key
        return self._runner

    def _create_runner(self, path: str, key: tuple) -> _Runner:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise NodeError("未安装 onnxruntime") from e
        avail = ort.get_available_providers()
        prov = self.get("provider")
        if prov == "auto":
            providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        else:
            want = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "tensorrt": "TensorrtExecutionProvider"}[prov]
            providers = [want] if want in avail else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        session = ort.InferenceSession(path, so, providers=providers)
        inp = session.get_inputs()[0]
        shape = list(inp.shape or [])
        dynamic = bool(shape) and not isinstance(shape[0], int)
        return _Runner(session, inp.name, dynamic, int(self.get("max_batch")),
                       float(self.get("wait_ms")) / 1000.0, name=os.path.basename(path))

    def _release_runner(self) -> None:
        if self._runner_key is not None:
            _release_runner(self._runner_key)
        self._runner, self._runner_key = None, None

    def setup(self, ctx=None):
        if self.get("model_path"):
            self._ensure_runner()

    def teardown(self):
        self._release_runner()

    @property
    def supports_batching(self) -> bool:
        return self._runner is not None and self._runner.dynamic_batch

    # ---- 预处理 ----
    def _preprocess(self, img: Image):
        w, h = int(self.get("width")), int(self.get("height"))
        color = self.get("color")
        data = img.data
        if color == "gray":
            data = data if img.is_gray else cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        else:
            data = as_bgr(img)
            if color == "rgb":
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        meta = {"scale": 1.0, "pad": (0, 0), "orig": (img.width, img.height)}
        if self.get("letterbox"):
            r = min(w / img.width, h / img.height)
            nw, nh = int(round(img.width * r)), int(round(img.height * r))
            resized = cv2.resize(data, (nw, nh))
            canvas = np.full((h, w) + (() if resized.ndim == 2 else (resized.shape[2],)), 114, dtype=np.uint8)
            dx, dy = (w - nw) // 2, (h - nh) // 2
            canvas[dy:dy + nh, dx:dx + nw] = resized
            data, meta["scale"], meta["pad"] = canvas, r, (dx, dy)
        else:
            data = cv2.resize(data, (w, h))
            meta["scale"] = (w / img.width, h / img.height)
        x = data.astype(np.float32) * float(self.get("scale"))
        if x.ndim == 2:
            x = x[:, :, None]
        c = x.shape[2]
        mean = np.array(parse_floats(self.get("mean"), c, 0.0), dtype=np.float32)
        std = np.array(parse_floats(self.get("std"), c, 1.0), dtype=np.float32)
        x = (x - mean) / std
        if self.get("layout") == "NCHW":
            x = x.transpose(2, 0, 1)
        return np.ascontiguousarray(x[None]), meta

    # ---- 批量推理 ----
    def _gather_images(self, inputs: dict) -> list:
        imgs = []
        for i in range(self.input_count):
            v = inputs.get(_input_port_name(i))
            if v is None:
                imgs.append(None)
            else:
                imgs.append(require_image(v, _input_port_name(i)))
        if all(im is None for im in imgs):
            raise NodeError("没有任何已连接的输入图像")
        return imgs

    def _infer_batch(self, images: list):
        """对所有非空图像做一次批量推理。

        返回 ``(每路输出字典, 每路 meta 字典, 耗时毫秒, 实际批大小)``，键是输入序号。
        """
        runner = self._ensure_runner()
        order = [i for i, im in enumerate(images) if im is not None]
        blobs, metas = [], {}
        for i in order:
            blob, meta = self._preprocess(images[i])
            blobs.append(blob)
            metas[i] = meta
        t0 = time.perf_counter()
        results = runner.executor.submit(blobs, timeout=float(self.get("timeout_s")))
        ms = (time.perf_counter() - t0) * 1000
        outs = {i: results[k][0] for k, i in enumerate(order)}
        batch = max((results[k][1] for k in range(len(order))), default=0)
        return outs, metas, ms, batch

    def _empty_result(self) -> dict:
        """未连接的输入对应的输出，全部为 None，保持端口语义稳定。"""
        return {base: None for base, _ in self.per_input_outputs}

    def process(self, ctx, inputs):
        images = self._gather_images(inputs)
        outs, metas, ms, batch = self._infer_batch(images)
        draw_on = int(self.get("overlay_input")) - 1
        result: dict = {"infer_ms": ms, "batch_size": batch}
        for i in range(self.input_count):
            one = (self.decode(ctx, outs[i], metas[i], images[i], draw=(i == draw_on))
                   if i in outs else self._empty_result())
            for base, _ in self.per_input_outputs:
                result[_output_port_name(base, i)] = one.get(base)
        return result

    def decode(self, ctx, outs: list, meta: dict, img: Image, draw: bool) -> dict:
        """把一路的原始输出转成该路的输出字典，子类实现。"""
        raise NotImplementedError


@register
class OnnxInference(_OnnxBase):
    type_id = "dl.onnx"
    category = "Deep Learning"
    label = "ONNX Inference"
    description = "Run any ONNX model; outputs the raw tensors for downstream post-processing (e.g. a script node)."
    color = _COLOR
    per_input_outputs = [("outputs", DataType.LIST), ("output0", DataType.TENSOR)]
    inputs, outputs = Node.inputs, Node.outputs      # 占位，下面用 build_ports 覆盖

    def decode(self, ctx, outs, meta, img, draw):
        return {"outputs": outs, "output0": outs[0]}


@register
class OnnxClassifier(_OnnxBase):
    type_id = "dl.onnx_classifier"
    category = "Deep Learning"
    label = "ONNX Classifier"
    description = "Image classification: softmax over the first output, labels from a text file."
    color = _COLOR
    per_input_outputs = [("class_id", DataType.INT), ("label", DataType.STRING),
                         ("score", DataType.FLOAT), ("probs", DataType.LIST)]
    params = _OnnxBase.params + [Param("labels_path", "", "file", filter="Text (*.txt)"),
                                 Param("softmax", True, "bool")]

    def decode(self, ctx, outs, meta, img, draw):
        logits = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        if self.get("softmax"):
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
        else:
            probs = logits
        cid = int(probs.argmax())
        labels = _load_labels(self.get("labels_path"))
        label = labels[cid] if cid < len(labels) else str(cid)
        if draw:
            ctx.add_overlay(Overlay.text(8, 20, f"{label} {float(probs[cid]):.3f}", "#ffff00"))
        return {"class_id": cid, "label": label, "score": float(probs[cid]), "probs": probs.tolist()}


@register
class OnnxDetector(_OnnxBase):
    type_id = "dl.onnx_detector"
    category = "Deep Learning"
    label = "ONNX Detector (YOLO)"
    description = "YOLOv5/v8/v11-style detector: decodes [1,84,N] or [1,N,85] outputs, applies NMS."
    color = _COLOR
    per_input_outputs = [("detections", DataType.LIST), ("count", DataType.INT),
                         ("boxes", DataType.LIST), ("best_label", DataType.STRING)]
    params = [p if p.name not in ("width", "height", "letterbox") else
              Param(p.name, {"width": 640, "height": 640, "letterbox": True}[p.name], p.kind, min=p.min, max=p.max,
                    description=p.description, advanced=p.advanced)
              for p in _OnnxBase.params] + [
        Param("labels_path", "", "file", filter="Text (*.txt)"),
        Param("conf", 0.25, "float", min=0, max=1, step=0.01), Param("iou", 0.45, "float", min=0, max=1, step=0.01)]

    def decode(self, ctx, outs, meta, img, draw):
        pred = np.asarray(outs[0], dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        # 属性维（4 个框坐标 + 类别分数）至少 5 个；v8 布局为 [84, N]，v5 为 [N, 85]
        if pred.shape[1] < 5 or (pred.shape[0] >= 5 and pred.shape[0] < pred.shape[1]):
            pred = pred.T
        if pred.shape[1] > 5 and pred.shape[1] - 5 >= 1 and self._looks_v5(pred):
            obj = pred[:, 4:5]
            cls_scores = pred[:, 5:] * obj
        else:
            cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores.max(1)
        keep = scores >= float(self.get("conf"))
        boxes_xywh, scores, cls_ids = pred[keep, :4], scores[keep], cls_ids[keep]
        if self.get("letterbox"):
            r, (dx, dy) = meta["scale"], meta["pad"]
            sx = sy = 1.0 / r
            ox, oy = dx, dy
        else:
            sx, sy = 1.0 / meta["scale"][0], 1.0 / meta["scale"][1]
            ox = oy = 0
        rects = []
        for (cx, cy, w, h) in boxes_xywh:
            x0 = (cx - w / 2 - ox) * sx
            y0 = (cy - h / 2 - oy) * sy
            rects.append([int(x0), int(y0), int(w * sx), int(h * sy)])
        idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), float(self.get("conf")), float(self.get("iou"))) if rects else []
        idx = np.asarray(idx).reshape(-1)
        labels = _load_labels(self.get("labels_path"))
        dets = []
        for i in idx:
            x, y, w, h = rects[i]
            cid = int(cls_ids[i])
            name = labels[cid] if cid < len(labels) else str(cid)
            dets.append({"x": x, "y": y, "w": w, "h": h, "score": float(scores[i]), "class_id": cid, "label": name})
            if draw:
                ctx.add_overlay(Overlay.rect(Rect(x, y, w, h), "#00ff00", f"{name} {scores[i]:.2f}"))
        dets.sort(key=lambda d: -d["score"])
        return {"detections": dets, "count": len(dets),
                "boxes": [Rect(d["x"], d["y"], d["w"], d["h"]) for d in dets],
                "best_label": dets[0]["label"] if dets else ""}

    @staticmethod
    def _looks_v5(pred: np.ndarray) -> bool:
        # v5: column 4 is objectness in [0,1] and there are 85 columns for COCO; v8 has 84 (no objectness).
        return pred.shape[1] in (85, 6) or (pred.shape[1] > 6 and float(pred[:, 4].max()) <= 1.0 and float(pred[:, 5:].max()) <= 1.0 and pred.shape[1] % 2 == 1)


for _cls in (OnnxInference, OnnxClassifier, OnnxDetector):
    _cls.inputs, _cls.outputs = _cls.build_ports(1)      # 类级端口用于节点库展示，实例按 input_count 重建


_SCRIPT_TEMPLATE = '''"""自定义节点代码。可用对象：np, cv2, Image, Overlay, Rect, Point, Line, Circle。

def setup(state): 可选，流程启动时执行一次。
def process(ctx, inputs, params, state) -> dict: 每张图执行一次。
   inputs: {"image": Image|None, "in1".."in4": 任意}
   返回 {"image": Image, "out1".."out4": 任意}
"""
def process(ctx, inputs, params, state):
    img = inputs["image"]
    if img is None:
        return {}
    gray = img.data if img.is_gray else cv2.cvtColor(img.data, cv2.COLOR_BGR2GRAY)
    state["runs"] = state.get("runs", 0) + 1
    ctx.add_overlay(Overlay.text(8, 20, f"script run {state['runs']}", "#ffff00"))
    return {"image": img.derive(gray), "out1": float(gray.mean()), "out2": state["runs"]}
'''


@register
class PythonScript(Node):
    type_id = "script.python"
    category = "Script"
    label = "Python Script"
    description = "Run user Python code as a node (the escape hatch for custom DL/RL logic)."
    color = "#ef6c00"
    inputs = [Port("image", DataType.IMAGE, optional=True)] + [Port(f"in{i}", DataType.ANY, optional=True) for i in range(1, 5)]
    outputs = [Port("image", DataType.IMAGE)] + [Port(f"out{i}", DataType.ANY) for i in range(1, 5)]
    params = [Param("code", _SCRIPT_TEMPLATE, "code")]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ns: dict | None = None
        self._compiled_src = None

    def _namespace(self):
        src = self.get("code")
        if self._ns is None or self._compiled_src != src:
            from ..core.types import Circle, Line, Point
            ns = {"np": np, "cv2": cv2, "Image": Image, "Overlay": Overlay, "Rect": Rect,
                  "Point": Point, "Line": Line, "Circle": Circle, "NodeError": NodeError, "__name__": f"script_{self.id}"}
            exec(compile(src, f"<script {self.name}>", "exec"), ns)
            if "process" not in ns or not callable(ns["process"]):
                raise NodeError("脚本必须定义 process(ctx, inputs, params, state)")
            self._ns, self._compiled_src = ns, src
            if callable(ns.get("setup")):
                ns["setup"](self.state)
        return self._ns

    def on_param_changed(self, name, value):
        if name == "code":
            self._ns = None

    def setup(self, ctx=None):
        self._namespace()

    def process(self, ctx, inputs):
        ns = self._namespace()
        out = ns["process"](ctx, inputs, dict(self.values), self.state)
        return out or {}

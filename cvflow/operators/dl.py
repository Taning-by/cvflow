"""Deep-learning and scripting nodes.

* OnnxInference / OnnxClassifier / OnnxDetector run any exported model with onnxruntime.
* PythonScript executes user code (PyTorch, an RL policy, anything importable) inside the flow.

For a reusable algorithm write a plugin file instead (see examples/plugins).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from ..core import paths
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


class _OnnxBase(Node):
    """Shared model loading and preprocessing."""

    params = [Param("model_path", "", "file", filter="ONNX models (*.onnx)"),
              Param("width", 224, "int", min=1, max=8192), Param("height", 224, "int", min=1, max=8192),
              Param("letterbox", False, "bool", description="Keep aspect ratio, pad to size"),
              Param("color", "rgb", "enum", choices=["rgb", "bgr", "gray"]),
              Param("scale", 1 / 255.0, "float", min=0, max=1000, description="Multiply pixels by this"),
              Param("mean", "0,0,0", "string", description="Per-channel mean (after scaling)"),
              Param("std", "1,1,1", "string", description="Per-channel std (after scaling)"),
              Param("layout", "NCHW", "enum", choices=["NCHW", "NHWC"]),
              Param("provider", "auto", "enum", choices=["auto", "cpu", "cuda", "tensorrt"], advanced=True)]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._session = None
        self._loaded_key = None
        self._input_name = ""

    def _ensure_session(self):
        path = paths.resolve(self.get("model_path"))
        if not path:
            raise NodeError("no model_path set")
        if not os.path.isfile(path):
            raise NodeError(f"model not found: {path}")
        key = (path, os.path.getmtime(path), self.get("provider"))
        if self._loaded_key == key and self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise NodeError("onnxruntime is not installed") from e
        avail = ort.get_available_providers()
        prov = self.get("provider")
        if prov == "auto":
            providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        else:
            want = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "tensorrt": "TensorrtExecutionProvider"}[prov]
            providers = [want] if want in avail else ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._session = ort.InferenceSession(path, so, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._loaded_key = key
        return self._session

    def setup(self, ctx=None):
        if self.get("model_path"):
            self._ensure_session()

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

    def _infer(self, img: Image):
        sess = self._ensure_session()
        blob, meta = self._preprocess(img)
        t0 = time.perf_counter()
        outs = sess.run(None, {self._input_name: blob})
        return outs, meta, (time.perf_counter() - t0) * 1000


@register
class OnnxInference(_OnnxBase):
    type_id = "dl.onnx"
    category = "Deep Learning"
    label = "ONNX Inference"
    description = "Run any ONNX model; outputs the raw tensors for downstream post-processing (e.g. a script node)."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("outputs", DataType.LIST), Port("output0", DataType.TENSOR), Port("infer_ms", DataType.FLOAT)]

    def process(self, ctx, inputs):
        outs, _, ms = self._infer(require_image(inputs["image"]))
        return {"outputs": outs, "output0": outs[0], "infer_ms": ms}


@register
class OnnxClassifier(_OnnxBase):
    type_id = "dl.onnx_classifier"
    category = "Deep Learning"
    label = "ONNX Classifier"
    description = "Image classification: softmax over the first output, labels from a text file."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("class_id", DataType.INT), Port("label", DataType.STRING), Port("score", DataType.FLOAT),
               Port("probs", DataType.LIST), Port("infer_ms", DataType.FLOAT)]
    params = _OnnxBase.params + [Param("labels_path", "", "file", filter="Text (*.txt)"),
                                 Param("softmax", True, "bool")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        outs, _, ms = self._infer(img)
        logits = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        if self.get("softmax"):
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
        else:
            probs = logits
        cid = int(probs.argmax())
        labels = _load_labels(self.get("labels_path"))
        label = labels[cid] if cid < len(labels) else str(cid)
        ctx.add_overlay(Overlay.text(8, 20, f"{label} {float(probs[cid]):.3f}", "#ffff00"))
        return {"class_id": cid, "label": label, "score": float(probs[cid]), "probs": probs.tolist(), "infer_ms": ms}


@register
class OnnxDetector(_OnnxBase):
    type_id = "dl.onnx_detector"
    category = "Deep Learning"
    label = "ONNX Detector (YOLO)"
    description = "YOLOv5/v8/v11-style detector: decodes [1,84,N] or [1,N,85] outputs, applies NMS."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("detections", DataType.LIST), Port("count", DataType.INT), Port("boxes", DataType.LIST),
               Port("best_label", DataType.STRING), Port("infer_ms", DataType.FLOAT)]
    params = [p if p.name not in ("width", "height", "letterbox") else
              Param(p.name, {"width": 640, "height": 640, "letterbox": True}[p.name], p.kind, min=p.min, max=p.max)
              for p in _OnnxBase.params] + [
        Param("labels_path", "", "file", filter="Text (*.txt)"),
        Param("conf", 0.25, "float", min=0, max=1, step=0.01), Param("iou", 0.45, "float", min=0, max=1, step=0.01)]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        outs, meta, ms = self._infer(img)
        pred = np.asarray(outs[0], dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:          # [84, N] (v8 layout) -> [N, 84]
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
            ctx.add_overlay(Overlay.rect(Rect(x, y, w, h), "#00ff00", f"{name} {scores[i]:.2f}"))
        dets.sort(key=lambda d: -d["score"])
        return {"detections": dets, "count": len(dets), "boxes": [Rect(d["x"], d["y"], d["w"], d["h"]) for d in dets],
                "best_label": dets[0]["label"] if dets else "", "infer_ms": ms}

    @staticmethod
    def _looks_v5(pred: np.ndarray) -> bool:
        # v5: column 4 is objectness in [0,1] and there are 85 columns for COCO; v8 has 84 (no objectness).
        return pred.shape[1] in (85, 6) or (pred.shape[1] > 6 and float(pred[:, 4].max()) <= 1.0 and float(pred[:, 5:].max()) <= 1.0 and pred.shape[1] % 2 == 1)


_SCRIPT_TEMPLATE = '''"""Custom node code. Available: np, cv2, Image, Overlay, Rect, Point, Line, Circle.

def setup(state): optional, runs once when the flow starts.
def process(ctx, inputs, params, state) -> dict: runs per image.
   inputs: {"image": Image|None, "in1".."in4": any}
   return {"image": Image, "out1".."out4": any}
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
                raise NodeError("script must define process(ctx, inputs, params, state)")
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

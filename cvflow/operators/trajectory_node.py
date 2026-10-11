"""ONNX 轨迹（关键点）节点。

把 train/export_onnx.py 导出的模型接进流程：固定 ROI 裁剪 → 缩放到网络输入 → 推理 → 20 点换回原图像素，
画出折线与序号，输出点列表 / 有效性 / 置信度 / 轨迹长度。

节点库「Deep Learning」分类里是「ONNX 轨迹 (关键点)」。参数里的宽/高/ROI 按导出时的 sidecar json 填
（节点也能从同名 json 自动读）。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from cvflow.core import DataType, Line, NodeError, Overlay, Param, Point, paths, register
from cvflow.core.types import Image
from cvflow.operators._util import as_bgr
from cvflow.operators.dl import _OnnxBase

_COLOR = "#6a1b9a"


@register
class OnnxTrajectory(_OnnxBase):
    type_id = "dl.onnx_trajectory"
    category = "Deep Learning"
    label = "ONNX 轨迹 (关键点)"
    description = "固定 ROI 裁剪后推理，输出 N 个有序轨迹点（原图像素坐标）、辅助点与可执行置信度。"
    color = _COLOR
    per_input_outputs = [("points", DataType.LIST), ("valid", DataType.BOOL), ("score", DataType.FLOAT),
                         ("length", DataType.FLOAT), ("aux", DataType.LIST), ("start", DataType.POINT), ("end", DataType.POINT)]
    inputs, outputs = [], []

    params = [p if p.name not in ("width", "height", "letterbox", "color", "scale") else
              Param(p.name, {"width": 768, "height": 576, "letterbox": False, "color": "rgb", "scale": 1 / 255.0}[p.name],
                    p.kind, min=p.min, max=p.max, choices=p.choices, description=p.description, advanced=p.advanced)
              for p in _OnnxBase.params] + [
        Param("roi", {"x": 288, "y": 256, "w": 768, "h": 576}, "rect", label="裁剪区 ROI",
              description="先裁这个区域再缩放到宽/高送入网络；留空则用整张图。训练时用的是哪个 ROI 就填哪个"),
        Param("num_points", 20, "int", min=2, max=256, label="轨迹点数"),
        Param("conf", 0.5, "float", min=0, max=1, step=0.01, label="可执行阈值",
              description="模型的可执行概率低于它就判 NG（空夹具 / 被遮挡 / 刀已插入）"),
        Param("judge", True, "bool", label="参与判定", description="可执行概率低于阈值时把本次运行判为 NG"),
        Param("draw_index", True, "bool", label="画序号"),
        Param("draw_aux", True, "bool", label="画辅助点"),
    ]

    # ---- 模型 sidecar：导出时写的同名 .json，自动把输入尺寸 / ROI 对上 ----
    def on_param_changed(self, name, value):
        super().on_param_changed(name, value)
        if name == "model_path" and value:
            side = os.path.splitext(paths.resolve(value))[0] + ".json"
            if os.path.isfile(side):
                try:
                    d = json.load(open(side, encoding="utf-8"))
                    for k in ("width", "height", "color", "scale", "mean", "std", "layout"):
                        if k in d:
                            self.values[k] = self.param_def(k).coerce(d[k])
                    if d.get("roi"):
                        x, y, w, h = d["roi"]
                        self.values["roi"] = {"x": float(x), "y": float(y), "w": float(w), "h": float(h), "angle": 0.0}
                    if d.get("num_points"):
                        self.values["num_points"] = int(d["num_points"])
                except Exception:   # sidecar 坏了就按手填的来
                    pass

    # ---- 预处理：先裁 ROI 再缩放 ----
    def _to_canvas(self, img: Image):
        w, h = int(self.get("width")), int(self.get("height"))
        r = self.rect("roi")
        data = as_bgr(img)
        if r is not None and not r.is_empty():
            x0, y0, rw, rh = r.clip(img.width, img.height).as_int()
            if rw <= 0 or rh <= 0:
                raise NodeError(f"裁剪区 ROI 与图像（{img.width}×{img.height}）没有交集，检查「裁剪区 ROI」或相机分辨率")
            data = data[y0:y0 + rh, x0:x0 + rw]
        else:
            x0, y0, rw, rh = 0, 0, img.width, img.height
        if data.shape[1] != w or data.shape[0] != h:
            data = cv2.resize(data, (w, h), interpolation=cv2.INTER_AREA if w < rw else cv2.INTER_LINEAR)
        meta = {"scale": (w / max(1, rw), h / max(1, rh)), "pad": (0, 0), "orig": (img.width, img.height), "roi": (x0, y0, rw, rh)}
        return data, meta

    def _preprocess_key(self) -> tuple:
        return super()._preprocess_key() + (json.dumps(self.get("roi"), sort_keys=True),)

    # ---- 后处理 ----
    def decode(self, ctx, outs, meta, img):
        if not outs:
            raise NodeError("模型没有输出")
        pts = np.asarray(outs[0], dtype=np.float32)
        pts = pts.reshape(-1, pts.shape[-1]) if pts.ndim >= 2 else pts.reshape(-1, 2)
        n = int(self.get("num_points"))
        if pts.shape[0] < n:
            raise NodeError(f"模型输出 {pts.shape[0]} 个点，少于设定的 {n}；检查模型或「轨迹点数」")
        pts = pts[:n, :2]
        aux = np.asarray(outs[1], dtype=np.float32).reshape(-1, 2) if len(outs) > 1 else np.zeros((0, 2), np.float32)
        prob = float(np.asarray(outs[2]).reshape(-1)[0]) if len(outs) > 2 else 1.0
        sx, sy = meta["scale"]
        x0, y0 = meta["roi"][0], meta["roi"][1]
        full = np.empty_like(pts)
        full[:, 0] = pts[:, 0] / sx + x0
        full[:, 1] = pts[:, 1] / sy + y0
        aux_full = np.empty_like(aux)
        if len(aux):
            aux_full[:, 0] = aux[:, 0] / sx + x0
            aux_full[:, 1] = aux[:, 1] / sy + y0
        valid = prob >= float(self.get("conf"))
        length = float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum())
        col = "#ff1744" if valid else "#9e9e9e"
        for i in range(n - 1):
            ctx.add_overlay(Overlay.line(Line(float(full[i, 0]), float(full[i, 1]), float(full[i + 1, 0]), float(full[i + 1, 1])), col))
        ctx.add_overlay(Overlay.points(full, "#76ff03" if valid else "#bdbdbd"))
        if self.get("draw_index"):
            for i, (x, y) in enumerate(full):
                if i in (0, n - 1) or i % 5 == 0:
                    ctx.add_overlay(Overlay.text(float(x) + 4, float(y) - 4, str(i), "#ffffff"))
        if self.get("draw_aux") and len(aux_full):
            names = ["籽尖入", "腔壁入", "籽尖出", "腔壁出"]
            for j, (x, y) in enumerate(aux_full):
                ctx.add_overlay(Overlay.points([(x, y)], "#ffeb3b"))
                ctx.add_overlay(Overlay.text(float(x) + 4, float(y) + 12, names[j] if j < len(names) else str(j), "#ffeb3b"))
        ctx.add_overlay(Overlay.text(x0 + 6, y0 + 18, f"traj {'OK' if valid else 'NG'} p={prob:.2f} L={length:.0f}px", col))
        if self.get("judge"):
            ctx.judge(valid, "轨迹不可执行（空夹具/遮挡/刀已插入）")
        return {"points": [[float(x), float(y)] for x, y in full], "valid": bool(valid), "score": prob,
                "length": length, "aux": [[float(x), float(y)] for x, y in aux_full],
                "start": Point(float(full[0, 0]), float(full[0, 1])), "end": Point(float(full[-1, 0]), float(full[-1, 1]))}


OnnxTrajectory.inputs, OnnxTrajectory.outputs = OnnxTrajectory.build_ports(1)

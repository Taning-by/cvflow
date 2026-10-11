"""ONNX 轨迹（关键点）节点。

把 train/export_onnx.py 导出的模型接进流程：固定 ROI 裁剪 → 缩放到网络输入 → 推理 → 20 点换回原图像素，
画出折线与序号，输出点列表 / 有效性 / 置信度 / 轨迹长度。

节点库「Deep Learning」分类里是「ONNX 轨迹 (关键点)」。参数里的宽/高/ROI 按导出时的 sidecar json 填
（节点也能从同名 json 自动读）。
"""
from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np

from cvflow.core import DataType, Line, NodeError, Overlay, Param, Point, paths, register
from cvflow.core.types import Image
from cvflow.operators._util import as_bgr
from cvflow.operators.dl import _OnnxBase

_COLOR = "#6a1b9a"
log = logging.getLogger("cvflow.dl")


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
              Param(p.name, {"width": 1280, "height": 1024, "letterbox": False, "color": "rgb", "scale": 1 / 255.0}[p.name],
                    p.kind, min=p.min, max=p.max, choices=p.choices, description=p.description, advanced=p.advanced)
              for p in _OnnxBase.params] + [
        Param("roi", {"x": 0, "y": 0, "w": 1280, "h": 1024}, "rect", label="裁剪区 ROI",
              description="先裁这个区域再缩放到宽/高送入网络。选模型后会按同名 .json 自动填好，一般不用动。"
                          "当它从 (0,0) 起、宽高又等于下面的输入宽/高时，裁剪和缩放都是空操作，等于整张原图直接送进网络"),
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
        if name != "model_path" or not value:
            return
        side = os.path.splitext(paths.resolve(value))[0] + ".json"
        if not os.path.isfile(side):
            # 没有 sidecar 时参数仍是上一个模型留下的，尺寸对不上会直接出错误结果，必须说出来
            log.warning("%s 旁边没有同名的 .json，输入尺寸与 ROI 仍沿用上一个模型的，请自行核对是否匹配",
                        os.path.basename(paths.resolve(value)))
            return
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
        except Exception as e:      # sidecar 坏了就按手填的来，但不能默不作声
            log.warning("读取 %s 失败（%s），输入尺寸与 ROI 仍按当前参数", side, e)
            return
        r = self.get("roi")
        log.info("按 %s 配置：输入 %sx%s，ROI %s", os.path.basename(side), self.get("width"), self.get("height"),
                 "整张图（不裁剪）" if self._roi_is_full() else f"x{r['x']:.0f} y{r['y']:.0f} w{r['w']:.0f} h{r['h']:.0f}")

    def _roi_is_full(self) -> bool:
        """ROI 左上角在原点、尺寸又等于网络输入时，裁剪与缩放都是空操作——就是"整张图直接送进去"。"""
        r = self.get("roi") or {}
        return (r.get("x"), r.get("y")) == (0.0, 0.0) and \
            (r.get("w"), r.get("h")) == (float(self.get("width")), float(self.get("height")))

    def _warn_once(self, key: str, msg: str, *args) -> None:
        """同一种情况只提醒一次：这些检查在每帧都会命中，照实打会把日志刷爆。"""
        seen = self.__dict__.setdefault("_warned", set())
        if key not in seen:                      # 两路不同分辨率的相机交替送图时，各自只提醒一次
            seen.add(key)
            log.warning("%s：" + msg, self.name, *args)

    # ---- 预处理：取出要送进网络的区域，再缩放到模型要求的输入尺寸 ----
    def _to_canvas(self, img: Image):
        w, h = int(self.get("width")), int(self.get("height"))
        r = self.rect("roi")
        data = as_bgr(img)
        # ROI 配成"整图"时不做裁剪：画面多大就整张送进去，再统一缩放到输入尺寸。
        # 这样换了分辨率更大的相机也不会被悄悄切掉一角——只是分辨率变了，内容一点不少。
        if r is not None and not r.is_empty() and not self._roi_is_full():
            x0, y0, rw, rh = r.clip(img.width, img.height).as_int()
            if rw <= 0 or rh <= 0:
                raise NodeError(f"裁剪区 ROI 与图像（{img.width}×{img.height}）没有交集，检查「裁剪区 ROI」或相机分辨率")
            if (x0, y0, rw, rh) != r.as_int():
                self._warn_once(f"roi_clip:{img.width}x{img.height}",
                                "裁剪区 ROI %s 超出图像（%d×%d），只取到交集 %d,%d %d×%d，送进网络的画面比配置的少",
                                r.as_int(), img.width, img.height, x0, y0, rw, rh)
            data = data[y0:y0 + rh, x0:x0 + rw]
        else:
            x0, y0, rw, rh = 0, 0, img.width, img.height
            if (rw, rh) != (w, h):
                self._warn_once(f"full_resize:{rw}x{rh}",
                                "图像 %d×%d 与模型输入 %d×%d 不一致，已整图缩放后推理（内容不丢，但与训练时的尺度不同，精度会下降）"
                                + ("；长宽比也变了" if abs(rw / max(1, rh) - w / max(1, h)) > 0.01 else ""),
                                rw, rh, w, h)
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
        # 把实际送进网络的那块区域写出来：看一眼就知道当前是整图推理还是裁了 ROI，不用去翻参数
        rx, ry, rw, rh = meta["roi"]
        src = "整图" if (rx, ry, rw, rh) == (0, 0, img.width, img.height) else f"ROI {rx},{ry} {rw}x{rh}"
        ctx.add_overlay(Overlay.text(x0 + 6, y0 + 18,
                                     f"traj {'OK' if valid else 'NG'} p={prob:.2f} L={length:.0f}px  [{src}]", col))
        if self.get("judge"):
            ctx.judge(valid, "轨迹不可执行（空夹具/遮挡/刀已插入）")
        return {"points": [[float(x), float(y)] for x, y in full], "valid": bool(valid), "score": prob,
                "length": length, "aux": [[float(x), float(y)] for x, y in aux_full],
                "start": Point(float(full[0, 0]), float(full[0, 1])), "end": Point(float(full[-1, 0]), float(full[-1, 1]))}


OnnxTrajectory.inputs, OnnxTrajectory.outputs = OnnxTrajectory.build_ports(1)

"""ONNX 轨迹（关键点）节点。

把 train/export_onnx.py 导出的模型接进流程：上游裁好的固定 ROI 小图 → 缩放到网络输入 → 推理 →
20 点换回原始帧像素，画出折线与各点，输出点列表 / 有效性 / 置信度 / 轨迹长度。

本节点**不自己裁剪**。裁剪只在一个地方做：采集节点（相机 / 图像文件 / 图像文件夹）的「固定 ROI 裁剪」，
或者流程中间的「裁剪 ROI」节点。送到这里的图应该已经是模型输入尺寸；尺寸对不上会整图缩放并每次都提醒。

坐标约定（见 ``Image.origin``）：裁剪位置记在图像的 meta 里，输出的点据此换算回**原始帧**坐标——
PLC / 执行机构拿到的就是原始帧坐标；叠加层则画在送进来的小图上，面板上一眼能看出轨迹落得对不对。

节点库「Deep Learning」分类里是「ONNX 轨迹 (关键点)」。参数里的宽/高按导出时的 sidecar json 填
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
    description = ("输入上游裁好的固定 ROI 图，推理后输出 N 个有序轨迹点（原始帧像素坐标）、辅助点与可执行置信度。"
                   "裁剪在采集节点或「裁剪 ROI」节点上做，本节点不再裁。")
    color = _COLOR
    per_input_outputs = [("points", DataType.LIST), ("valid", DataType.BOOL), ("score", DataType.FLOAT),
                         ("length", DataType.FLOAT), ("aux", DataType.LIST), ("start", DataType.POINT), ("end", DataType.POINT)]
    inputs, outputs = [], []

    params = [p if p.name not in ("width", "height", "letterbox", "color", "scale") else
              Param(p.name, {"width": 1280, "height": 1024, "letterbox": False, "color": "rgb", "scale": 1 / 255.0}[p.name],
                    p.kind, min=p.min, max=p.max, choices=p.choices, description=p.description, advanced=p.advanced)
              for p in _OnnxBase.params] + [
        Param("num_points", 20, "int", min=2, max=256, label="轨迹点数"),
        Param("conf", 0.5, "float", min=0, max=1, step=0.01, label="可执行阈值",
              description="模型的可执行概率低于它就判 NG（空夹具 / 被遮挡 / 刀已插入）"),
        Param("judge", True, "bool", label="参与判定", description="可执行概率低于阈值时把本次运行判为 NG"),
        Param("draw_aux", True, "bool", label="画辅助点", description="把籽尖/腔壁四个辅助点也标出来（黄色）"),
    ]

    # ---- 模型 sidecar：导出时写的同名 .json，自动把输入尺寸对上 ----
    def on_param_changed(self, name, value):
        super().on_param_changed(name, value)
        if name != "model_path" or not value:
            return
        side = os.path.splitext(paths.resolve(value))[0] + ".json"
        if not os.path.isfile(side):
            # 没有 sidecar 时参数仍是上一个模型留下的，尺寸对不上会直接出错误结果，必须说出来
            log.warning("%s 旁边没有同名的 .json，输入尺寸仍沿用上一个模型的，请自行核对是否匹配",
                        os.path.basename(paths.resolve(value)))
            return
        try:
            d = json.load(open(side, encoding="utf-8"))
            for k in ("width", "height", "color", "scale", "mean", "std", "layout"):
                if k in d:
                    self.values[k] = self.param_def(k).coerce(d[k])
            if d.get("num_points"):
                self.values["num_points"] = int(d["num_points"])
        except Exception as e:      # sidecar 坏了就按手填的来，但不能默不作声
            log.warning("读取 %s 失败（%s），输入尺寸仍按当前参数", side, e)
            return
        w, h = self.get("width"), self.get("height")
        log.info("按 %s 配置：输入 %sx%s", os.path.basename(side), w, h)
        # 裁剪不在本节点做：sidecar 里的 ROI 只用来告诉人该在采集节点 /「裁剪 ROI」节点上填什么
        st = d.get("station_rois") or {}
        if st:
            log.info("这个模型是按工位固定 ROI 训的：%s。请在采集节点上开「固定 ROI 裁剪」（或接「裁剪 ROI」节点），"
                     "ROI 宽/高填 %s×%s，位置按工位填",
                     "  ".join(f"{c}:({v[0]},{v[1]})" for c, v in sorted(st.items())), w, h)
        elif d.get("roi"):
            x, y, rw, rh = d["roi"]
            if (int(x), int(y)) != (0, 0) or (int(rw), int(rh)) != (int(w), int(h)):
                log.info("这个模型训练时裁的 ROI 是 x%s y%s %s×%s：请在采集节点上开「固定 ROI 裁剪」"
                         "（或接「裁剪 ROI」节点），宽/高填 %s×%s，位置拖到 (%s, %s)", x, y, rw, rh, w, h, x, y)

    # ---- 预处理：送进来的图直接缩放到模型要求的输入尺寸 ----
    def _to_canvas(self, img: Image):
        w, h = int(self.get("width")), int(self.get("height"))
        data = as_bgr(img)
        warn = ""
        if (img.width, img.height) != (w, h):
            # 这条每次运行都要说。整图硬缩到模型输入尺寸，结果一定是错的，而且错得很像
            # "模型没读到 ROI"——只在日志里提醒一次的话，人翻不到，排查半天。
            warn = (f"图像 {img.width}×{img.height} 与模型输入 {w}×{h} 不一致，已整图缩放后推理，结果不可用。"
                    f"请在上游采集节点打开「固定 ROI 裁剪」（或接一个「裁剪 ROI」节点）并把 ROI 宽/高填成 {w}/{h}"
                    + ("；长宽比也变了" if abs(img.width / max(1, img.height) - w / max(1, h)) > 0.01 else ""))
            data = cv2.resize(data, (w, h), interpolation=cv2.INTER_AREA if w < img.width else cv2.INTER_LINEAR)
        # scale：网络输入 → 送进来这张图的换算比；origin：这张图在原始帧里的位置（没裁过就是 0,0）
        return data, {"scale": (w / max(1, img.width), h / max(1, img.height)), "pad": (0, 0),
                      "orig": (img.width, img.height), "origin": img.origin, "warn": warn}

    # ---- 后处理 ----
    def decode(self, ctx, outs, meta, img):
        if not outs:
            raise NodeError("模型没有输出")
        if meta.get("warn"):
            ctx.log(meta["warn"], "warning")       # 每次运行都报，这种错配不该只在日志里出现一次
        pts = np.asarray(outs[0], dtype=np.float32)
        pts = pts.reshape(-1, pts.shape[-1]) if pts.ndim >= 2 else pts.reshape(-1, 2)
        n = int(self.get("num_points"))
        if pts.shape[0] < n:
            raise NodeError(f"模型输出 {pts.shape[0]} 个点，少于设定的 {n}；检查模型或「轨迹点数」")
        pts = pts[:n, :2]
        aux = np.asarray(outs[1], dtype=np.float32).reshape(-1, 2) if len(outs) > 1 else np.zeros((0, 2), np.float32)
        prob = float(np.asarray(outs[2]).reshape(-1)[0]) if len(outs) > 2 else 1.0
        sx, sy = meta["scale"]
        origin = np.array(meta.get("origin", (0, 0)), np.float32)
        # disp = 在"送进本节点的那张图"里的坐标 → 叠加层画在它上面才对得齐，面板上看得出轨迹落得对不对
        # full = 原始帧坐标（加上上游裁掉的原点）→ 这才是输出给下游和 PLC 的，裁不裁都不变
        disp = pts / np.array([sx, sy], np.float32)
        full = disp + origin
        aux_disp = aux / np.array([sx, sy], np.float32) if len(aux) else aux
        aux_full = aux_disp + origin if len(aux) else aux
        valid = prob >= float(self.get("conf"))
        length = float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum())
        # 只画点和连线：轨迹点本来就密，压上序号或状态文字会把点本身盖住。
        # 状态和置信度从 valid / score 两个输出口取，不往图上写。
        for i in range(n - 1):
            ctx.add_overlay(Overlay.line(Line(float(disp[i, 0]), float(disp[i, 1]),
                                              float(disp[i + 1, 0]), float(disp[i + 1, 1])),
                                         "#ff1744" if valid else "#9e9e9e"))
        if valid:
            ctx.add_overlay(Overlay.points(disp[1:-1], "#ffffff"))      # 中间点
            ctx.add_overlay(Overlay.points(disp[:1], "#76ff03"))        # 第 0 点：进刀侧
            ctx.add_overlay(Overlay.points(disp[-1:], "#00b0ff"))       # 末点：出刀侧
        else:
            ctx.add_overlay(Overlay.points(disp, "#bdbdbd"))
        if self.get("draw_aux") and len(aux_disp):
            ctx.add_overlay(Overlay.points(aux_disp, "#ffeb3b"))
        if self.get("judge"):
            ctx.judge(valid, "轨迹不可执行（空夹具/遮挡/刀已插入）")
        return {"points": [[float(x), float(y)] for x, y in full], "valid": bool(valid), "score": prob,
                "length": length, "aux": [[float(x), float(y)] for x, y in aux_full],
                "start": Point(float(full[0, 0]), float(full[0, 1])), "end": Point(float(full[-1, 0]), float(full[-1, 1]))}


OnnxTrajectory.inputs, OnnxTrajectory.outputs = OnnxTrajectory.build_ports(1)

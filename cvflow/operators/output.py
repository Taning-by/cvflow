"""Output operators: publish results, save images, render overlays, log, send over comm."""
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2

from ..core import paths
from ..core.draw import render_overlays
from ..core.engine import align_overlays
from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType, Image
from ._util import require_image

_COLOR = "#5d4037"


@register
class Publish(Node):
    type_id = "output.publish"
    category = "Output"
    label = "Publish Result"
    description = "Name values so communication templates and the result panel can use them ({out.name})."
    color = _COLOR
    inputs = [Port(n, DataType.ANY, optional=True) for n in "abcd"]
    params = [Param("name_a", "a", "string"), Param("name_b", "b", "string"),
              Param("name_c", "c", "string"), Param("name_d", "d", "string")]

    def process(self, ctx, inputs):
        links = ctx.graph.input_links(self.id)
        for k in "abcd":
            if k in links:
                ctx.publish(self.get(f"name_{k}") or k, inputs.get(k))
        return {}


@register
class RenderOverlays(Node):
    type_id = "output.render"
    category = "Output"
    label = "Render Overlays"
    description = "Draw all overlays produced so far in this run onto the image (for saving / display)."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE),
              Port("after", DataType.ANY, optional=True, description="Link any judge output to force running after it")]
    outputs = [Port("image", DataType.IMAGE)]
    params = [Param("thickness", 1, "int", min=1, max=10), Param("stamp_result", True, "bool")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        # 各节点的叠加层画在各自输入图的坐标里；裁剪小图上画的轨迹搬到整帧（或反过来）要按原点平移
        overlays = align_overlays(ctx.results.values(), img)
        out = render_overlays(img, overlays, int(self.get("thickness")))
        if self.get("stamp_result"):
            txt = "NG" if ctx.judgement is False else "OK" if ctx.judgement else ""
            if txt:
                cv2.putText(out.data, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255) if txt == "NG" else (0, 200, 0), 2)
        return {"image": out}


@register
class SaveImage(Node):
    """存图。文件名 = 前缀 + [name 输入] + 后缀，后缀按「命名方式」是采集时间、序号或两者都有。

    和 VisionMaster 的「图像保存」一样：时间格式可改（默认到毫秒），序号从「起始序号」开始、
    按位数补零，设了上限就循环覆盖——产线上长期跑不会把盘存满。序号记在节点状态里，随方案保存，
    重开软件接着数；改「起始序号」就从那个数重新数。
    """

    type_id = "output.save_image"
    category = "Output"
    label = "Save Image"
    description = ("Save the image to disk. File name = prefix + linked name + suffix; the suffix is the capture "
                   "time and/or a running index. Optionally only NG/OK runs; place it after the Judge nodes.")
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE),
              Port("name", DataType.STRING, optional=True,
                   description="自定义的文件名主体（例如条码、产品号）：接上后放在前缀和时间/序号后缀之间"),
              Port("after", DataType.ANY, optional=True, description="Link a judge output to force running after it")]
    outputs = [Port("path", DataType.STRING, description="实际写出的文件路径；按条件没存时为空"),
               Port("index", DataType.INT, description="本次用掉的序号；没存图或命名方式不含序号时为 0")]
    params = [Param("directory", "./captures", "dir"),
              Param("when", "always", "enum", choices=["always", "ng_only", "ok_only"]),
              Param("format", "png", "enum", choices=["png", "jpg", "bmp"]),
              Param("subfolder_by_result", True, "bool"),
              Param("prefix", "", "string", description="文件名开头固定的字样，例如工位号 \"S1_\""),
              Param("naming", "time", "enum", choices=["time", "index", "time_index", "none"], label="命名方式",
                    description="文件名后缀：time=采集时间　index=序号　time_index=时间_序号　"
                                "none=不加后缀（同名会覆盖，一般只在接了 name 输入时用）"),
              Param("time_format", "%Y%m%d_%H%M%S_%3f", "string", label="时间格式", advanced=True,
                    description="strftime 写法，%3f 是毫秒、%f 是微秒。默认写出来像 20261018_093012_345。"
                                "时间取的是这帧图像的采集时间"),
              Param("index_start", 1, "int", min=0, max=1_000_000_000, label="起始序号",
                    description="改了它就从这个数重新数"),
              Param("index_digits", 4, "int", min=1, max=12, label="序号位数", advanced=True,
                    description="不够的位数前面补 0（4 位：0001）；超过位数照实写"),
              Param("index_max", 0, "int", min=0, max=1_000_000_000, label="序号上限", advanced=True,
                    description="0 表示不限。到了上限就回到起始序号循环覆盖旧图，磁盘不会被存满"),
              Param("jpeg_quality", 92, "int", min=1, max=100, advanced=True)]

    def on_param_changed(self, name, value):
        if name == "index_start":
            self.state.pop("next_index", None)          # 改起始序号＝从头重新数

    def _next_index(self) -> int:
        """取本次要用的序号并把计数器往前拨一格。越过上限（或起始序号被改大）就回到起始序号。"""
        start, top = int(self.get("index_start")), int(self.get("index_max"))
        i = int(self.state.get("next_index", start))
        if i < start or (top and i > top):
            i = start
        self.state["next_index"] = i + 1
        return i

    @staticmethod
    def _stamp(ts: float, fmt: str) -> str:
        dt = datetime.fromtimestamp(ts) if ts else datetime.now()
        ms = f"{dt.microsecond // 1000:03d}"
        try:
            out = dt.strftime((fmt or "%Y%m%d_%H%M%S_%3f").replace("%3f", ms))
        except ValueError:                               # Windows 上不认识的指令会直接抛
            out = ""
        if not out or "%" in out:                        # Linux 上不认识的指令原样留下：也算写坏了
            out = dt.strftime("%Y%m%d_%H%M%S_") + ms     # 格式写坏了别让存图失败，退回默认
        return out

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        when = self.get("when")
        ng = ctx.judgement is False
        if (when == "ng_only" and not ng) or (when == "ok_only" and ng):
            return {"path": "", "index": 0}
        d = paths.resolve(self.get("directory") or ".")
        if self.get("subfolder_by_result"):
            d = os.path.join(d, "NG" if ng else "OK")
        os.makedirs(d, exist_ok=True)
        naming = self.get("naming")
        parts: list[str] = []
        linked = inputs.get("name")
        if linked is not None and str(linked) != "":
            parts.append(os.path.basename(str(linked)))
        if naming in ("time", "time_index"):
            parts.append(self._stamp(img.timestamp, self.get("time_format")))
        idx = 0
        if naming in ("index", "time_index"):
            idx = self._next_index()
            parts.append(f"{idx:0{int(self.get('index_digits'))}d}")
        stem = f"{self.get('prefix')}{'_'.join(parts)}" or str(ctx.run_id)
        fmt = self.get("format")
        path = os.path.join(d, f"{stem}.{fmt}")
        if naming == "time":                             # 同一毫秒存两张：加个尾号，别悄悄把上一张冲掉。
            k = 1                                        # 序号模式撞名则是到了上限回头循环，本来就是要覆盖
            while os.path.exists(path):
                path = os.path.join(d, f"{stem}-{k}.{fmt}")
                k += 1
        params = [cv2.IMWRITE_JPEG_QUALITY, int(self.get("jpeg_quality"))] if fmt == "jpg" else []
        if not cv2.imwrite(path, img.data, params):
            raise NodeError(f"无法写入 {path}")
        return {"path": path, "index": idx}


@register
class Log(Node):
    type_id = "output.log"
    category = "Output"
    label = "Log"
    color = _COLOR
    inputs = [Port("value", DataType.ANY)]
    outputs = [Port("value", DataType.ANY)]
    params = [Param("prefix", "", "string"), Param("level", "info", "enum", choices=["debug", "info", "warning", "error"])]

    def process(self, ctx, inputs):
        ctx.log(f"{self.get('prefix')}{inputs['value']!r}", self.get("level"))
        return {"value": inputs["value"]}


@register
class CommSend(Node):
    type_id = "output.comm_send"
    category = "Output"
    label = "Send Message (legacy)"
    description = ("Send text to a communication device. Kept for existing solutions - the "
                   "Communication category's 'Comm Send' supersedes it (HEX, reply to the "
                   "requesting client, templates).")
    color = _COLOR
    inputs = [Port("text", DataType.STRING)]
    outputs = [Port("sent", DataType.BOOL)]
    params = [Param("device", "", "choice", options="comm.devices"),
              Param("append", "\\n", "string", description="Terminator (escapes allowed)")]

    def process(self, ctx, inputs):
        from ..comm.manager import get_manager
        mgr = get_manager()
        if mgr is None:
            raise NodeError("通信管理器未启动")
        text = str(inputs["text"]) + self.get("append").encode().decode("unicode_escape")
        ok = mgr.send(self.get("device"), text)
        return {"sent": bool(ok)}


@register
class ModbusWrite(Node):
    type_id = "output.modbus_write"
    category = "Output"
    label = "Modbus Write (by address)"
    description = ("Write a value into holding register(s) by raw address. Kept for existing "
                   "solutions - 'Write Data Point' writes by data point name instead, which "
                   "carries the type, byte order and scaling with it.")
    color = _COLOR
    inputs = [Port("value", DataType.ANY)]
    outputs = [Port("ok", DataType.BOOL)]
    params = [Param("device", "", "choice", options="comm.devices"),
              Param("address", 0, "int", min=0, max=65535),
              Param("kind", "int16", "enum", choices=["int16", "uint16", "int32", "float32", "bool"]),
              Param("scale", 1.0, "float", description="value * scale before conversion")]

    def process(self, ctx, inputs):
        from ..comm.manager import get_manager
        mgr = get_manager()
        if mgr is None:
            raise NodeError("通信管理器未启动")
        v = inputs["value"]
        if self.get("kind") != "bool":
            v = float(v) * float(self.get("scale"))
        ok = mgr.modbus_write(self.get("device"), self.get("address"), v, self.get("kind"))
        return {"ok": bool(ok)}

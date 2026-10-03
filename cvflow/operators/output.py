"""Output operators: publish results, save images, render overlays, log, send over comm."""
from __future__ import annotations

import os
import time
from datetime import datetime

import cv2

from ..core import paths
from ..core.draw import render_overlays
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
        overlays = [ov for r in ctx.results.values() for ov in r.overlays]
        out = render_overlays(img, overlays, int(self.get("thickness")))
        if self.get("stamp_result"):
            txt = "NG" if ctx.judgement is False else "OK" if ctx.judgement else ""
            if txt:
                cv2.putText(out.data, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255) if txt == "NG" else (0, 200, 0), 2)
        return {"image": out}


@register
class SaveImage(Node):
    type_id = "output.save_image"
    category = "Output"
    label = "Save Image"
    description = "Save the image to disk (optionally only NG runs; place it after the Judge nodes)."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE), Port("name", DataType.STRING, optional=True),
              Port("after", DataType.ANY, optional=True, description="Link a judge output to force running after it")]
    outputs = [Port("path", DataType.STRING)]
    params = [Param("directory", "./captures", "dir"),
              Param("when", "always", "enum", choices=["always", "ng_only", "ok_only"]),
              Param("format", "png", "enum", choices=["png", "jpg", "bmp"]),
              Param("subfolder_by_result", True, "bool"),
              Param("prefix", "", "string"),
              Param("jpeg_quality", 92, "int", min=1, max=100, advanced=True)]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        when = self.get("when")
        ng = ctx.judgement is False
        if (when == "ng_only" and not ng) or (when == "ok_only" and ng):
            return {"path": ""}
        d = paths.resolve(self.get("directory") or ".")
        if self.get("subfolder_by_result"):
            d = os.path.join(d, "NG" if ng else "OK")
        os.makedirs(d, exist_ok=True)
        base = inputs.get("name") or f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{ctx.run_id}"
        name = f"{self.get('prefix')}{os.path.basename(str(base))}.{self.get('format')}"
        path = os.path.join(d, name)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(self.get("jpeg_quality"))] if self.get("format") == "jpg" else []
        if not cv2.imwrite(path, img.data, params):
            raise NodeError(f"无法写入 {path}")
        return {"path": path}


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
    label = "Send Message"
    description = "Send text to a communication device (TCP/UDP/serial) configured in the Communication panel."
    color = _COLOR
    inputs = [Port("text", DataType.STRING)]
    outputs = [Port("sent", DataType.BOOL)]
    params = [Param("device", "", "string"), Param("append", "\\n", "string", description="Terminator (escapes allowed)")]

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
    label = "Modbus Write"
    description = "Write a value into holding register(s) of a Modbus device (client or server)."
    color = _COLOR
    inputs = [Port("value", DataType.ANY)]
    outputs = [Port("ok", DataType.BOOL)]
    params = [Param("device", "", "string"), Param("address", 0, "int", min=0, max=65535),
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
        ok = mgr.modbus_write(self.get("device"), int(self.get("address")), v, self.get("kind"))
        return {"ok": bool(ok)}

"""Image preprocessing operators (thin, well-typed wrappers over OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType, Overlay, Point, Rect
from ._util import as_gray, as_bgr, crop, odd, require_image

_IMG_IN = [Port("image", DataType.IMAGE)]
_IMG_OUT = [Port("image", DataType.IMAGE)]
_COLOR = "#6a1b9a"


@register
class ColorConvert(Node):
    type_id = "preprocess.color"
    category = "Preprocess"
    label = "Color Convert"
    description = "Convert colour space; 'channel' picks one plane of the result."
    color = _COLOR
    inputs, outputs = _IMG_IN, _IMG_OUT
    params = [Param("mode", "gray", "enum", choices=["gray", "bgr", "hsv", "hls", "lab", "rgb"]),
              Param("channel", -1, "int", min=-1, max=2, description="-1 keeps all channels")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        mode = self.get("mode")
        if mode == "gray":
            out = as_gray(img)
        else:
            bgr = as_bgr(img)
            code = {"bgr": None, "hsv": cv2.COLOR_BGR2HSV, "hls": cv2.COLOR_BGR2HLS,
                    "lab": cv2.COLOR_BGR2LAB, "rgb": cv2.COLOR_BGR2RGB}[mode]
            out = bgr if code is None else cv2.cvtColor(bgr, code)
        ch = int(self.get("channel"))
        if ch >= 0 and out.ndim == 3:
            out = out[:, :, ch]
        return {"image": img.derive(out)}


@register
class Blur(Node):
    type_id = "preprocess.blur"
    category = "Preprocess"
    label = "Blur"
    color = _COLOR
    inputs, outputs = _IMG_IN, _IMG_OUT
    params = [Param("method", "gaussian", "enum", choices=["gaussian", "median", "box", "bilateral"]),
              Param("ksize", 5, "int", min=1, max=99, step=2),
              Param("sigma", 0.0, "float", min=0, max=50, description="Gaussian sigma / bilateral sigma colour")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        k = odd(self.get("ksize"))
        m = self.get("method")
        d = img.data
        if m == "gaussian":
            out = cv2.GaussianBlur(d, (k, k), float(self.get("sigma")))
        elif m == "median":
            out = cv2.medianBlur(d, k)
        elif m == "box":
            out = cv2.blur(d, (k, k))
        else:
            s = float(self.get("sigma")) or 50.0
            out = cv2.bilateralFilter(d, k, s, s)
        return {"image": img.derive(out)}


@register
class Threshold(Node):
    type_id = "preprocess.threshold"
    category = "Preprocess"
    label = "Threshold"
    description = "Binarise a grayscale image (colour input is converted first)."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE),
              Port("thresh", DataType.INT, optional=True, description="Overrides the 'thresh' parameter when linked")]
    outputs = [Port("image", DataType.IMAGE), Port("thresh", DataType.FLOAT)]
    params = [Param("method", "binary", "enum",
                    choices=["binary", "binary_inv", "otsu", "otsu_inv", "adaptive_mean", "adaptive_gaussian", "range"]),
              Param("thresh", 128, "int", min=0, max=255),
              Param("high", 255, "int", min=0, max=255, description="Upper bound for 'range'"),
              Param("block_size", 31, "int", min=3, max=255, step=2, advanced=True),
              Param("c", 5.0, "float", min=-50, max=50, advanced=True)]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        g = as_gray(img)
        m = self.get("method")
        t = int(inputs["thresh"]) if inputs.get("thresh") is not None else int(self.get("thresh"))
        t = max(0, min(255, t))
        used = float(t)
        if m == "binary":
            _, out = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
        elif m == "binary_inv":
            _, out = cv2.threshold(g, t, 255, cv2.THRESH_BINARY_INV)
        elif m == "otsu":
            used, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif m == "otsu_inv":
            used, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        elif m == "range":
            out = cv2.inRange(g, t, int(self.get("high")))
        else:
            flag = cv2.ADAPTIVE_THRESH_MEAN_C if m == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            out = cv2.adaptiveThreshold(g, 255, flag, cv2.THRESH_BINARY, odd(self.get("block_size")), float(self.get("c")))
        return {"image": img.derive(out), "thresh": float(used)}


@register
class Morphology(Node):
    type_id = "preprocess.morphology"
    category = "Preprocess"
    label = "Morphology"
    color = _COLOR
    inputs, outputs = _IMG_IN, _IMG_OUT
    params = [Param("op", "open", "enum", choices=["erode", "dilate", "open", "close", "gradient", "tophat", "blackhat"]),
              Param("shape", "ellipse", "enum", choices=["rect", "ellipse", "cross"]),
              Param("ksize", 3, "int", min=1, max=99),
              Param("iterations", 1, "int", min=1, max=20)]

    _OPS = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE, "gradient": cv2.MORPH_GRADIENT, "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT}
    _SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        k = max(1, int(self.get("ksize")))
        kernel = cv2.getStructuringElement(self._SHAPES[self.get("shape")], (k, k))
        out = cv2.morphologyEx(img.data, self._OPS[self.get("op")], kernel, iterations=int(self.get("iterations")))
        return {"image": img.derive(out)}


@register
class Resize(Node):
    type_id = "preprocess.resize"
    category = "Preprocess"
    label = "Resize"
    color = _COLOR
    inputs, outputs = _IMG_IN, [Port("image", DataType.IMAGE), Port("scale_x", DataType.FLOAT), Port("scale_y", DataType.FLOAT)]
    params = [Param("mode", "scale", "enum", choices=["scale", "size"]),
              Param("scale", 0.5, "float", min=0.01, max=10),
              Param("width", 640, "int", min=1, max=16384),
              Param("height", 480, "int", min=1, max=16384),
              Param("interpolation", "area", "enum", choices=["nearest", "linear", "area", "cubic"])]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        interp = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
                  "cubic": cv2.INTER_CUBIC}[self.get("interpolation")]
        if self.get("mode") == "scale":
            s = float(self.get("scale"))
            out = cv2.resize(img.data, None, fx=s, fy=s, interpolation=interp)
        else:
            out = cv2.resize(img.data, (int(self.get("width")), int(self.get("height"))), interpolation=interp)
        return {"image": img.derive(out), "scale_x": out.shape[1] / img.width, "scale_y": out.shape[0] / img.height}


@register
class Crop(Node):
    type_id = "preprocess.crop"
    category = "Preprocess"
    label = "Crop ROI"
    description = "Crop to a rectangle given as a parameter or supplied by an upstream node."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE), Port("roi", DataType.RECT, optional=True)]
    outputs = [Port("image", DataType.IMAGE), Port("offset", DataType.POINT), Port("roi", DataType.RECT)]
    params = [Param("roi", None, "rect")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        roi = inputs.get("roi") if isinstance(inputs.get("roi"), Rect) else self.rect("roi")
        if roi is None:
            raise NodeError("未设置 ROI（请在图像上绘制，或连接 'roi' 输入）")
        view, eff = crop(img, roi)
        ctx.add_overlay(Overlay.rect(eff, "#ffa500", "ROI"))
        out = img.derive(np.ascontiguousarray(view))
        out.meta["offset"] = (eff.x, eff.y)
        return {"image": out, "offset": Point(eff.x, eff.y), "roi": eff}


@register
class Canny(Node):
    type_id = "preprocess.canny"
    category = "Preprocess"
    label = "Canny Edges"
    color = _COLOR
    inputs, outputs = _IMG_IN, _IMG_OUT
    params = [Param("low", 50, "int", min=0, max=1000), Param("high", 150, "int", min=0, max=1000),
              Param("aperture", 3, "int", min=3, max=7, step=2), Param("l2", False, "bool", label="L2 gradient")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        out = cv2.Canny(as_gray(img), int(self.get("low")), int(self.get("high")),
                        apertureSize=odd(self.get("aperture")), L2gradient=bool(self.get("l2")))
        return {"image": img.derive(out)}


@register
class Enhance(Node):
    type_id = "preprocess.enhance"
    category = "Preprocess"
    label = "Enhance"
    description = "Contrast / brightness, histogram equalisation or CLAHE."
    color = _COLOR
    inputs, outputs = _IMG_IN, _IMG_OUT
    params = [Param("method", "linear", "enum", choices=["linear", "equalize", "clahe", "normalize"]),
              Param("alpha", 1.0, "float", min=0, max=10, description="Contrast gain (linear)"),
              Param("beta", 0.0, "float", min=-255, max=255, description="Brightness offset (linear)"),
              Param("clip_limit", 2.0, "float", min=0.1, max=40, advanced=True),
              Param("tile", 8, "int", min=1, max=64, advanced=True)]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        m = self.get("method")
        d = img.data
        if m == "linear":
            out = cv2.convertScaleAbs(d, alpha=float(self.get("alpha")), beta=float(self.get("beta")))
        elif m == "normalize":
            out = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            g = as_gray(img)
            if m == "equalize":
                out = cv2.equalizeHist(g)
            else:
                t = int(self.get("tile"))
                out = cv2.createCLAHE(float(self.get("clip_limit")), (t, t)).apply(g)
        return {"image": img.derive(out)}


@register
class Transform(Node):
    type_id = "preprocess.transform"
    category = "Preprocess"
    label = "Rotate / Flip"
    color = _COLOR
    inputs, outputs = _IMG_IN, _IMG_OUT
    params = [Param("rotate", "0", "enum", choices=["0", "90", "180", "270"]),
              Param("flip", "none", "enum", choices=["none", "horizontal", "vertical", "both"])]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        d = img.data
        rot = {"0": None, "90": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180,
               "270": cv2.ROTATE_90_COUNTERCLOCKWISE}[self.get("rotate")]
        if rot is not None:
            d = cv2.rotate(d, rot)
        fl = {"none": None, "horizontal": 1, "vertical": 0, "both": -1}[self.get("flip")]
        if fl is not None:
            d = cv2.flip(d, fl)
        return {"image": img.derive(d)}


@register
class Bitwise(Node):
    type_id = "preprocess.bitwise"
    category = "Preprocess"
    label = "Bitwise"
    description = "Combine two binary/gray images (b is ignored for 'not')."
    color = _COLOR
    inputs = [Port("a", DataType.IMAGE), Port("b", DataType.IMAGE, optional=True)]
    outputs = _IMG_OUT
    params = [Param("op", "and", "enum", choices=["and", "or", "xor", "not", "subtract", "absdiff"])]

    def process(self, ctx, inputs):
        a = require_image(inputs["a"], "a")
        op = self.get("op")
        if op == "not":
            return {"image": a.derive(cv2.bitwise_not(a.data))}
        b = require_image(inputs.get("b"), "b")
        if a.data.shape != b.data.shape:
            raise NodeError(f"两张图像尺寸不同：{a.data.shape} 与 {b.data.shape}")
        fn = {"and": cv2.bitwise_and, "or": cv2.bitwise_or, "xor": cv2.bitwise_xor,
              "subtract": cv2.subtract, "absdiff": cv2.absdiff}[op]
        return {"image": a.derive(fn(a.data, b.data))}

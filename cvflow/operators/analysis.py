"""Measurement and inspection operators: blobs, contours, template matching,
edge caliper, circle finding, intensity statistics, QR codes."""
from __future__ import annotations

import os

import cv2
import numpy as np

from ..core import paths
from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import Circle, DataType, Image, Line, Overlay, Point, Rect
from ._util import as_gray, crop, odd, require_image

_COLOR = "#1565c0"


@register
class BlobAnalysis(Node):
    type_id = "analysis.blob"
    category = "Analysis"
    label = "Blob Analysis"
    description = "Connected components of a binary image with area/size filters."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE, description="Binary image (non-zero = foreground)")]
    outputs = [Port("count", DataType.INT), Port("blobs", DataType.LIST), Port("largest_area", DataType.FLOAT),
               Port("total_area", DataType.FLOAT), Port("centers", DataType.LIST), Port("mask", DataType.IMAGE)]
    params = [Param("min_area", 50, "int", min=0, max=10_000_000),
              Param("max_area", 1_000_000, "int", min=1, max=100_000_000),
              Param("min_width", 0, "int", min=0, max=100000, advanced=True),
              Param("min_height", 0, "int", min=0, max=100000, advanced=True),
              Param("connectivity", "8", "enum", choices=["4", "8"]),
              Param("sort", "area_desc", "enum", choices=["area_desc", "area_asc", "left_to_right", "top_to_bottom"]),
              Param("max_count", 100, "int", min=1, max=10000)]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        g = as_gray(img)
        binary = (g > 0).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=int(self.get("connectivity")))
        lo, hi = int(self.get("min_area")), int(self.get("max_area"))
        mw, mh = int(self.get("min_width")), int(self.get("min_height"))
        blobs = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not (lo <= area <= hi) or w < mw or h < mh:
                continue
            blobs.append({"label": i, "area": float(area), "x": x, "y": y, "w": w, "h": h,
                          "cx": float(cents[i][0]), "cy": float(cents[i][1])})
        key = {"area_desc": (lambda b: -b["area"]), "area_asc": (lambda b: b["area"]),
               "left_to_right": (lambda b: b["cx"]), "top_to_bottom": (lambda b: b["cy"])}[self.get("sort")]
        blobs.sort(key=key)
        blobs = blobs[: int(self.get("max_count"))]
        keep = np.zeros(n, dtype=np.uint8)
        for b in blobs:
            keep[b["label"]] = 255
        mask = keep[labels]
        for b in blobs:
            ctx.add_overlay(Overlay.rect(Rect(b["x"], b["y"], b["w"], b["h"]), "#00ff00"))
            ctx.add_overlay(Overlay.text(b["x"], b["y"] - 4, f"{int(b['area'])}"))
        ctx.add_overlay(Overlay.points([(b["cx"], b["cy"]) for b in blobs], "#ff0000"))
        return {"count": len(blobs), "blobs": blobs,
                "largest_area": blobs[0]["area"] if blobs and self.get("sort") == "area_desc" else
                (max((b["area"] for b in blobs), default=0.0)),
                "total_area": float(sum(b["area"] for b in blobs)),
                "centers": [Point(b["cx"], b["cy"]) for b in blobs], "mask": img.derive(mask)}


@register
class FindContours(Node):
    type_id = "analysis.contours"
    category = "Analysis"
    label = "Find Contours"
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("contours", DataType.CONTOURS), Port("count", DataType.INT), Port("areas", DataType.LIST),
               Port("largest", DataType.CONTOURS)]
    params = [Param("mode", "external", "enum", choices=["external", "list", "tree"]),
              Param("min_area", 10.0, "float", min=0, max=1e9),
              Param("approx_eps", 0.0, "float", min=0, max=50, description="Polygon approximation epsilon (px)")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        g = as_gray(img)
        mode = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST, "tree": cv2.RETR_TREE}[self.get("mode")]
        cnts, _ = cv2.findContours((g > 0).astype(np.uint8), mode, cv2.CHAIN_APPROX_SIMPLE)
        eps = float(self.get("approx_eps"))
        if eps > 0:
            cnts = [cv2.approxPolyDP(c, eps, True) for c in cnts]
        pairs = [(c, cv2.contourArea(c)) for c in cnts]
        pairs = [(c, a) for c, a in pairs if a >= float(self.get("min_area"))]
        pairs.sort(key=lambda p: -p[1])
        cnts = [c for c, _ in pairs]
        ctx.add_overlay(Overlay.contours(cnts, "#00ffff"))
        return {"contours": cnts, "count": len(cnts), "areas": [a for _, a in pairs],
                "largest": cnts[:1]}


def _nms(boxes: list[tuple[int, int, int, int, float]], overlap: float):
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in boxes:
        x, y, w, h, s = b
        ok = True
        for k in kept:
            ix = max(0, min(x + w, k[0] + k[2]) - max(x, k[0]))
            iy = max(0, min(y + h, k[1] + k[3]) - max(y, k[1]))
            inter = ix * iy
            if inter / float(w * h + k[2] * k[3] - inter) > overlap:
                ok = False
                break
        if ok:
            kept.append(b)
    return kept


@register
class TemplateMatch(Node):
    type_id = "analysis.template_match"
    category = "Analysis"
    label = "Template Match"
    description = "Normalised cross-correlation template matching (translation only)."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE), Port("template", DataType.IMAGE, optional=True)]
    outputs = [Port("found", DataType.BOOL), Port("score", DataType.FLOAT), Port("center", DataType.POINT),
               Port("rect", DataType.RECT), Port("count", DataType.INT), Port("matches", DataType.LIST)]
    params = [Param("template_path", "", "file", filter="Images (*.png *.jpg *.bmp *.tif)"),
              Param("method", "ccoeff_normed", "enum", choices=["ccoeff_normed", "ccorr_normed", "sqdiff_normed"]),
              Param("min_score", 0.7, "float", min=0, max=1, step=0.01),
              Param("max_matches", 1, "int", min=1, max=500),
              Param("overlap", 0.3, "float", min=0, max=1, advanced=True),
              Param("search_roi", None, "rect")]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._tpl_cache = None

    def _template(self, inp) -> np.ndarray:
        if isinstance(inp, Image):
            return as_gray(inp)
        path = paths.resolve(self.get("template_path"))
        if not path:
            raise NodeError("no template image (set template_path or connect 'template')")
        if not os.path.isfile(path):
            raise NodeError(f"template not found: {path}")
        key = (path, os.path.getmtime(path))
        if not self._tpl_cache or self._tpl_cache[0] != key:
            t = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if t is None:
                raise NodeError(f"cannot read template {path}")
            self._tpl_cache = (key, t)
        return self._tpl_cache[1]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        tpl = self._template(inputs.get("template"))
        g = as_gray(img)
        view, eff = crop(img.derive(g), self.rect("search_roi"))
        th, tw = tpl.shape[:2]
        if th > view.shape[0] or tw > view.shape[1]:
            raise NodeError("template larger than search area")
        method = {"ccoeff_normed": cv2.TM_CCOEFF_NORMED, "ccorr_normed": cv2.TM_CCORR_NORMED,
                  "sqdiff_normed": cv2.TM_SQDIFF_NORMED}[self.get("method")]
        res = cv2.matchTemplate(view, tpl, method)
        if method == cv2.TM_SQDIFF_NORMED:
            res = 1.0 - res
        min_score = float(self.get("min_score"))
        max_n = int(self.get("max_matches"))
        if max_n == 1:
            _, mx, _, loc = cv2.minMaxLoc(res)
            cands = [(loc[0], loc[1], tw, th, float(mx))] if mx >= min_score else []
        else:
            ys, xs = np.where(res >= min_score)
            cands = _nms([(int(x), int(y), tw, th, float(res[y, x])) for y, x in zip(ys, xs)], float(self.get("overlap")))[:max_n]
        matches = []
        for x, y, w, h, s in cands:
            r = Rect(x + eff.x, y + eff.y, w, h)
            matches.append({"x": r.x, "y": r.y, "w": w, "h": h, "cx": r.cx, "cy": r.cy, "score": s})
            ctx.add_overlay(Overlay.rect(r, "#00ff00", f"{s:.2f}"))
            ctx.add_overlay(Overlay.text(r.x, r.y - 4, f"{s:.2f}"))
        if self.rect("search_roi"):
            ctx.add_overlay(Overlay.rect(eff, "#ffa500", "search"))
        found = bool(matches)
        best = matches[0] if found else None
        return {"found": found, "score": best["score"] if best else 0.0,
                "center": Point(best["cx"], best["cy"]) if best else Point(-1, -1),
                "rect": Rect(best["x"], best["y"], best["w"], best["h"]) if best else Rect(0, 0, 0, 0),
                "count": len(matches), "matches": matches}


def _subpixel_peak(profile: np.ndarray, i: int) -> float:
    """Parabolic interpolation of a 1-D peak at index i."""
    if i <= 0 or i >= len(profile) - 1:
        return float(i)
    a, b, c = profile[i - 1], profile[i], profile[i + 1]
    denom = a - 2 * b + c
    return float(i) if denom == 0 else float(i + 0.5 * (a - c) / denom)


@register
class EdgeCaliper(Node):
    type_id = "analysis.caliper"
    category = "Analysis"
    label = "Edge Caliper"
    description = ("Project an ROI to a 1-D profile and locate edges with sub-pixel accuracy. "
                   "Scan direction 'horizontal' finds vertical edges (x positions).")
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE), Port("roi", DataType.RECT, optional=True)]
    outputs = [Port("edges", DataType.LIST), Port("count", DataType.INT), Port("first", DataType.FLOAT),
               Port("last", DataType.FLOAT), Port("width", DataType.FLOAT), Port("profile", DataType.LIST)]
    params = [Param("roi", None, "rect"),
              Param("direction", "horizontal", "enum", choices=["horizontal", "vertical"]),
              Param("polarity", "any", "enum", choices=["any", "dark_to_light", "light_to_dark"]),
              Param("min_contrast", 20.0, "float", min=0, max=255, description="Min gradient magnitude"),
              Param("select", "all", "enum", choices=["all", "first", "last", "strongest", "first_last"]),
              Param("smooth", 3, "int", min=1, max=51, description="Gaussian smoothing of the profile")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        roi = inputs.get("roi") if isinstance(inputs.get("roi"), Rect) else self.rect("roi")
        if roi is None:
            raise NodeError("no ROI set")
        g = as_gray(img)
        view, eff = crop(img.derive(g), roi)
        horizontal = self.get("direction") == "horizontal"
        profile = view.astype(np.float32).mean(axis=0 if horizontal else 1)
        k = odd(self.get("smooth"))
        if k > 1:
            profile = cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), 0).ravel()
        grad = np.gradient(profile)
        pol = self.get("polarity")
        mag = grad.copy()
        if pol == "dark_to_light":
            mag[mag < 0] = 0
        elif pol == "light_to_dark":
            mag = -mag
            mag[mag < 0] = 0
        else:
            mag = np.abs(mag)
        thr = float(self.get("min_contrast"))
        peaks = [i for i in range(1, len(mag) - 1) if mag[i] >= thr and mag[i] >= mag[i - 1] and mag[i] > mag[i + 1]]
        edges = [(_subpixel_peak(mag, i), float(mag[i])) for i in peaks]
        sel = self.get("select")
        if edges:
            if sel == "first":
                edges = edges[:1]
            elif sel == "last":
                edges = edges[-1:]
            elif sel == "strongest":
                edges = [max(edges, key=lambda e: e[1])]
            elif sel == "first_last" and len(edges) > 1:
                edges = [edges[0], edges[-1]]
        offset = eff.x if horizontal else eff.y
        positions = [p + offset for p, _ in edges]
        ctx.add_overlay(Overlay.rect(eff, "#ffa500", "caliper"))
        for p in positions:
            ln = Line(p, eff.y, p, eff.y + eff.h) if horizontal else Line(eff.x, p, eff.x + eff.w, p)
            ctx.add_overlay(Overlay.line(ln, "#ff00ff"))
        first = positions[0] if positions else -1.0
        last = positions[-1] if positions else -1.0
        width = (last - first) if len(positions) >= 2 else 0.0
        return {"edges": positions, "count": len(positions), "first": first, "last": last,
                "width": width, "profile": profile.tolist()}


def _fit_circle_lsq(pts: np.ndarray) -> tuple[float, float, float]:
    """Algebraic least-squares circle fit (Kasa)."""
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0))
    return float(cx), float(cy), float(r)


@register
class CircleFind(Node):
    type_id = "analysis.circle"
    category = "Analysis"
    label = "Find Circle"
    description = "Hough circle detection, or least-squares fit on the largest contour of a binary image."
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE), Port("roi", DataType.RECT, optional=True)]
    outputs = [Port("found", DataType.BOOL), Port("circle", DataType.CIRCLE), Port("cx", DataType.FLOAT),
               Port("cy", DataType.FLOAT), Port("radius", DataType.FLOAT), Port("count", DataType.INT)]
    params = [Param("method", "hough", "enum", choices=["hough", "contour_fit"]),
              Param("roi", None, "rect"),
              Param("min_radius", 10, "int", min=0, max=10000), Param("max_radius", 200, "int", min=1, max=10000),
              Param("min_dist", 20, "int", min=1, max=10000, advanced=True),
              Param("param1", 100, "int", min=1, max=500, advanced=True, description="Canny high threshold"),
              Param("param2", 30, "int", min=1, max=500, advanced=True, description="Accumulator threshold"),
              Param("max_count", 1, "int", min=1, max=100)]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        roi = inputs.get("roi") if isinstance(inputs.get("roi"), Rect) else self.rect("roi")
        g = as_gray(img)
        view, eff = crop(img.derive(g), roi)
        circles: list[Circle] = []
        if self.get("method") == "hough":
            blurred = cv2.medianBlur(view, 5)
            res = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(self.get("min_dist")),
                                   param1=int(self.get("param1")), param2=int(self.get("param2")),
                                   minRadius=int(self.get("min_radius")), maxRadius=int(self.get("max_radius")))
            if res is not None:
                for cx, cy, r in res[0][: int(self.get("max_count"))]:
                    circles.append(Circle(float(cx) + eff.x, float(cy) + eff.y, float(r)))
        else:
            cnts, _ = cv2.findContours((view > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[: int(self.get("max_count"))]
            for c in cnts:
                if len(c) < 5:
                    continue
                cx, cy, r = _fit_circle_lsq(c.reshape(-1, 2))
                if int(self.get("min_radius")) <= r <= int(self.get("max_radius")):
                    circles.append(Circle(cx + eff.x, cy + eff.y, r))
        if roi is not None:
            ctx.add_overlay(Overlay.rect(eff, "#ffa500"))
        for c in circles:
            ctx.add_overlay(Overlay.circle(c, "#00ff00", f"r={c.r:.1f}"))
            ctx.add_overlay(Overlay.points([(c.cx, c.cy)], "#ff0000"))
        best = circles[0] if circles else Circle(-1, -1, 0)
        return {"found": bool(circles), "circle": best, "cx": best.cx, "cy": best.cy, "radius": best.r,
                "count": len(circles)}


@register
class IntensityStats(Node):
    type_id = "analysis.intensity"
    category = "Analysis"
    label = "Intensity Stats"
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE), Port("roi", DataType.RECT, optional=True)]
    outputs = [Port("mean", DataType.FLOAT), Port("std", DataType.FLOAT), Port("min", DataType.FLOAT),
               Port("max", DataType.FLOAT), Port("fill_ratio", DataType.FLOAT)]
    params = [Param("roi", None, "rect"),
              Param("fill_thresh", 128, "int", min=0, max=255, description="Pixels above count as filled")]

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        roi = inputs.get("roi") if isinstance(inputs.get("roi"), Rect) else self.rect("roi")
        view, eff = crop(img.derive(as_gray(img)), roi)
        m, sd = cv2.meanStdDev(view)
        mean, std = float(m[0][0]), float(sd[0][0])
        fill = float((view > int(self.get("fill_thresh"))).mean())
        ctx.add_overlay(Overlay.rect(eff, "#ffa500"))
        ctx.add_overlay(Overlay.text(eff.x, eff.y - 4, f"mean {mean:.1f}"))
        return {"mean": mean, "std": std, "min": float(view.min()), "max": float(view.max()),
                "fill_ratio": fill}


@register
class QRCode(Node):
    type_id = "analysis.qrcode"
    category = "Analysis"
    label = "QR Code"
    color = _COLOR
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("found", DataType.BOOL), Port("text", DataType.STRING), Port("points", DataType.LIST)]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._det = cv2.QRCodeDetector()

    def process(self, ctx, inputs):
        img = require_image(inputs["image"])
        text, pts, _ = self._det.detectAndDecode(img.data)
        found = bool(text)
        points = pts.reshape(-1, 2).tolist() if found and pts is not None else []
        if found:
            ctx.add_overlay(Overlay("polygon", {"points": points}, "#00ff00", label=text))
        return {"found": found, "text": text or "", "points": points}

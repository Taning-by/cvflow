"""Rasterise overlays onto an image with OpenCV (for saving annotated images and headless use)."""
from __future__ import annotations

import cv2
import numpy as np

from .types import Image, Overlay


def _bgr(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) != 6:
        return (0, 255, 0)
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (b, g, r)


def render_overlays(img: Image, overlays: list[Overlay], thickness: int = 1) -> Image:
    canvas = cv2.cvtColor(img.data, cv2.COLOR_GRAY2BGR) if img.is_gray else img.data.copy()
    for ov in overlays:
        col = _bgr(ov.color)
        g = ov.geometry
        k = ov.kind
        try:
            if k == "rect":
                x, y, w, h = int(g["x"]), int(g["y"]), int(g["w"]), int(g["h"])
                if g.get("angle"):
                    box = cv2.boxPoints(((x + w / 2, y + h / 2), (w, h), g["angle"])).astype(np.int32)
                    cv2.polylines(canvas, [box], True, col, thickness)
                else:
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), col, thickness)
                if ov.label:
                    cv2.putText(canvas, ov.label, (x, max(10, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            elif k == "circle":
                cv2.circle(canvas, (int(g["cx"]), int(g["cy"])), int(g["r"]), col, thickness)
            elif k == "line":
                cv2.line(canvas, (int(g["x1"]), int(g["y1"])), (int(g["x2"]), int(g["y2"])), col, thickness)
            elif k == "polygon":
                pts = np.array(g["points"], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, col, thickness)
            elif k == "points":
                for x, y in g["points"]:
                    cv2.drawMarker(canvas, (int(x), int(y)), col, cv2.MARKER_CROSS, 8, 1)
            elif k == "text":
                cv2.putText(canvas, str(g["text"]), (int(g["x"]), int(g["y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            elif k == "contours":
                cv2.drawContours(canvas, [np.asarray(c, dtype=np.int32) for c in g["contours"]], -1, col, thickness)
        except Exception:  # never let a drawing problem break a run
            continue
    return img.derive(canvas)

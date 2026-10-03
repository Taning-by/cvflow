"""Helpers shared by built-in operators."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.node import NodeError
from ..core.types import Image, Rect


def require_image(value, name: str = "image") -> Image:
    if not isinstance(value, Image):
        raise NodeError(f"输入 '{name}' 必须是图像，实际为 {type(value).__name__}")
    return value


def as_gray(img: Image) -> np.ndarray:
    return img.data if img.is_gray else cv2.cvtColor(img.data, cv2.COLOR_BGR2GRAY)


def as_bgr(img: Image) -> np.ndarray:
    return cv2.cvtColor(img.data, cv2.COLOR_GRAY2BGR) if img.is_gray else img.data


def crop(img: Image, roi: Rect | None) -> tuple[np.ndarray, Rect]:
    """Crop ``img`` to ``roi`` (clipped). Returns (view, effective_rect). Full image if roi is None."""
    if roi is None or roi.is_empty():
        return img.data, Rect(0, 0, img.width, img.height)
    r = roi.clip(img.width, img.height)
    x, y, w, h = r.as_int()
    if w <= 0 or h <= 0:
        raise NodeError("ROI 在图像范围之外")
    return img.data[y:y + h, x:x + w], Rect(x, y, w, h)


def odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def parse_floats(text: str, n: int, default: float) -> list[float]:
    parts = [p for p in str(text).replace(";", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts] if parts else []
    if len(vals) == 1:
        vals = vals * n
    if len(vals) != n:
        return [default] * n
    return vals

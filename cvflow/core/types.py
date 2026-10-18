"""Core data types exchanged between nodes.

Everything here is plain Python / numpy so the core stays Qt-free and can run
headless on a production PC or inside a test.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class DataType(str, enum.Enum):
    """Port data types. Links are only allowed between compatible types."""

    IMAGE = "image"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"
    POINT = "point"
    RECT = "rect"
    CIRCLE = "circle"
    LINE = "line"
    CONTOURS = "contours"
    LIST = "list"
    DICT = "dict"
    TENSOR = "tensor"
    ANY = "any"


_NUMERIC = {DataType.INT, DataType.FLOAT, DataType.BOOL}


def types_compatible(src: DataType, dst: DataType) -> bool:
    """Return True if an output of type ``src`` may be linked into an input of type ``dst``."""
    if src == dst or DataType.ANY in (src, dst):
        return True
    if src in _NUMERIC and dst in _NUMERIC:
        return True
    if dst == DataType.STRING and src in _NUMERIC:
        return True
    if dst == DataType.LIST and src == DataType.CONTOURS:
        return True
    return False


@dataclass
class Image:
    """An image plus acquisition metadata.

    ``data`` is a numpy array in OpenCV convention: HxW (gray) or HxWx3 (BGR), uint8
    unless a node deliberately produces float data (e.g. a heat map).
    """

    data: np.ndarray
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.data.shape[2])

    @property
    def is_gray(self) -> bool:
        return self.data.ndim == 2

    @property
    def origin(self) -> tuple[int, int]:
        """这张图左上角落在**原始帧**（相机取到的整幅）里的哪个位置。

        固定 ROI 裁剪（采集节点或「裁剪 ROI」节点）会在 ``meta["roi"] = [x, y, w, h]`` 里记下
        裁的是原始帧的哪一块，连着裁两次也是相对最原始那帧累加的。没裁过就是 (0, 0)。

        约定：节点画的叠加层用送进它那张图的坐标（和面板上显示的图对得齐）；
        节点输出给下游 / PLC 的坐标用原始帧坐标，差的就是这个原点。
        """
        r = self.meta.get("roi")
        if r and len(r) >= 4:
            return int(round(float(r[0]))), int(round(float(r[1])))
        return 0, 0

    def derive(self, data: np.ndarray) -> "Image":
        """New Image carrying this image's metadata but different pixel data."""
        return Image(data=data, frame_id=self.frame_id, timestamp=self.timestamp,
                     source=self.source, meta=dict(self.meta))

    def copy(self) -> "Image":
        return self.derive(self.data.copy())


@dataclass
class Point:
    x: float
    y: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, d: dict) -> "Point":
        return cls(float(d["x"]), float(d["y"]))


@dataclass
class Rect:
    """Axis-aligned (angle 0) or rotated rectangle. ``angle`` in degrees, about the centre."""

    x: float
    y: float
    w: float
    h: float
    angle: float = 0.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def as_int(self) -> tuple[int, int, int, int]:
        return int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h))

    def clip(self, width: int, height: int) -> "Rect":
        x0 = min(max(0.0, self.x), float(width))
        y0 = min(max(0.0, self.y), float(height))
        x1 = min(max(0.0, self.x + self.w), float(width))
        y1 = min(max(0.0, self.y + self.h), float(height))
        return Rect(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0), self.angle)

    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "angle": self.angle}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rect | None":
        if not d:
            return None
        return cls(float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"]), float(d.get("angle", 0.0)))


@dataclass
class Circle:
    cx: float
    cy: float
    r: float

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "r": self.r}


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class Overlay:
    """A drawable result for the image viewer (kept as plain data; the UI decides how to paint).

    kind: "rect" | "circle" | "line" | "polygon" | "points" | "text" | "contours"
    geometry per kind:
      rect     -> {"x","y","w","h","angle"}
      circle   -> {"cx","cy","r"}
      line     -> {"x1","y1","x2","y2"}
      polygon  -> {"points": [[x,y],...]}
      points   -> {"points": [[x,y],...]}
      text     -> {"x","y","text"}
      contours -> {"contours": [np.ndarray Nx1x2, ...]}
    color: "#rrggbb"
    """

    kind: str
    geometry: dict[str, Any]
    color: str = "#00ff00"
    width: float = 1.5
    label: str = ""
    group: str = ""          # 多输入节点用它标记这个叠加层属于哪一路输入，单输入节点留空

    @staticmethod
    def rect(r: Rect, color: str = "#00ff00", label: str = "") -> "Overlay":
        return Overlay("rect", r.to_dict(), color, label=label)

    @staticmethod
    def circle(c: Circle, color: str = "#00ff00", label: str = "") -> "Overlay":
        return Overlay("circle", c.to_dict(), color, label=label)

    @staticmethod
    def line(ln: Line, color: str = "#00ff00", label: str = "") -> "Overlay":
        return Overlay("line", ln.to_dict(), color, label=label)

    @staticmethod
    def text(x: float, y: float, text: str, color: str = "#ffff00") -> "Overlay":
        return Overlay("text", {"x": x, "y": y, "text": text}, color)

    @staticmethod
    def points(pts, color: str = "#ff00ff") -> "Overlay":
        return Overlay("points", {"points": [[float(p[0]), float(p[1])] for p in pts]}, color)

    @staticmethod
    def contours(cnts, color: str = "#00ffff") -> "Overlay":
        return Overlay("contours", {"contours": list(cnts)}, color)

    def translated(self, dx: float, dy: float) -> "Overlay":
        """整体平移后的副本。把画在裁剪小图上的叠加层搬到原始帧（或别的裁法的图）上显示时用。"""
        if not dx and not dy:
            return self
        g = dict(self.geometry)
        k = self.kind
        if k in ("rect", "text"):
            g["x"], g["y"] = g["x"] + dx, g["y"] + dy
        elif k == "circle":
            g["cx"], g["cy"] = g["cx"] + dx, g["cy"] + dy
        elif k == "line":
            g["x1"], g["y1"], g["x2"], g["y2"] = g["x1"] + dx, g["y1"] + dy, g["x2"] + dx, g["y2"] + dy
        elif k in ("polygon", "points"):
            g["points"] = [[float(x) + dx, float(y) + dy] for x, y in g["points"]]
        elif k == "contours":
            off = np.array([dx, dy], dtype=np.float32)
            g["contours"] = [np.asarray(c) + off for c in g["contours"]]
        return Overlay(k, g, self.color, self.width, self.label, self.group)


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of node outputs into JSON-serialisable values (for logs / comm)."""
    if isinstance(value, Image):
        return {"image": f"{value.width}x{value.height}x{value.channels}", "frame_id": value.frame_id}
    if isinstance(value, (Point, Rect, Circle, Line)):
        return value.to_dict()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else {"ndarray": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

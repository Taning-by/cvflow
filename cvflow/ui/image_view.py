"""Zoomable image viewer with overlay rendering and interactive ROI drawing."""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView)

from ..core.types import Image, Overlay, Rect
from .i18n import tr


def to_qimage(data: np.ndarray) -> QImage:
    """numpy (gray / BGR / BGRA uint8, or float) -> QImage. Caller keeps the returned image alive."""
    if data.dtype != np.uint8:
        d = data.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        data = ((d - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(d, dtype=np.uint8)
    data = np.ascontiguousarray(data)
    h, w = data.shape[:2]
    if data.ndim == 2:
        img = QImage(data.data, w, h, data.strides[0], QImage.Format_Grayscale8)
    elif data.shape[2] == 3:
        img = QImage(data.data, w, h, data.strides[0], QImage.Format_BGR888)
    else:
        img = QImage(data.data, w, h, data.strides[0], QImage.Format_ARGB32)
    return img.copy()  # detach from the numpy buffer


class ImageView(QGraphicsView):
    pixel_info = Signal(str)
    roi_drawn = Signal(object)  # Rect

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix = QGraphicsPixmapItem()
        self._pix.setTransformationMode(Qt.FastTransformation)
        self._scene.addItem(self._pix)
        self._overlay_items: list[QGraphicsItem] = []
        self._image: Image | None = None
        self._roi_cb: Callable[[Rect], None] | None = None
        self._roi_start: QPointF | None = None
        self._roi_item: QGraphicsRectItem | None = None
        self._fit_pending = True
        self._rpan: QPointF | None = None
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(QColor("#121418")))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setMouseTracking(True)

    # ---- content ----
    @property
    def image(self) -> Image | None:
        return self._image

    def set_image(self, image: Image | None) -> None:
        self._image = image
        if image is None:
            self._pix.setPixmap(QPixmap())
            return
        was_empty = self._pix.pixmap().isNull()
        size_changed = self._pix.pixmap().size() != (image.width, image.height)
        self._pix.setPixmap(QPixmap.fromImage(to_qimage(image.data)))
        self._scene.setSceneRect(QRectF(0, 0, image.width, image.height))
        if was_empty or self._fit_pending or size_changed:
            self.fit()
            self._fit_pending = False

    def clear(self) -> None:
        self.set_image(None)
        self.set_overlays([])

    def fit(self) -> None:
        if not self._pix.pixmap().isNull():
            self.fitInView(self._pix, Qt.KeepAspectRatio)

    def set_overlays(self, overlays: list[Overlay]) -> None:
        for it in self._overlay_items:
            self._scene.removeItem(it)
        self._overlay_items.clear()
        for ov in overlays:
            for it in self._make_items(ov):
                it.setZValue(10)
                self._scene.addItem(it)
                self._overlay_items.append(it)

    def _make_items(self, ov: Overlay) -> list[QGraphicsItem]:
        pen = QPen(QColor(ov.color), ov.width)
        pen.setCosmetic(True)
        g = ov.geometry
        items: list[QGraphicsItem] = []
        try:
            if ov.kind == "rect":
                r = QGraphicsRectItem(QRectF(g["x"], g["y"], g["w"], g["h"]))
                if g.get("angle"):
                    r.setTransformOriginPoint(g["x"] + g["w"] / 2, g["y"] + g["h"] / 2)
                    r.setRotation(g["angle"])
                r.setPen(pen)
                items.append(r)
                if ov.label:
                    items.append(self._text(g["x"], g["y"] - self._label_h(), ov.label, ov.color))
            elif ov.kind == "circle":
                c = QGraphicsEllipseItem(QRectF(g["cx"] - g["r"], g["cy"] - g["r"], 2 * g["r"], 2 * g["r"]))
                c.setPen(pen)
                items.append(c)
                if ov.label:
                    items.append(self._text(g["cx"] - g["r"], g["cy"] - g["r"] - self._label_h(), ov.label, ov.color))
            elif ov.kind == "line":
                ln = QGraphicsLineItem(g["x1"], g["y1"], g["x2"], g["y2"])
                ln.setPen(pen)
                items.append(ln)
            elif ov.kind == "polygon":
                path = QPainterPath()
                path.addPolygon(QPolygonF([QPointF(x, y) for x, y in g["points"]]))
                path.closeSubpath()
                p = QGraphicsPathItem(path)
                p.setPen(pen)
                items.append(p)
                if ov.label and g["points"]:
                    items.append(self._text(g["points"][0][0], g["points"][0][1] - self._label_h(), ov.label, ov.color))
            elif ov.kind == "points":
                path = QPainterPath()
                for x, y in g["points"]:
                    path.moveTo(x - 5, y); path.lineTo(x + 5, y)
                    path.moveTo(x, y - 5); path.lineTo(x, y + 5)
                p = QGraphicsPathItem(path)
                p.setPen(pen)
                items.append(p)
            elif ov.kind == "text":
                items.append(self._text(g["x"], g["y"] - self._label_h(), str(g["text"]), ov.color))
            elif ov.kind == "contours":
                path = QPainterPath()
                for cnt in g["contours"]:
                    pts = np.asarray(cnt).reshape(-1, 2)
                    if len(pts) == 0:
                        continue
                    path.moveTo(float(pts[0][0]), float(pts[0][1]))
                    for x, y in pts[1:]:
                        path.lineTo(float(x), float(y))
                    path.closeSubpath()
                p = QGraphicsPathItem(path)
                p.setPen(pen)
                items.append(p)
        except (KeyError, TypeError, ValueError):
            pass
        return items

    def _text(self, x: float, y: float, text: str, color: str) -> QGraphicsItem:
        """叠加文字：字号随图像尺寸缩放（像素单位），带半透明深色底，便于在任何底色上阅读。"""
        h = self._image.height if self._image is not None else 480
        px = max(9, int(h / 36))
        t = QGraphicsSimpleTextItem(text)
        t.setBrush(QBrush(QColor(color)))
        f = QFont("Sans")
        f.setPixelSize(px)
        f.setBold(True)
        t.setFont(f)
        br = t.boundingRect()
        bg = QGraphicsRectItem(QRectF(0, 0, br.width() + px * 0.5, br.height() + px * 0.2))
        bg.setBrush(QBrush(QColor(0, 0, 0, 150)))
        bg.setPen(Qt.NoPen)
        bg.setPos(x, max(0.0, y))
        t.setParentItem(bg)
        t.setPos(px * 0.25, px * 0.1)
        return bg

    def _label_h(self) -> float:
        h = self._image.height if self._image is not None else 480
        return max(9, int(h / 36)) * 1.4

    # ---- ROI editing ----
    def begin_roi_edit(self, callback: Callable[[Rect], None]) -> None:
        self._roi_cb = callback
        self.setDragMode(QGraphicsView.NoDrag)
        self.setCursor(Qt.CrossCursor)
        self.pixel_info.emit(tr("Draw a rectangle on the image (Esc to cancel)"))

    def cancel_roi_edit(self) -> None:
        self._roi_cb = None
        self._roi_start = None
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.unsetCursor()

    @property
    def roi_editing(self) -> bool:
        return self._roi_cb is not None

    # ---- events ----
    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_pending:
            self.fit()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._rpan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self._roi_cb is not None and event.button() == Qt.LeftButton:
            self._roi_start = self.mapToScene(event.position().toPoint())
            self._roi_item = QGraphicsRectItem(QRectF(self._roi_start, self._roi_start))
            pen = QPen(QColor("#ffa500"), 1.5, Qt.DashLine)
            pen.setCosmetic(True)
            self._roi_item.setPen(pen)
            self._roi_item.setZValue(20)
            self._scene.addItem(self._roi_item)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._rpan is not None:
            d = event.position() - self._rpan
            self._rpan = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        pos = self.mapToScene(event.position().toPoint())
        if self._roi_start is not None and self._roi_item is not None:
            self._roi_item.setRect(QRectF(self._roi_start, pos).normalized())
        if self._image is not None:
            x, y = int(pos.x()), int(pos.y())
            if 0 <= x < self._image.width and 0 <= y < self._image.height:
                v = self._image.data[y, x]
                val = int(v) if np.ndim(v) == 0 else tuple(int(c) for c in v)
                self.pixel_info.emit(f"x={x} y={y} value={val}")
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._rpan is not None and event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._rpan = None
            self.unsetCursor()
            return
        if self._roi_start is not None and self._roi_item is not None and event.button() == Qt.LeftButton:
            r = self._roi_item.rect()
            cb = self._roi_cb
            self.cancel_roi_edit()
            if r.width() >= 2 and r.height() >= 2:
                rect = Rect(round(r.x(), 1), round(r.y(), 1), round(r.width(), 1), round(r.height(), 1))
                self.roi_drawn.emit(rect)
                if cb is not None:
                    cb(rect)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape and self._roi_cb is not None:
            self.cancel_roi_edit()
            return
        super().keyPressEvent(event)

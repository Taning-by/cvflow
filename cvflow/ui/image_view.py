"""Zoomable image viewer with overlay rendering and interactive ROI drawing."""
from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView, QLabel)

from ..core.types import Image, Overlay, Rect
from .i18n import tr
from .theme import C, ui_font


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
    zoom_changed = Signal(float)
    MIN_ZOOM, MAX_ZOOM = 0.02, 64.0

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
        self._roi_move: tuple[int, int] | None = None    # 固定尺寸模式：(宽, 高)，只拖位置
        self._roi_grab: tuple[float, float] | None = None
        self._roi_orig: Rect | None = None               # 进入拖动模式时的框，Esc 用它还原
        self._fit_pending = True
        self._fit_mode = True          # 处于"适配"状态：窗格尺寸变化时自动重新适配
        self._rpan: QPointF | None = None
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # 查看器底色是中性深灰：图像本身不加任何滤镜或色调，深灰只是让图像边界清楚。
        self.setBackgroundBrush(QBrush(QColor(C["viewer_bg"])))
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setMouseTracking(True)
        # 没有图像时在查看器中央给一句说明，而不是留一块空的深灰
        self._hint = QLabel("", self)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("background: transparent; color: #9AA6B2; font-size: 13px;")
        self._hint.hide()

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)
        self._hint.setVisible(bool(text))
        self._place_hint()

    def _place_hint(self) -> None:
        m = 24
        self._hint.setGeometry(m, 0, max(0, self.viewport().width() - 2 * m), self.viewport().height())
        self._hint.raise_()

    # ---- content ----
    @property
    def image(self) -> Image | None:
        return self._image

    def set_image(self, image: Image | None) -> None:
        self._image = image
        if image is not None:
            self.set_hint("")
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
        self._fit_mode = True
        if not self._pix.pixmap().isNull():
            self.fitInView(self._pix, Qt.KeepAspectRatio)     # 始终等比，不拉伸图像
            self.zoom_changed.emit(self.zoom())

    # ---- 缩放 ----
    def zoom(self) -> float:
        return float(self.transform().m11())

    def set_zoom(self, value: float) -> None:
        value = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(value)))
        self.setTransform(self.transform().fromScale(value, value))
        self._fit_pending = False
        self._fit_mode = False
        self.zoom_changed.emit(self.zoom())

    def actual_size(self) -> None:
        """1:1 显示（一个图像像素对应一个逻辑像素），检查细节时用。"""
        self.set_zoom(1.0)

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
        f = ui_font(px, True)
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
        self._roi_move = None
        self._roi_grab = None
        self._roi_orig = None
        if self._roi_item is not None:
            self._scene.removeItem(self._roi_item)
            self._roi_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.unsetCursor()

    # ---- 固定尺寸 ROI：只拖位置 ----
    def begin_roi_move(self, size, rect: Rect | None, callback: Callable[[Rect], None]) -> None:
        """尺寸锁死的框：拖动改位置、方向键微调（Shift 十倍），回车结束，Esc 还原并结束。

        尺寸不给拖——它是送进模型的张量尺寸，拖是拖不出一个正确数字的，只能在参数里填。
        改动是即时生效的：松手或按一下方向键就写回参数，面板上的数值同步跟着变。
        """
        w, h = max(1, int(size[0])), max(1, int(size[1]))
        self._roi_cb = callback
        self._roi_move = (w, h)
        self._roi_orig = rect
        iw = self._image.width if self._image is not None else w
        ih = self._image.height if self._image is not None else h
        x, y = ((rect.cx - w / 2, rect.cy - h / 2) if rect is not None and not rect.is_empty()
                else ((iw - w) / 2, (ih - h) / 2))
        self._roi_item = QGraphicsRectItem(QRectF(x, y, w, h))
        pen = QPen(QColor(C["roi"]), 1.8)
        pen.setCosmetic(True)
        self._roi_item.setPen(pen)
        self._roi_item.setZValue(20)
        self._scene.addItem(self._roi_item)
        self._move_roi_to(x, y)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setCursor(Qt.SizeAllCursor)
        self.setFocus(Qt.OtherFocusReason)
        self.pixel_info.emit(tr("Drag to move the ROI; arrow keys nudge, Enter to finish, Esc to revert"))

    def _move_roi_to(self, x: float, y: float) -> None:
        if self._roi_item is None or self._roi_move is None:
            return
        w, h = self._roi_move
        if self._image is not None:                       # 框不许出图：出了图就是在喂黑边
            x = min(max(0.0, x), max(0.0, self._image.width - w))
            y = min(max(0.0, y), max(0.0, self._image.height - h))
        self._roi_item.setRect(QRectF(round(x), round(y), w, h))
        self.pixel_info.emit(f"ROI  x {round(x):.0f}  y {round(y):.0f}  {w} × {h}")

    def _commit_roi_move(self) -> None:
        if self._roi_item is None or self._roi_cb is None:
            return
        r = self._roi_item.rect()
        rect = Rect(round(r.x(), 1), round(r.y(), 1), round(r.width(), 1), round(r.height(), 1))
        self.roi_drawn.emit(rect)
        self._roi_cb(rect)

    def _nudge_roi(self, dx: float, dy: float) -> None:
        if self._roi_item is None:
            return
        r = self._roi_item.rect()
        self._move_roi_to(r.x() + dx, r.y() + dy)
        self._commit_roi_move()

    @property
    def roi_editing(self) -> bool:
        return self._roi_cb is not None

    # ---- events ----
    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if self.MIN_ZOOM <= self.zoom() * factor <= self.MAX_ZOOM:
            self.scale(factor, factor)
            self._fit_pending = False
            self._fit_mode = False
            self.zoom_changed.emit(self.zoom())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_hint()
        if self._fit_pending or self._fit_mode:   # 还没手动缩放过：跟着窗格大小保持适配
            self.fit()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._rpan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if self._roi_move is not None and event.button() == Qt.LeftButton:
            p = self.mapToScene(event.position().toPoint())
            r = self._roi_item.rect() if self._roi_item is not None else QRectF()
            if not r.contains(p):                          # 按在框外：把框挪到鼠标下，再接着拖
                self._move_roi_to(p.x() - r.width() / 2, p.y() - r.height() / 2)
                r = self._roi_item.rect()
            self._roi_grab = (p.x() - r.x(), p.y() - r.y())
            return
        if self._roi_cb is not None and event.button() == Qt.LeftButton:
            self._roi_start = self.mapToScene(event.position().toPoint())
            self._roi_item = QGraphicsRectItem(QRectF(self._roi_start, self._roi_start))
            pen = QPen(QColor(C["roi"]), 1.6, Qt.DashLine)
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
        if self._roi_grab is not None:
            self._move_roi_to(pos.x() - self._roi_grab[0], pos.y() - self._roi_grab[1])
            return
        if self._roi_start is not None and self._roi_item is not None:
            r = QRectF(self._roi_start, pos).normalized()
            self._roi_item.setRect(r)
            self.pixel_info.emit(f"ROI  x {r.x():.0f}  y {r.y():.0f}  {r.width():.0f} × {r.height():.0f}")
            return
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
        if self._roi_grab is not None and event.button() == Qt.LeftButton:
            self._roi_grab = None
            self._commit_roi_move()                        # 松手即生效，但不退出：还能接着微调
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
        if self._roi_move is not None:
            step = 10 if event.modifiers() & Qt.ShiftModifier else 1
            d = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0),
                 Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}.get(event.key())
            if d is not None:
                self._nudge_roi(*d)
                return
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.cancel_roi_edit()                     # 已经即时生效了，这里只是退出拖动
                return
            if event.key() == Qt.Key_Escape:
                orig, cb = self._roi_orig, self._roi_cb
                self.cancel_roi_edit()
                if cb is not None:
                    cb(orig)                               # 还原成进入拖动前的位置（原本就没有则清空）
                return
        if event.key() == Qt.Key_Escape and self._roi_cb is not None:
            self.cancel_roi_edit()
            return
        super().keyPressEvent(event)

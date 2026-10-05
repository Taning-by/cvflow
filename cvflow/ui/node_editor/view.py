"""Graphics view: zoom/pan, context menu, drag & drop from the palette, delete key."""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction, QPainter
from PySide6.QtWidgets import QGraphicsView, QMenu

from ...core.registry import registry
from ..i18n import tr
from .items import NodeItem
from .scene import NodeScene

MIME = "application/x-cvflow-node"


class NodeView(QGraphicsView):
    zoom_changed = Signal(float)
    MIN_ZOOM, MAX_ZOOM, FIT_FLOOR = 0.25, 4.0, 0.6

    def __init__(self, scene: NodeScene, parent=None) -> None:
        super().__init__(scene, parent)
        self._scene = scene
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setAcceptDrops(True)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setContextMenuPolicy(Qt.NoContextMenu)   # 右键菜单由鼠标释放时手动弹出（区分拖动与点击）
        self._fit_mode = True            # 处于"适配"状态：窗格尺寸变化时自动重新适配
        self._mouse_down = False         # 用户正在视图里按着鼠标（点选、框选、拖动节点或连线）
        self._panning = False
        self._pan_start = QPointF()
        self._pan_button = None
        self._pan_moved = False
        self._press_pos = QPointF()

    # ---- zoom / pan ----
    def zoom(self) -> float:
        return float(self.transform().m11())

    def set_zoom(self, value: float) -> None:
        self._fit_mode = False
        value = max(self.MIN_ZOOM, min(self.MAX_ZOOM, value))
        self.setTransform(self.transform().fromScale(value, value))
        self.zoom_changed.emit(value)

    def wheelEvent(self, event) -> None:
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        cur = self.zoom()
        if self.MIN_ZOOM <= cur * factor <= self.MAX_ZOOM:
            self._fit_mode = False
            self.scale(factor, factor)
            self.zoom_changed.emit(self.zoom())

    def is_interacting(self) -> bool:
        """用户正按着鼠标操作视图。这期间任何代码都不该去滚动视图：
        滚动会改变光标与场景的对应关系，接着的一点点移动就会把节点整块拖走。"""
        return self._mouse_down

    def mousePressEvent(self, event) -> None:
        b = event.button()
        self._mouse_down = True
        if b == Qt.RightButton or b == Qt.MiddleButton or (b == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self._panning = True
            self._pan_button = b
            self._pan_moved = False
            self._pan_start = event.position()
            self._press_pos = event.position()
            if b != Qt.RightButton:
                self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            d = event.position() - self._pan_start
            self._pan_start = event.position()
            if (event.position() - self._press_pos).manhattanLength() > 4:
                if not self._pan_moved:
                    self.setCursor(Qt.ClosedHandCursor)
                self._pan_moved = True
            self._fit_mode = False       # 用户自己平移过视图，就别再自动适配
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._mouse_down = False
        if self._panning and event.button() == self._pan_button:
            self._panning = False
            self._pan_button = None
            self.unsetCursor()
            if event.button() == Qt.RightButton and not self._pan_moved:
                self._show_context_menu(event.position().toPoint(), event.globalPosition().toPoint())
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._scene.remove_selected()
            return
        if event.key() == Qt.Key_F and not event.modifiers():
            self.fit_all()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:            # 还没手动缩放过：跟着窗格大小保持适配
            self.fit_all()

    def fit_all(self) -> None:
        """适配全部节点。缩小有下限：节点名与参数摘要必须还能读，读不全时宁可留给用户滚动。"""
        self._fit_mode = True
        rect = self._scene.itemsBoundingRect()
        if rect.isValid():
            self.fitInView(rect.adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)
            z = self.zoom()
            if z > 1.0:
                self.resetTransform()
                self.centerOn(rect.center())
            elif z < self.FIT_FLOOR:
                # 放不下也不再缩小：按下限缩放，并对齐到流程起点（左上），因为流程是从左往右读的
                self.setTransform(self.transform().fromScale(self.FIT_FLOOR, self.FIT_FLOOR))
                vis = self.mapToScene(self.viewport().rect()).boundingRect()
                self.centerOn(rect.left() + vis.width() / 2 - 20, rect.center().y())
        self.zoom_changed.emit(self.zoom())

    # ---- context menu ----
    def _show_context_menu(self, pos, global_pos) -> None:
        scene_pos = self.mapToScene(pos)
        menu = QMenu(self)
        add = menu.addMenu(tr("Add node"))
        for cat, classes in registry.categories().items():
            sub = add.addMenu(tr(cat))
            for cls in classes:
                act = QAction(tr(cls.label), sub)
                act.setToolTip(tr(cls.description))
                act.triggered.connect(lambda checked=False, t=cls.type_id, p=scene_pos: self._scene.add_node(t, p))
                sub.addAction(act)
        item = self.itemAt(pos)
        node_item = item if isinstance(item, NodeItem) else (item.parentItem() if item and isinstance(item.parentItem(), NodeItem) else None)
        if node_item is not None:
            menu.addSeparator()
            node = node_item.node
            tog = menu.addAction(tr("Disable") if node.enabled else tr("Enable"))
            tog.triggered.connect(lambda: self._scene.set_node_enabled(node.id, not node.enabled))
            dup = menu.addAction(tr("Duplicate"))
            dup.triggered.connect(lambda: self._duplicate(node))
        if self._scene.selectedItems():
            menu.addSeparator()
            menu.addAction(tr("Delete selected")).triggered.connect(self._scene.remove_selected)
        menu.addSeparator()
        menu.addAction(tr("Fit view (F)")).triggered.connect(self.fit_all)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        menu.popup(global_pos)   # 非阻塞弹出

    def _duplicate(self, node) -> None:
        new = self._scene.add_node(node.type_id, QPointF(node.position[0] + 30, node.position[1] + 30))
        if new is not None:
            new.values = {k: v for k, v in node.values.items()}
            new.name = ""                                   # 先清空，避免自己占用一个编号
            new.name = self._scene.graph.unique_name(node.name)
            self._scene.node_items[new.id].update()

    # ---- drag & drop from palette ----
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME):
            type_id = bytes(event.mimeData().data(MIME)).decode()
            self._scene.add_node(type_id, self.mapToScene(event.position().toPoint()))
            event.acceptProposedAction()

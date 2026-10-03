"""Graphics view: zoom/pan, context menu, drag & drop from the palette, delete key."""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QAction, QPainter
from PySide6.QtWidgets import QGraphicsView, QMenu

from ...core.registry import registry
from ..i18n import tr
from .items import NodeItem
from .scene import NodeScene

MIME = "application/x-cvflow-node"


class NodeView(QGraphicsView):
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
        self._panning = False
        self._pan_start = QPointF()
        self._pan_button = None
        self._pan_moved = False
        self._press_pos = QPointF()

    # ---- zoom / pan ----
    def wheelEvent(self, event) -> None:
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        cur = self.transform().m11()
        if 0.2 <= cur * factor <= 3.0:
            self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        b = event.button()
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
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
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

    def fit_all(self) -> None:
        rect = self._scene.itemsBoundingRect()
        if rect.isValid():
            self.fitInView(rect.adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)
            if self.transform().m11() > 1.0:
                self.resetTransform()

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

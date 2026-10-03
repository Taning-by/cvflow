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
        self._panning = False
        self._pan_start = QPointF()

    # ---- zoom / pan ----
    def wheelEvent(self, event) -> None:
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        cur = self.transform().m11()
        if 0.2 <= cur * factor <= 3.0:
            self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            d = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self.unsetCursor()
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
    def contextMenuEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())
        menu = QMenu(self)
        add = menu.addMenu(tr("Add node"))
        for cat, classes in registry.categories().items():
            sub = add.addMenu(tr(cat))
            for cls in classes:
                act = QAction(tr(cls.label), sub)
                act.setToolTip(tr(cls.description))
                act.triggered.connect(lambda checked=False, t=cls.type_id, p=scene_pos: self._scene.add_node(t, p))
                sub.addAction(act)
        item = self.itemAt(event.pos())
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
        menu.exec(event.globalPos())

    def _duplicate(self, node) -> None:
        new = self._scene.add_node(node.type_id, QPointF(node.position[0] + 30, node.position[1] + 30))
        if new is not None:
            new.values = {k: v for k, v in node.values.items()}
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

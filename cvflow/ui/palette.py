"""Node palette: categories tree with filter, drag source for the editor."""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QFont
from PySide6.QtWidgets import QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..core.registry import registry
from .i18n import tr
from .node_editor.view import MIME
from .theme import C, make_icon


class _Tree(QTreeWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)

    def startDrag(self, actions) -> None:
        item = self.currentItem()
        if item is None or not item.data(0, Qt.UserRole):
            return
        mime = QMimeData()
        mime.setData(MIME, QByteArray(str(item.data(0, Qt.UserRole)).encode()))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class NodePalette(QWidget):
    node_activated = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(tr("Filter nodes…"))
        self.filter.textChanged.connect(self._apply_filter)
        self.tree = _Tree()
        self.tree.setIndentation(14)
        self.tree.setIconSize(QSize(12, 12))
        self.tree.setFrameShape(QTreeWidget.NoFrame)
        self.tree.itemDoubleClicked.connect(self._activated)
        lay.addWidget(self.filter)
        lay.addWidget(self.tree)
        self.reload()

    def reload(self) -> None:
        self.tree.clear()
        for cat, classes in registry.categories().items():
            top = QTreeWidgetItem([tr(cat)])
            top.setFlags(top.flags() & ~Qt.ItemIsDragEnabled)
            f = QFont(); f.setBold(True)
            top.setFont(0, f)
            top.setForeground(0, QColor(C["muted"]))
            top.setIcon(0, make_icon("dot", classes[0].color if classes else C["muted"], 12))
            self.tree.addTopLevelItem(top)
            for cls in classes:
                it = QTreeWidgetItem([tr(cls.label)])
                it.setData(0, Qt.UserRole, cls.type_id)
                it.setToolTip(0, f"{cls.type_id}\n{tr(cls.description)}")
                it.setIcon(0, make_icon("dot", cls.color, 12))
                top.addChild(it)
            top.setExpanded(True)
        self._apply_filter(self.filter.text())

    def _apply_filter(self, text: str) -> None:
        t = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            visible = 0
            for j in range(top.childCount()):
                ch = top.child(j)
                show = not t or t in ch.text(0).lower() or t in str(ch.data(0, Qt.UserRole)).lower() or t in ch.toolTip(0).lower()
                ch.setHidden(not show)
                visible += show
            top.setHidden(visible == 0)

    def _activated(self, item, col) -> None:
        tid = item.data(0, Qt.UserRole)
        if tid:
            self.node_activated.emit(str(tid))

"""算法工具箱：可搜索的分类树，同时是节点编辑器的拖放来源。"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDrag
from PySide6.QtWidgets import QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..core.registry import registry
from .i18n import tr
from .node_editor.view import MIME
from .theme import C, M, make_icon, ui_font


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
        lay.setContentsMargins(M["gap"], M["gap"], M["gap"], M["gap"])
        lay.setSpacing(M["gap"])
        self.filter = QLineEdit()
        self.filter.setPlaceholderText(tr("Filter nodes…"))
        self.filter.setClearButtonEnabled(True)
        self.filter.addAction(make_icon("search", C["faint"], 16), QLineEdit.LeadingPosition)
        self.filter.setFixedHeight(M["ctl_h"])
        self.filter.textChanged.connect(self._apply_filter)
        self.tree = _Tree()
        self.tree.setIndentation(12)
        self.tree.setIconSize(QSize(10, 10))
        self.tree.setFrameShape(QTreeWidget.NoFrame)
        self.tree.setUniformRowHeights(True)
        self.tree.setExpandsOnDoubleClick(False)       # 双击＝加入流程，不要顺带折叠分类
        self.tree.itemDoubleClicked.connect(self._activated)
        self.empty_hint = None
        lay.addWidget(self.filter)
        lay.addWidget(self.tree, 1)
        self.reload()

    def reload(self) -> None:
        self.tree.clear()
        cat_font = ui_font(M["font_s"], True)
        item_font = ui_font(M["font"])
        for cat, classes in registry.categories().items():
            top = QTreeWidgetItem([f"{tr(cat)}  ({len(classes)})"])
            top.setFlags(top.flags() & ~Qt.ItemIsDragEnabled & ~Qt.ItemIsSelectable)
            top.setFont(0, cat_font)
            top.setForeground(0, QColor(C["muted"]))
            self.tree.addTopLevelItem(top)
            for cls in classes:
                it = QTreeWidgetItem([tr(cls.label)])
                it.setData(0, Qt.UserRole, cls.type_id)
                it.setFont(0, item_font)
                it.setSizeHint(0, QSize(0, 24))
                it.setToolTip(0, f"{tr(cls.label)}　{cls.type_id}\n{tr(cls.description)}\n\n双击加入流程，或拖到画布上")
                it.setIcon(0, make_icon("dot", cls.color, 10))
                top.addChild(it)
            top.setExpanded(True)
        self._apply_filter(self.filter.text())

    def _apply_filter(self, text: str) -> None:
        t = text.strip().lower()
        shown = 0
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            visible = 0
            for j in range(top.childCount()):
                ch = top.child(j)
                show = not t or t in ch.text(0).lower() or t in str(ch.data(0, Qt.UserRole)).lower() or t in ch.toolTip(0).lower()
                ch.setHidden(not show)
                visible += show
            top.setHidden(visible == 0)
            if visible and t:
                top.setExpanded(True)
            shown += visible
        self.tree.setToolTip("" if shown else tr("No node matches the filter."))

    def _activated(self, item, col) -> None:
        tid = item.data(0, Qt.UserRole)
        if tid:
            self.node_activated.emit(str(tid))

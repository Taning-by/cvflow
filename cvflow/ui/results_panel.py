"""显示最近一次运行：状态横幅、发布的结果、每个节点的输出与错误。

表格里的文字可以选中、复制（Ctrl+C 或右键菜单），悬停能看到未截断的完整内容，
便于把报错信息贴到别处排查。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QHBoxLayout, QHeaderView, QLabel, QMenu,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.engine import RunResult
from ..core.graph import Graph
from ..core.node import NodeStatus
from ..core.types import to_jsonable
from .i18n import tr
from .theme import C, M, STATUS_COLORS, STATUS_GLYPH, STATUS_TEXT, tabular, ui_font

FULL_TEXT = Qt.UserRole + 1          # 单元格的完整文本，显示可能被截断，复制用这个
MAX_SHOWN = 200


def _texts(v) -> tuple[str, str]:
    """返回 (显示文本, 完整文本)。显示保持简洁，复制与悬停给完整值。"""
    j = to_jsonable(v)
    if isinstance(j, float):
        return f"{j:.4g}", repr(j)
    full = str(j)
    return (full if len(full) <= MAX_SHOWN else full[: MAX_SHOWN - 1] + "…"), full


def _cell(item: QTreeWidgetItem, col: int, value, raw: bool = False) -> None:
    """设置一格：显示可能截断，完整文本存起来供复制与悬停查看。``raw`` 表示 value 已经是字符串。"""
    shown, full = (value if len(value) <= MAX_SHOWN else value[: MAX_SHOWN - 1] + "…", value) if raw else _texts(value)
    item.setData(col, FULL_TEXT, full)
    item.setText(col, shown)
    if full:
        item.setToolTip(col, full)


class _ResultTree(QTreeWidget):
    """支持 Ctrl+C 与右键复制的结果表。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    # ---- 取文本 ----
    @staticmethod
    def _text(item: QTreeWidgetItem, col: int) -> str:
        return item.data(col, FULL_TEXT) or item.text(col)

    def _row_text(self, item: QTreeWidgetItem) -> str:
        cols = [self._text(item, c) for c in range(self.columnCount())]
        while cols and not cols[-1]:
            cols.pop()
        return "\t".join(cols)

    def selection_text(self) -> str:
        items = self.selectedItems() or ([self.currentItem()] if self.currentItem() else [])
        return "\n".join(self._row_text(it) for it in items)

    def all_text(self) -> str:
        lines = []

        def walk(item: QTreeWidgetItem, depth: int) -> None:
            lines.append("    " * depth + self._row_text(item))
            for i in range(item.childCount()):
                walk(item.child(i), depth + 1)
        for i in range(self.topLevelItemCount()):
            walk(self.topLevelItem(i), 0)
        return "\n".join(lines)

    # ---- 复制 ----
    @staticmethod
    def _to_clipboard(text: str) -> None:
        if text:
            QApplication.clipboard().setText(text)

    def copy_selection(self) -> None:
        self._to_clipboard(self.selection_text())

    def copy_all(self) -> None:
        self._to_clipboard(self.all_text())

    def copy_cell(self, item: QTreeWidgetItem, col: int) -> None:
        self._to_clipboard(self._text(item, col))

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            return
        if event.matches(QKeySequence.SelectAll):
            self.selectAll()
            return
        super().keyPressEvent(event)

    def _menu(self, pos) -> None:
        item = self.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            col = self.columnAt(pos.x())
            cell = self._text(item, col)
            act = menu.addAction("复制此格")
            act.setEnabled(bool(cell))
            act.triggered.connect(lambda: self.copy_cell(item, col))
            menu.addAction("复制该行").triggered.connect(lambda: self._to_clipboard(self._row_text(item)))
            if len(self.selectedItems()) > 1:
                menu.addAction(f"复制选中的 {len(self.selectedItems())} 行").triggered.connect(self.copy_selection)
            menu.addSeparator()
        menu.addAction("复制全部结果").triggered.connect(self.copy_all)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        menu.popup(self.viewport().mapToGlobal(pos))


class ResultsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(M["gap"], M["gap"], M["gap"], M["gap"])
        lay.setSpacing(M["gap"])
        self._last: tuple[RunResult | None, Graph | None] = (None, None)

        head = QHBoxLayout()
        head.setSpacing(M["gap"])
        self.banner = QLabel(tr("no run yet"))
        self.banner.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.banner.setTextInteractionFlags(Qt.TextSelectableByMouse)   # 横幅里的错误也要能选中复制
        self.banner.setCursor(Qt.IBeamCursor)
        self.banner.setFont(tabular(ui_font(M["font"], True)))
        self._banner_style(C["panel2"], C["muted"], C["border"])
        self.only_problems = QCheckBox(tr("Problems only"))
        self.only_problems.setToolTip(tr("Hide nodes that finished OK"))
        self.only_problems.toggled.connect(lambda _: self.show_result(*self._last))
        head.addWidget(self.banner, 1)
        head.addWidget(self.only_problems)

        self.tree = _ResultTree()
        self.tree.setHeaderLabels([tr("Name"), tr("Value"), tr("Status / ms")])
        self.tree.setColumnWidth(0, 220)
        self.tree.setColumnWidth(1, 420)
        # 名称与值固定在左侧，状态紧跟其后：视线不用横跨整屏去找状态
        self.tree.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Interactive)
        self.tree.header().setStretchLastSection(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.setToolTip("选中后 Ctrl+C 复制，右键可复制单元格、整行或全部")
        lay.addLayout(head)
        lay.addWidget(self.tree)

    def _banner_style(self, bg: str, fg: str, border: str) -> None:
        self.banner.setStyleSheet(f"font-size:{M['font']}px; font-weight:600; padding:5px 10px; background:{bg}; "
                                  f"border:1px solid {border}; border-radius:{M['radius']}px; color:{fg}")

    def show_result(self, result: RunResult | None, graph: Graph | None = None) -> None:
        self._last = (result, graph)
        self.tree.clear()
        if result is None:
            self.banner.setText(tr("no run yet"))
            self._banner_style(C["panel2"], C["muted"], C["border"])
            return
        color = {"OK": C["ok"], "NG": C["ng"], "ERROR": C["warn"]}[result.status_text]
        bg = {"OK": C["ok_bg"], "NG": C["ng_bg"], "ERROR": C["warn_bg"]}[result.status_text]
        extra = f" — {result.error}" if result.error else ""
        self.banner.setText(tr("{status}  ·  run {id}  ·  {ms:.1f} ms").format(
            status=tr(result.status_text), id=result.run_id, ms=result.duration_ms) + extra)
        self.banner.setToolTip(self.banner.text())
        self._banner_style(bg, color, color)
        if result.outputs:
            pub = QTreeWidgetItem()
            _cell(pub, 0, tr("Published"), raw=True)
            for k, v in result.outputs.items():
                child = QTreeWidgetItem()
                _cell(child, 0, k, raw=True)
                _cell(child, 1, v)
                pub.addChild(child)
            self.tree.addTopLevelItem(pub)
            pub.setExpanded(True)
        digits = tabular(ui_font())
        for nr in result.node_results.values():
            if self.only_problems.isChecked() and nr.status in (NodeStatus.OK, NodeStatus.IDLE):
                continue
            it = QTreeWidgetItem()
            _cell(it, 0, nr.node_name, raw=True)
            if nr.status in (NodeStatus.ERROR, NodeStatus.SKIPPED):
                _cell(it, 1, nr.error, raw=True)
                it.setForeground(1, QColor(C["ng"] if nr.status == NodeStatus.ERROR else C["muted"]))
            # 状态列：字形 + 文字 + 颜色，成功 / 不合格 / 失败不会只靠颜色区分
            it.setData(2, FULL_TEXT, f"{STATUS_TEXT.get(nr.status, nr.status.value)}  {nr.time_ms:.2f} ms")
            it.setText(2, f"{STATUS_GLYPH.get(nr.status, '')}  {STATUS_TEXT.get(nr.status, nr.status.value)}"
                          f"   {nr.time_ms:.2f} ms")
            it.setForeground(2, QColor(STATUS_COLORS[nr.status]))
            it.setFont(2, digits)
            it.setFont(1, digits)
            f = it.font(0)
            f.setBold(True)
            it.setFont(0, f)
            for k, v in nr.outputs.items():
                child = QTreeWidgetItem()
                _cell(child, 0, k, raw=True)
                _cell(child, 1, v)
                it.addChild(child)
            self.tree.addTopLevelItem(it)

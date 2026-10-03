"""Shows the last run: status banner, published outputs and per-node outputs."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from ..core.engine import RunResult
from ..core.graph import Graph
from ..core.node import NodeStatus
from ..core.types import to_jsonable
from .i18n import tr
from .theme import STATUS_COLORS


def _fmt(v) -> str:
    j = to_jsonable(v)
    s = str(j) if not isinstance(j, float) else f"{j:.4g}"
    return s if len(s) <= 120 else s[:117] + "…"


class ResultsPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.banner = QLabel(tr("no run yet"))
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setStyleSheet("font-size:16px; font-weight:bold; padding:4px; background:#333; border-radius:4px")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([tr("Name"), tr("Value"), tr("Status / ms")])
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 320)
        lay.addWidget(self.banner)
        lay.addWidget(self.tree)

    def show_result(self, result: RunResult | None, graph: Graph | None = None) -> None:
        self.tree.clear()
        if result is None:
            self.banner.setText(tr("no run yet"))
            self.banner.setStyleSheet("font-size:16px; font-weight:bold; padding:4px; background:#333; border-radius:4px")
            return
        color = {"OK": "#2e7d32", "NG": "#c62828", "ERROR": "#ef6c00"}[result.status_text]
        extra = f" — {result.error}" if result.error else ""
        self.banner.setText(tr("{status}  ·  run {id}  ·  {ms:.1f} ms").format(status=tr(result.status_text), id=result.run_id, ms=result.duration_ms) + extra)
        self.banner.setStyleSheet(f"font-size:16px; font-weight:bold; padding:4px; background:{color}; border-radius:4px")
        if result.outputs:
            pub = QTreeWidgetItem([tr("Published"), "", ""])
            for k, v in result.outputs.items():
                pub.addChild(QTreeWidgetItem([k, _fmt(v), ""]))
            self.tree.addTopLevelItem(pub)
            pub.setExpanded(True)
        for nr in result.node_results.values():
            it = QTreeWidgetItem([nr.node_name, nr.error if nr.status in (NodeStatus.ERROR, NodeStatus.SKIPPED) else "",
                                  f"{tr(nr.status.value)}  {nr.time_ms:.2f}"])
            it.setForeground(2, Qt.white)
            it.setBackground(2, QColor(STATUS_COLORS[nr.status]))
            for k, v in nr.outputs.items():
                it.addChild(QTreeWidgetItem([k, _fmt(v), ""]))
            self.tree.addTopLevelItem(it)

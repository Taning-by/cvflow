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
from .theme import C, STATUS_COLORS, chip_style


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
        self.banner.setStyleSheet(f"font-size:15px; font-weight:600; padding:6px; background:{C['panel2']}; border-radius:8px; color:{C['muted']}")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([tr("Name"), tr("Value"), tr("Status / ms")])
        self.tree.setColumnWidth(0, 200)
        self.tree.setColumnWidth(1, 360)
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(16)
        lay.addWidget(self.banner)
        lay.addWidget(self.tree)

    def show_result(self, result: RunResult | None, graph: Graph | None = None) -> None:
        self.tree.clear()
        if result is None:
            self.banner.setText(tr("no run yet"))
            self.banner.setStyleSheet(f"font-size:15px; font-weight:600; padding:6px; background:{C['panel2']}; border-radius:8px; color:{C['muted']}")
            return
        color = {"OK": "#1f7a3f", "NG": "#a12d2d", "ERROR": "#9a5a00"}[result.status_text]
        extra = f" — {result.error}" if result.error else ""
        self.banner.setText(tr("{status}  ·  run {id}  ·  {ms:.1f} ms").format(status=tr(result.status_text), id=result.run_id, ms=result.duration_ms) + extra)
        self.banner.setStyleSheet(f"font-size:15px; font-weight:600; padding:6px; background:{color}; border-radius:8px; color:white")
        if result.outputs:
            pub = QTreeWidgetItem([tr("Published"), "", ""])
            for k, v in result.outputs.items():
                pub.addChild(QTreeWidgetItem([k, _fmt(v), ""]))
            self.tree.addTopLevelItem(pub)
            pub.setExpanded(True)
        for nr in result.node_results.values():
            it = QTreeWidgetItem([nr.node_name, nr.error if nr.status in (NodeStatus.ERROR, NodeStatus.SKIPPED) else "",
                                  f"●  {tr(nr.status.value)}   {nr.time_ms:.2f} ms"])
            it.setForeground(2, QColor(STATUS_COLORS[nr.status]))
            f = it.font(0); f.setBold(True); it.setFont(0, f)
            if nr.status in (NodeStatus.ERROR, NodeStatus.SKIPPED):
                it.setForeground(1, QColor(C["muted"]))
            for k, v in nr.outputs.items():
                it.addChild(QTreeWidgetItem([k, _fmt(v), ""]))
            self.tree.addTopLevelItem(it)

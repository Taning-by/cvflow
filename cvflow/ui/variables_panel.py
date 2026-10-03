"""Global variables table."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QInputDialog, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from ..core.variables import GlobalVariables


class VariablesPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._vars: GlobalVariables | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QHBoxLayout()
        add = QPushButton("Add")
        rem = QPushButton("Remove")
        add.clicked.connect(self._add)
        rem.clicked.connect(self._remove)
        bar.addWidget(add)
        bar.addWidget(rem)
        bar.addStretch(1)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Value", "Description"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 140)
        self.table.setColumnWidth(1, 220)
        self.table.cellChanged.connect(self._edited)
        lay.addLayout(bar)
        lay.addWidget(self.table)
        self._updating = False

    def set_variables(self, variables: GlobalVariables) -> None:
        self._vars = variables
        self.refresh()

    def refresh(self) -> None:
        if self._vars is None:
            return
        self._updating = True
        entries = self._vars.entries()
        self.table.setRowCount(len(entries))
        for row, (name, e) in enumerate(sorted(entries.items())):
            n = QTableWidgetItem(name)
            n.setFlags(n.flags() & ~Qt.ItemIsEditable)
            v = e["value"]
            self.table.setItem(row, 0, n)
            self.table.setItem(row, 1, QTableWidgetItem(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v))
            self.table.setItem(row, 2, QTableWidgetItem(e.get("description", "")))
        self._updating = False

    def _edited(self, row: int, col: int) -> None:
        if self._updating or self._vars is None:
            return
        name = self.table.item(row, 0).text()
        if col == 1:
            text = self.table.item(row, 1).text()
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text
            self._vars.set(name, value)
        elif col == 2:
            entry = self._vars.entries().get(name)
            if entry:
                self._vars.define(name, entry["value"], entry.get("dtype", "any"), self.table.item(row, 2).text())

    def _add(self) -> None:
        if self._vars is None:
            return
        name, ok = QInputDialog.getText(self, "New variable", "Name:")
        if ok and name.strip():
            self._vars.define(name.strip(), "", "any", "")
            self.refresh()

    def _remove(self) -> None:
        if self._vars is None:
            return
        row = self.table.currentRow()
        if row >= 0:
            self._vars.remove(self.table.item(row, 0).text())
            self.refresh()

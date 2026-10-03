"""Parameter panel generated from a node's Param descriptors."""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ..core.graph import Graph
from ..core.node import Node, Param
from .i18n import tr


class ParamPanel(QScrollArea):
    param_changed = Signal(str, str, object)      # node_id, param name, value
    node_renamed = Signal(str, str)
    node_enabled_changed = Signal(str, bool)
    roi_edit_requested = Signal(str, str)         # node_id, param name
    roi_show_requested = Signal(str, str)         # node_id, param name (preview on image)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._node: Node | None = None
        self._graph: Graph | None = None
        self._locked = False
        self._container = QWidget()
        self.setWidget(self._container)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._widgets: dict[str, Any] = {}
        self._show_advanced = False
        self.set_node(None, None)

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self._container.setEnabled(not locked)

    def set_node(self, node: Node | None, graph: Graph | None) -> None:
        self._node, self._graph = node, graph
        self._rebuild()

    def refresh(self) -> None:
        """Update the error line without rebuilding widgets."""
        if self._node is not None and "__error" in self._widgets:
            self._widgets["__error"].setText(self._node.last_error or "")
            self._widgets["__error"].setVisible(bool(self._node.last_error))

    # ---- building ----
    def _clear(self) -> None:
        # QScrollArea.setWidget deletes the previous container together with every child widget/layout
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self.setWidget(self._container)
        self._widgets.clear()

    def _rebuild(self) -> None:
        self._clear()
        node = self._node
        if node is None:
            lbl = QLabel(tr("Select a node to edit its parameters.\n\nDrag nodes from the palette, connect ports by "
                            "dragging, press Delete to remove, F to fit."))
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color:#888")
            self._layout.addWidget(lbl)
            self._layout.addStretch(1)
            return
        head = QLabel(f"<b>{tr(node.label)}</b> <span style='color:#888'>{node.type_id}</span>")
        self._layout.addWidget(head)
        if node.description:
            d = QLabel(tr(node.description))
            d.setWordWrap(True)
            d.setStyleSheet("color:#aaa")
            self._layout.addWidget(d)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        name = QLineEdit(node.name)
        name.editingFinished.connect(lambda: self.node_renamed.emit(node.id, name.text()))
        form.addRow(tr("Name"), name)
        en = QCheckBox(tr("enabled"))
        en.setChecked(node.enabled)
        en.toggled.connect(lambda v: self.node_enabled_changed.emit(node.id, v))
        form.addRow("", en)
        self._layout.addLayout(form)
        err = QLabel(node.last_error or "")
        err.setWordWrap(True)
        err.setStyleSheet("color:#ff8a65")
        err.setVisible(bool(node.last_error))
        self._widgets["__error"] = err
        self._layout.addWidget(err)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#444")
        self._layout.addWidget(line)

        pform = QFormLayout()
        pform.setLabelAlignment(Qt.AlignRight)
        adv = [p for p in node.params if p.advanced]
        for p in node.params:
            if p.advanced and not self._show_advanced:
                continue
            w = self._make_widget(node, p)
            lbl = QLabel(tr(p.label))
            if p.description:
                lbl.setToolTip(tr(p.description))
                w.setToolTip(tr(p.description))
            pform.addRow(lbl, w)
        self._layout.addLayout(pform)
        if adv:
            btn = QPushButton((tr("Hide advanced") if self._show_advanced else tr("Show advanced")) + f" ({len(adv)})")
            btn.setFlat(True)
            btn.clicked.connect(self._toggle_advanced)
            self._layout.addWidget(btn)
        if node.state:
            st = QLabel(f"<span style='color:#888'>{tr('state:')}</span> " +
                        ", ".join(f"{k}={str(v)[:24]}" for k, v in list(node.state.items())[:6]))
            st.setWordWrap(True)
            self._layout.addWidget(st)
        self._layout.addStretch(1)
        self._container.setEnabled(not self._locked)

    def _toggle_advanced(self) -> None:
        self._show_advanced = not self._show_advanced
        self._rebuild()

    def _emit(self, p: Param, value: Any) -> None:
        if self._node is not None:
            self.param_changed.emit(self._node.id, p.name, value)

    def _make_widget(self, node: Node, p: Param) -> QWidget:
        v = node.get(p.name)
        k = p.kind
        if k == "int":
            if p.min is not None and p.max is not None and (p.max - p.min) <= 100000:
                return self._int_slider(p, int(v))
            sb = QSpinBox()
            sb.setRange(int(p.min) if p.min is not None else -2_000_000_000, int(p.max) if p.max is not None else 2_000_000_000)
            sb.setValue(int(v))
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda x: self._emit(p, x))
            return sb
        if k == "float":
            sb = QDoubleSpinBox()
            sb.setDecimals(4)
            sb.setRange(float(p.min) if p.min is not None else -1e12, float(p.max) if p.max is not None else 1e12)
            sb.setSingleStep(float(p.step) if p.step else (0.1 if (p.max or 1) <= 1 else 1.0))
            sb.setValue(float(v))
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda x: self._emit(p, x))
            return sb
        if k == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(v))
            cb.toggled.connect(lambda x: self._emit(p, x))
            return cb
        if k == "enum":
            combo = QComboBox()
            combo.addItems([str(c) for c in p.choices])
            combo.setCurrentText(str(v))
            combo.currentTextChanged.connect(lambda x: self._emit(p, x))
            return combo
        if k in ("file", "dir"):
            return self._path_widget(p, str(v or ""))
        if k == "rect":
            return self._rect_widget(node, p)
        if k in ("code", "json"):
            return self._text_widget(p, v, k == "json")
        if k == "list":
            le = QLineEdit(", ".join(str(x) for x in (v or [])))
            le.editingFinished.connect(lambda: self._emit(p, [s.strip() for s in le.text().split(",") if s.strip()]))
            return le
        le = QLineEdit(str(v if v is not None else ""))
        le.editingFinished.connect(lambda: self._emit(p, le.text()))
        return le

    def _int_slider(self, p: Param, value: int) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        sl = QSlider(Qt.Horizontal)
        sl.setRange(int(p.min), int(p.max))
        sl.setValue(value)
        sb = QSpinBox()
        sb.setRange(int(p.min), int(p.max))
        sb.setValue(value)
        sb.setKeyboardTracking(False)
        step = int(p.step) if p.step else 1
        if step > 1:
            sl.setSingleStep(step)
            sb.setSingleStep(step)
        sl.valueChanged.connect(sb.setValue)
        sb.valueChanged.connect(sl.setValue)
        sb.valueChanged.connect(lambda x: self._emit(p, x))
        lay.addWidget(sl, 3)
        lay.addWidget(sb, 1)
        return w

    def _path_widget(self, p: Param, value: str) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        le = QLineEdit(value)
        le.editingFinished.connect(lambda: self._emit(p, le.text()))
        btn = QPushButton("…")
        btn.setFixedWidth(28)

        def browse():
            if p.kind == "dir":
                path = QFileDialog.getExistingDirectory(self, p.label, le.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, p.label, le.text(), p.filter or "All files (*)")
            if path:
                le.setText(path)
                self._emit(p, path)
        btn.clicked.connect(browse)
        lay.addWidget(le)
        lay.addWidget(btn)
        return w

    def _rect_widget(self, node: Node, p: Param) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        r = node.get(p.name)
        txt = f"x{r['x']:.0f} y{r['y']:.0f} {r['w']:.0f}×{r['h']:.0f}" if r else tr("not set")
        lbl = QLabel(txt)
        lbl.setStyleSheet("color:#ffa500" if r else "color:#888")
        draw = QPushButton(tr("Draw"))
        draw.setToolTip(tr("Drag a rectangle on the image"))
        draw.clicked.connect(lambda: self.roi_edit_requested.emit(node.id, p.name))
        show = QPushButton(tr("Show"))
        show.clicked.connect(lambda: self.roi_show_requested.emit(node.id, p.name))
        clear = QPushButton("✕")
        clear.setFixedWidth(26)
        clear.clicked.connect(lambda: self._emit(p, None))
        lay.addWidget(lbl, 1)
        lay.addWidget(draw)
        lay.addWidget(show)
        lay.addWidget(clear)
        return w

    def _text_widget(self, p: Param, value: Any, is_json: bool) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        ed = QPlainTextEdit()
        ed.setFont(QFont("Monospace", 9))
        ed.setMinimumHeight(140 if not is_json else 60)
        ed.setPlainText(json.dumps(value, indent=1, ensure_ascii=False) if is_json else str(value or ""))
        row = QHBoxLayout()
        apply_btn = QPushButton(tr("Apply"))
        status = QLabel("")
        status.setStyleSheet("color:#ff8a65")

        def apply():
            text = ed.toPlainText()
            if is_json:
                try:
                    val = json.loads(text) if text.strip() else {}
                except json.JSONDecodeError as e:
                    status.setText(tr("JSON error: {msg} (line {line})").format(msg=e.msg, line=e.lineno))
                    return
            else:
                val = text
            status.setText("")
            self._emit(p, val)
        apply_btn.clicked.connect(apply)
        row.addWidget(apply_btn)
        row.addWidget(status, 1)
        lay.addWidget(ed)
        lay.addLayout(row)
        return w

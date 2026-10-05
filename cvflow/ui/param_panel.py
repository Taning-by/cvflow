"""参数面板：按节点的 Param 描述生成编辑器。

布局规则：
* 顶部是节点身份卡（分类色、名称、类型、状态），下面才是参数，避免每次都要去流程图上确认选中了谁。
* 参数分组可折叠：常用参数常驻，高级参数与"本次输出"按需展开，低频内容默认收起。
* 每行是对齐的「标签 — 控件 — 单位」，标签带虚线下划线表示有说明（悬停查看）。
* 输入错误就地显示在该字段下方，写清原因和改法。
"""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSpinBox, QVBoxLayout, QWidget)

from ..core.graph import Graph
from ..core.node import Node, NodeStatus, Param
from ..core.types import to_jsonable
from .i18n import tr
from .theme import C, M, STATUS_COLORS, STATUS_GLYPH, STATUS_TEXT, mono_font, soft_chip_style, tabular, ui_font

# 参数名后缀 → 单位。单位只写在控件后面，不改参数标签本身。
_UNITS = {"_ms": "ms", "_s": "s", "_us": "µs", "_px": "px", "_pct": "%", "_deg": "°", "_mm": "mm"}
_UNIT_BY_NAME = {"exposure": "µs", "gain": "dB", "timeout": "s", "angle": "°", "interval": "s"}


def _unit_for(p: Param) -> str:
    label = tr(p.label)
    if "(" in label or "（" in label:        # 标签里已经写了单位，不再重复
        return ""
    for suffix, unit in _UNITS.items():
        if p.name.endswith(suffix):
            return unit
    return _UNIT_BY_NAME.get(p.name, "")


def _decimals(p: Param) -> int:
    """小数位按步长取，默认 2 位；没有步长但上限不超过 1 的（置信度之类）给 3 位。"""
    if p.step:
        step = abs(float(p.step))
        for d in range(5):
            if abs(step * 10 ** d - round(step * 10 ** d)) < 1e-9:
                return d
    if p.max is not None and abs(float(p.max)) <= 1:
        return 3
    return 2


def _fmt(v) -> str:
    """输出值的紧凑写法：浮点保留 4 位有效数字，列表逐项处理，别让一屏都是小数。"""
    j = to_jsonable(v)
    if isinstance(j, float):
        return f"{j:.4g}"
    if isinstance(j, (list, tuple)):
        inner = ", ".join(_fmt(x) for x in list(j)[:12])
        return "[" + inner + (", …" if len(j) > 12 else "") + "]"
    return str(j)


class _Section(QWidget):
    """可折叠分组：标题行 + 内容。折叠状态由面板按标题记忆，重建后保持不变。"""

    def __init__(self, title: str, panel: "ParamPanel", collapsed: bool = False, badge: str = "") -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(M["gap_s"])
        self._panel = panel
        self._title = title
        self.header = QPushButton()
        self.header.setObjectName("sectionHeader")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet(
            f"QPushButton#sectionHeader {{ background: transparent; border: none; border-bottom: 1px solid {C['border']};"
            f" color: {C['muted']}; font-size: {M['font_s']}px; font-weight: 600; text-align: left;"
            f" padding: 5px 2px; }}"
            f"QPushButton#sectionHeader:hover {{ color: {C['accent']}; }}")
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, M["gap_s"], 0, M["gap"])
        self.body_layout.setSpacing(M["gap_s"])
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self._badge = badge
        self._collapsed = collapsed
        self.header.clicked.connect(self._toggle)
        self._sync()

    def _sync(self) -> None:
        arrow = "▸" if self._collapsed else "▾"
        self.header.setText(f"  {arrow}  {self._title}" + (f"   {self._badge}" if self._badge else ""))
        self.body.setVisible(not self._collapsed)

    def _toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._panel.set_collapsed(self._title, self._collapsed)
        self._sync()

    def add(self, w) -> None:
        if isinstance(w, QWidget):
            self.body_layout.addWidget(w)
        else:
            self.body_layout.addLayout(w)


class ParamPanel(QScrollArea):
    param_changed = Signal(str, str, object)      # node_id, param name, value
    node_renamed = Signal(str, str)
    node_enabled_changed = Signal(str, bool)
    roi_edit_requested = Signal(str, str)         # node_id, param name
    roi_show_requested = Signal(str, str)         # node_id, param name (preview on image)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(232)          # 内容必须能缩到这个宽度，否则会把停靠面板撑出窗口
        self._node: Node | None = None
        self._graph: Graph | None = None
        self._locked = False
        self._result = None                        # 当前节点最近一次运行的结果（NodeResult）
        self._collapsed: set[str] = {tr("Node state")}
        self._container = QWidget()
        self.setWidget(self._container)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(M["gap_l"], M["gap_l"], M["gap_l"], M["gap_l"])
        self._widgets: dict[str, Any] = {}
        self._show_advanced = False
        self.set_node(None, None)

    # ---- 外部接口 ----
    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self._container.setEnabled(not locked)

    def set_node(self, node: Node | None, graph: Graph | None) -> None:
        self._node, self._graph = node, graph
        self._rebuild()

    def set_result(self, node_result) -> None:
        """把当前节点最近一次运行的输出交给面板（由主窗口在每次运行后调用）。"""
        self._result = node_result
        self._fill_outputs()

    def set_collapsed(self, title: str, collapsed: bool) -> None:
        self._collapsed.discard(title)
        if collapsed:
            self._collapsed.add(title)

    def refresh(self) -> None:
        """只更新错误行与状态胶囊，不重建控件（运行中频繁调用）。"""
        node = self._node
        if node is None:
            return
        err = self._widgets.get("__error")
        if err is not None:
            err.setText(node.last_error or "")
            err.setVisible(bool(node.last_error))
        chip = self._widgets.get("__status")
        if chip is not None:
            self._style_status(chip, node.status)

    # ---- 构建 ----
    def _clear(self) -> None:
        # 取下旧容器而不是让 setWidget 直接删除它：重建常常是由面板内某个控件自己的信号触发的
        # （键入后回车、失焦提交、按钮点击），同步销毁会让 Qt 在信号返回后访问已释放的控件而崩溃。
        old = self.takeWidget()
        if old is not None:
            old.setParent(None)
            old.deleteLater()
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(M["gap_l"], M["gap_l"], M["gap_l"], M["gap_l"])
        self._layout.setSpacing(M["gap"])
        self.setWidget(self._container)
        self._widgets.clear()

    def _section(self, title: str, badge: str = "", collapsed: bool | None = None) -> _Section:
        if collapsed is None:
            collapsed = title in self._collapsed
        s = _Section(title, self, collapsed, badge)
        self._layout.addWidget(s)
        return s

    def _rebuild(self) -> None:
        self._clear()
        node = self._node
        if node is None:
            self._layout.addWidget(self._placeholder())
            self._layout.addStretch(1)
            return

        self._layout.addWidget(self._identity_card(node))

        err = QLabel(node.last_error or "")
        err.setWordWrap(True)
        err.setTextInteractionFlags(Qt.TextSelectableByMouse)   # 错误信息要能选中复制
        err.setCursor(Qt.IBeamCursor)
        err.setStyleSheet(f"color:{C['ng']}; background:{C['ng_bg']}; border:1px solid {C['ng']};"
                          f" border-radius:{M['radius_s']}px; padding:6px 8px")
        err.setVisible(bool(node.last_error))
        self._widgets["__error"] = err
        self._layout.addWidget(err)

        basic = [p for p in node.params if not p.advanced]
        adv = [p for p in node.params if p.advanced]
        sec = self._section(tr("Parameters"), f"{len(basic)}")
        if basic:
            sec.add(self._form(node, basic))
        else:
            hint = QLabel(tr("This node has no parameters."))
            hint.setObjectName("muted")
            sec.add(hint)
        if adv:
            btn = QPushButton((tr("Hide advanced") if self._show_advanced else tr("Show advanced")) + f" ({len(adv)})")
            btn.setObjectName("linklike")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(self._toggle_advanced)
            sec.add(btn)
            if self._show_advanced:
                sec.add(self._form(node, adv))

        out_sec = self._section(tr("Outputs"))
        self._widgets["__outputs"] = out_sec
        self._widgets["__outputs_form"] = QFormLayout()
        self._widgets["__outputs_form"].setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._widgets["__outputs_form"].setHorizontalSpacing(M["gap_l"])
        self._widgets["__outputs_form"].setVerticalSpacing(M["gap_s"])
        out_sec.add(self._widgets["__outputs_form"])
        self._fill_outputs()

        if node.state:
            st_sec = self._section(tr("Node state"))
            st = QLabel(", ".join(f"{k}={str(v)[:24]}" for k, v in list(node.state.items())[:8]))
            st.setWordWrap(True)
            st.setObjectName("muted")
            st_sec.add(st)
        self._layout.addStretch(1)
        self._container.setEnabled(not self._locked)

    def _placeholder(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, M["gap_l"], 0, 0)
        title = QLabel(tr("No node selected"))
        title.setFont(ui_font(M["font_title"], True))
        tip = QLabel(tr("Select a node to edit its parameters.\n\nDrag nodes from the palette, connect ports by "
                        "dragging, press Delete to remove, F to fit."))
        tip.setWordWrap(True)
        tip.setObjectName("muted")
        lay.addWidget(title)
        lay.addWidget(tip)
        return w

    def _identity_card(self, node: Node) -> QWidget:
        """节点身份卡：分类色、名称、类型、状态、启用开关。"""
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(f"QFrame#card {{ background:{C['panel2']}; border:1px solid {C['border']};"
                           f" border-left:4px solid {node.color}; border-radius:{M['radius']}px; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(M["gap_l"], M["gap"], M["gap"], M["gap"])
        lay.setSpacing(M["gap_s"])

        top = QHBoxLayout()
        top.setSpacing(M["gap"])
        label = QLabel(tr(node.label))
        label.setFont(ui_font(M["font_title"], True))
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        tid = QLabel(node.type_id)
        tid.setObjectName("muted")
        tid.setFont(mono_font(11))
        tid.setToolTip(f"{node.type_id}　{tr('Node type id')}")
        tid.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # 窄面板里先让类型标识让位
        chip = QLabel()
        self._widgets["__status"] = chip
        self._style_status(chip, node.status)
        top.addWidget(label, 3)
        top.addWidget(tid, 2)
        top.addStretch(0)
        top.addWidget(chip)
        lay.addLayout(top)

        if node.description:
            d = QLabel(tr(node.description))
            d.setWordWrap(True)
            d.setObjectName("muted")
            lay.addWidget(d)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(M["gap_l"])
        form.setVerticalSpacing(M["gap_s"])
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        name = QLineEdit(node.name)
        name.setFixedHeight(M["ctl_h"])
        name.setToolTip(tr("Name shown on the node and in results"))
        name.editingFinished.connect(lambda: self.node_renamed.emit(node.id, name.text()))
        form.addRow(self._label(tr("Name")), name)
        en = QCheckBox(tr("enabled"))
        en.setChecked(node.enabled)
        en.setToolTip(tr("Disabled nodes are skipped when the flow runs"))
        en.toggled.connect(lambda v: self.node_enabled_changed.emit(node.id, v))
        form.addRow("", en)
        lay.addLayout(form)
        return card

    def _style_status(self, chip: QLabel, status: NodeStatus) -> None:
        col = STATUS_COLORS.get(status, C["muted"])
        bg = {NodeStatus.OK: C["ok_bg"], NodeStatus.NG: C["ng_bg"], NodeStatus.ERROR: C["warn_bg"]}.get(status, C["panel"])
        chip.setText(f"{STATUS_GLYPH.get(status, '')} {STATUS_TEXT.get(status, status.value)}")
        chip.setStyleSheet(soft_chip_style(col, bg, col))

    def _label(self, text: str, description: str = "") -> QLabel:
        lbl = QLabel(text)
        lbl.setMinimumWidth(M["label_w"])
        lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        if description:
            # 虚线下划线＝有说明可看，悬停即是帮助入口（图标按钮同样带名称提示）
            lbl.setStyleSheet(f"border-bottom: 1px dotted {C['border2']};")
            lbl.setToolTip(description)
            lbl.setCursor(Qt.WhatsThisCursor)
        return lbl

    def _form(self, node: Node, params: list[Param]) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setVerticalSpacing(M["gap"])
        form.setHorizontalSpacing(M["gap_l"])
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for p in params:
            w = self._make_widget(node, p)
            desc = tr(p.description) if p.description else ""
            if desc:
                w.setToolTip(desc)
            form.addRow(self._label(tr(p.label), desc), w)
        return form

    def _toggle_advanced(self) -> None:
        self._show_advanced = not self._show_advanced
        self._rebuild()

    def _emit(self, p: Param, value: Any) -> None:
        if self._node is not None:
            self.param_changed.emit(self._node.id, p.name, value)

    # ---- 输出区 ----
    def _fill_outputs(self) -> None:
        form = self._widgets.get("__outputs_form")
        sec = self._widgets.get("__outputs")
        if form is None or sec is None:
            return
        while form.rowCount():
            form.removeRow(0)
        res = self._result
        if res is None:
            lbl = QLabel(tr("Run the flow to see this node's outputs."))
            lbl.setObjectName("muted")
            lbl.setWordWrap(True)
            form.addRow(lbl)
            sec._badge = ""
            sec._sync()
            return
        status = QLabel(f"{STATUS_GLYPH.get(res.status, '')} {STATUS_TEXT.get(res.status, res.status.value)}"
                        f"　{res.time_ms:.2f} ms")
        status.setStyleSheet(f"color:{STATUS_COLORS.get(res.status, C['muted'])}; font-weight:600")
        form.addRow(self._label(tr("Status")), status)
        if res.error:
            e = QLabel(res.error)
            e.setWordWrap(True)
            e.setTextInteractionFlags(Qt.TextSelectableByMouse)
            e.setStyleSheet(f"color:{C['ng']}")
            form.addRow(self._label(tr("Message")), e)
        for k, v in res.outputs.items():
            text = _fmt(v)
            val = QLabel(text if len(text) <= 120 else text[:119] + "…")
            val.setToolTip(text)
            val.setFont(tabular(ui_font()))
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val.setWordWrap(True)
            form.addRow(self._label(k), val)
        sec._badge = f"{len(res.outputs)}"
        sec._sync()

    # ---- 控件工厂 ----
    def _make_widget(self, node: Node, p: Param) -> QWidget:
        v = node.get(p.name)
        k = p.kind
        unit = _unit_for(p)
        if k == "int":
            if p.min is not None and p.max is not None and (p.max - p.min) <= 100000:
                return self._int_slider(p, int(v), unit)
            sb = QSpinBox()
            sb.setRange(int(p.min) if p.min is not None else -2_000_000_000, int(p.max) if p.max is not None else 2_000_000_000)
            sb.setValue(int(v))
            sb.setKeyboardTracking(False)
            self._finish_number(sb, unit)
            sb.valueChanged.connect(lambda x: self._emit(p, x))
            return sb
        if k == "float":
            sb = QDoubleSpinBox()
            sb.setDecimals(_decimals(p))
            sb.setRange(float(p.min) if p.min is not None else -1e12, float(p.max) if p.max is not None else 1e12)
            sb.setSingleStep(float(p.step) if p.step else (0.1 if (p.max or 1) <= 1 else 1.0))
            sb.setValue(float(v))
            sb.setKeyboardTracking(False)
            self._finish_number(sb, unit)
            sb.valueChanged.connect(lambda x: self._emit(p, x))
            return sb
        if k == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(v))
            cb.setFixedHeight(M["ctl_h"])
            cb.toggled.connect(lambda x: self._emit(p, x))
            return cb
        if k == "enum":
            combo = QComboBox()
            combo.addItems([str(c) for c in p.choices])
            combo.setCurrentText(str(v))
            combo.setFixedHeight(M["ctl_h"])
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
            le.setFixedHeight(M["ctl_h"])
            le.setPlaceholderText(tr("comma separated"))
            le.editingFinished.connect(lambda: self._emit(p, [s.strip() for s in le.text().split(",") if s.strip()]))
            return le
        le = QLineEdit(str(v if v is not None else ""))
        le.setFixedHeight(M["ctl_h"])
        le.editingFinished.connect(lambda: self._emit(p, le.text()))
        return le

    @staticmethod
    def _finish_number(sb, unit: str) -> None:
        sb.setFixedHeight(M["ctl_h"])
        sb.setFont(tabular(ui_font()))          # 数值用等宽数字，纵向可比
        sb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if unit:
            sb.setSuffix(f" {unit}")

    def _int_slider(self, p: Param, value: int, unit: str = "") -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(M["gap"])
        sl = QSlider(Qt.Horizontal)
        sl.setRange(int(p.min), int(p.max))
        sl.setValue(value)
        sb = QSpinBox()
        sb.setRange(int(p.min), int(p.max))
        sb.setValue(value)
        sb.setKeyboardTracking(False)
        self._finish_number(sb, unit)
        sb.setMinimumWidth(72)
        sb.setMaximumWidth(120)
        sl.setMinimumWidth(52)
        step = int(p.step) if p.step else 1
        if step > 1:
            sl.setSingleStep(step)
            sb.setSingleStep(step)
        sl.valueChanged.connect(sb.setValue)
        sb.valueChanged.connect(sl.setValue)
        sb.valueChanged.connect(lambda x: self._emit(p, x))
        lay.addWidget(sl, 3)
        lay.addWidget(sb, 2)
        return w

    def _path_widget(self, p: Param, value: str) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(M["gap_s"])
        le = QLineEdit(value)
        le.setFixedHeight(M["ctl_h"])
        le.setFont(mono_font())
        le.setToolTip(value or tr("not set"))
        le.editingFinished.connect(lambda: self._emit(p, le.text()))
        btn = QPushButton("…")
        btn.setFixedSize(M["ctl_h"], M["ctl_h"])
        btn.setToolTip(tr("Browse…"))

        def browse():
            if p.kind == "dir":
                path = QFileDialog.getExistingDirectory(self, p.label, le.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, p.label, le.text(), p.filter or "All files (*)")
            if path:
                le.setText(path)
                self._emit(p, path)
        btn.clicked.connect(browse)
        lay.addWidget(le, 1)
        lay.addWidget(btn)
        return w

    def _rect_widget(self, node: Node, p: Param) -> QWidget:
        """ROI 行：按钮一行，矩形数值单独一行，窄面板里数值也不会被挤没。"""
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(M["gap_s"])
        r = node.get(p.name)
        txt = f"x {r['x']:.0f}  y {r['y']:.0f}  {r['w']:.0f}×{r['h']:.0f}" if r else tr("not set")
        lbl = QLabel(txt)
        lbl.setFont(tabular(ui_font(M["font_s"])))
        lbl.setStyleSheet(f"color:{C['text'] if r else C['faint']}; font-size:{M['font_s']}px")
        lbl.setToolTip(f"{txt}　{tr('ROI in image pixels')}")
        draw = QPushButton(tr("Draw"))
        draw.setObjectName("compact")
        draw.setToolTip(tr("Drag a rectangle on the image"))
        draw.setFixedHeight(M["ctl_h"])
        draw.clicked.connect(lambda: self.roi_edit_requested.emit(node.id, p.name))
        show = QPushButton(tr("Show"))
        show.setObjectName("compact")
        show.setToolTip(tr("Show this ROI on the image"))
        show.setFixedHeight(M["ctl_h"])
        show.setEnabled(bool(r))
        show.clicked.connect(lambda: self.roi_show_requested.emit(node.id, p.name))
        clear = QPushButton("✕")
        clear.setToolTip(tr("Clear ROI"))
        clear.setObjectName("compact")
        clear.setFixedSize(M["ctl_h"], M["ctl_h"])
        clear.setEnabled(bool(r))
        clear.clicked.connect(lambda: self._emit(p, None))
        lay.addWidget(draw)
        lay.addWidget(show)
        lay.addWidget(clear)
        lay.addStretch(1)
        outer.addWidget(row)
        outer.addWidget(lbl)
        return w

    def _text_widget(self, p: Param, value: Any, is_json: bool) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(M["gap_s"])
        ed = QPlainTextEdit()
        ed.setFont(mono_font(12))
        ed.setMinimumHeight(60 if is_json else 160)
        ed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ed.setPlainText(json.dumps(value, indent=1, ensure_ascii=False) if is_json else str(value or ""))
        row = QHBoxLayout()
        row.setSpacing(M["gap"])
        apply_btn = QPushButton(tr("Apply"))
        apply_btn.setFixedHeight(M["ctl_h"])
        status = QLabel("")
        status.setWordWrap(True)
        status.setStyleSheet(f"color:{C['ng']}")

        def apply():
            text = ed.toPlainText()
            if is_json:
                try:
                    val = json.loads(text) if text.strip() else {}
                except json.JSONDecodeError as e:
                    # 就地说明原因与位置，用户不必去别处找错
                    status.setText(tr("JSON error: {msg} (line {line})").format(msg=e.msg, line=e.lineno))
                    ed.setProperty("invalid", True)
                    ed.setStyleSheet(f"border:1px solid {C['ng']};")
                    return
            else:
                val = text
            status.setText("")
            ed.setProperty("invalid", False)
            ed.setStyleSheet("")
            self._emit(p, val)
        apply_btn.clicked.connect(apply)
        row.addWidget(apply_btn)
        row.addWidget(status, 1)
        lay.addWidget(ed)
        lay.addLayout(row)
        return w

"""通信配置对话框：设备、数据点、触发规则、发送规则、解析/格式化规则、检测握手。

所有表单都由**声明**生成：设备参数来自 ``CommDevice.schema()``，数据点来自
``modbus_map`` 的取值表。所以新增一种协议或一种数据类型，界面不用改代码。

校验一律在点"确定"时做，错误信息是中文、指明哪一项不对（例如"保持寄存器 (4x) 只能放布尔值"）。
"""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from ..comm.framing import FRAMING_LABELS, LENGTH_SEMANTIC_LABELS
from ..comm.handshake import (BUSY_POLICY_LABELS, DEDUP_LABELS, ERROR_TEXT, HandshakeConfig)
from ..comm.manager import (ACTION_LABELS, DATA_MATCH_LABELS, DEVICE_KIND_LABELS, DEVICE_KINDS,
                            DEVICE_STATUS, DEVICE_STATUS_LABELS, PLANNED_PROTOCOLS,
                            RULE_SOURCE_LABELS, SEND_TARGET_LABELS, TEXT_MATCH_LABELS,
                            WHEN_LABELS, ReceiveRule, SendRule)
from ..comm.modbus_map import (AREA_INFO, DIRECTION_LABELS, DTYPE_LABELS, LAYOUT_LABELS,
                               DataPoint, MapError, layout_example, reference_text)
from ..comm.parsing import (FORMAT_LABELS, PARSE_LABELS, SOURCE_SPECIALS, FormatField, FormatRule,
                            ParseField, ParseRule, SCALAR_FORMATS, VALUE_TYPES)
from .i18n import tr
from .theme import C, M

#: 发送规则 / 格式化规则里字段来源的提示文字。
SOURCE_HELP = ("可用来源：" + "、".join(SOURCE_SPECIALS)
               + "；out.名字（发布结果节点）、var.名字（全局变量）、field.名字（本次请求的字段）、"
                 "node[节点名].端口、literal:常量")


# ------------------------------------------------------------------ 通用
def _widget_for(kind: str, value: Any, desc: str = "") -> QWidget:
    """按声明里的类型造一个编辑控件。"""
    if kind == "int":
        w = QSpinBox()
        w.setRange(-1, 2_000_000_000)
        w.setValue(int(value or 0))
    elif kind == "float":
        w = QDoubleSpinBox()
        w.setRange(-1e9, 1e9)
        w.setDecimals(3)
        w.setValue(float(value or 0.0))
    elif kind == "bool":
        w = QCheckBox()
        w.setChecked(bool(value))
    elif kind.startswith("enum:"):
        w = QComboBox()
        for choice in kind[5:].split(","):
            w.addItem(_choice_label(choice), choice)
        idx = w.findData(str(value))
        w.setCurrentIndex(idx if idx >= 0 else 0)
    else:
        w = QLineEdit("" if value is None else str(value))
    if desc:
        w.setToolTip(tr(desc))
    return w


def _choice_label(choice: str) -> str:
    """枚举值的中文显示。认不出来就原样显示（协议里的字母参数该保持原样）。"""
    for table in (FRAMING_LABELS, LENGTH_SEMANTIC_LABELS, DIRECTION_LABELS, LAYOUT_LABELS,
                  SEND_TARGET_LABELS, WHEN_LABELS, BUSY_POLICY_LABELS):
        if choice in table:
            return f"{choice} — {table[choice]}" if len(choice) > 2 else table[choice]
    return {"big": "big 大端", "little": "little 小端"}.get(choice, choice)


def _read_widget(w: QWidget) -> Any:
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QComboBox):
        return w.currentData() if w.currentData() is not None else w.currentText()
    if isinstance(w, QPlainTextEdit):
        return w.toPlainText()
    return w.text()


class _BaseDialog(QDialog):
    """带底部按钮与一行错误提示的对话框。``validate`` 返回错误文字就不关闭。"""

    def __init__(self, parent, title: str, width: int = 560) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr(title))
        self.setMinimumWidth(width)
        self._outer = QVBoxLayout(self)
        self.body = QWidget()
        self._outer.addWidget(self.body)
        self.err = QLabel("")
        self.err.setWordWrap(True)
        self.err.setStyleSheet(f"color:{C['ng']}")
        self.err.setVisible(False)
        self._outer.addWidget(self.err)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._try_accept)
        self.buttons.rejected.connect(self.reject)
        self._outer.addWidget(self.buttons)

    def _try_accept(self) -> None:
        problem = self.validate()
        if problem:
            self.err.setText(problem)
            self.err.setVisible(True)
            return
        self.accept()

    def validate(self) -> str:
        return ""


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color:{C['muted']};font-size:{M['font_s']}px")
    return lbl


# ------------------------------------------------------------------ 设备
class DeviceDialog(_BaseDialog):
    """新增 / 编辑一台通信设备。协议参数、分帧、连接与重试分三组显示。"""

    def __init__(self, parent=None, name: str = "", kind: str = "tcp_server",
                 config: dict | None = None, enabled: bool = True,
                 device_id: str = "", used_names: list[str] | None = None) -> None:
        super().__init__(parent, "通信设备", 620)
        self.device_id = device_id
        self._used = [n for n in (used_names or []) if n != name]
        self._config = dict(config or {})
        self._fields: dict[str, QWidget] = {}

        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QFormLayout()
        self.name = QLineEdit(name or "plc")
        self.kind = QComboBox()
        for k in sorted(DEVICE_KINDS):
            note = DEVICE_STATUS_LABELS.get(DEVICE_STATUS.get(k, ""), "")
            self.kind.addItem(f"{DEVICE_KIND_LABELS.get(k, k)}{note}", k)
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        self.kind.setEnabled(not name)          # 类型定了就不改了，改类型等于换一台设备
        self.enabled = QCheckBox(tr("启用"))
        self.enabled.setChecked(bool(enabled))
        top.addRow(tr("名称"), self.name)
        top.addRow(tr("类型"), self.kind)
        top.addRow(tr("状态"), self.enabled)
        if device_id:
            ident = QLabel(device_id)
            ident.setStyleSheet(f"color:{C['muted']}")
            ident.setToolTip(tr("设备的稳定标识：改名不会破坏流程与规则里的引用"))
            top.addRow(tr("标识"), ident)
        lay.addLayout(top)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        self.kind.currentIndexChanged.connect(lambda _: self._build())
        self._build()
        if PLANNED_PROTOCOLS:
            lay.addWidget(_hint("还未实现的协议（不会出现在上面的列表里）："
                                + "、".join(PLANNED_PROTOCOLS.values())))

    # ---- 表单 ----
    def _build(self) -> None:
        while self.tabs.count():
            self.tabs.removeTab(0)
        self._fields.clear()
        cls = DEVICE_KINDS[self.kind.currentData()]
        groups: dict[str, list] = {"协议参数": [], "报文分帧": [], "连接与重试": []}
        from ..comm.base import COMMON_SCHEMA, FRAMING_SCHEMA
        framing_keys = {n for n, *_ in FRAMING_SCHEMA}
        common_keys = {n for n, *_ in COMMON_SCHEMA}
        for item in cls.schema():
            key = item[0]
            group = "报文分帧" if key in framing_keys else ("连接与重试" if key in common_keys else "协议参数")
            groups[group].append(item)
        for title, items in groups.items():
            if not items:
                continue
            page = QWidget()
            form = QFormLayout(page)
            for key, kind, label, default, desc in items:
                w = _widget_for(kind, self._config.get(key, default), desc)
                self._fields[key] = w
                form.addRow(tr(label), w)
            if title == "报文分帧":
                form.addRow(_hint("TCP 与串口是字节流：一次 recv 可能只有半条报文，也可能有好几条。"
                                  "分帧方式决定一条报文到哪里结束，必须和对端一致。"))
            self.tabs.addTab(page, tr(title))

    # ---- 校验 ----
    def validate(self) -> str:
        name = self.name.text().strip()
        if not name:
            return "设备名称不能为空。"
        if name in self._used:
            return f"已经有一台设备叫「{name}」了，请换一个名称。"
        cfg = self.result()[2]
        port = cfg.get("port")
        if isinstance(port, int) and not 0 <= port <= 65535:
            return f"端口 {port} 不在 0…65535 范围内（0 表示由系统分配）。"
        if self.kind.currentData() in ("tcp_client", "modbus_tcp_client", "mc", "s7") and not str(cfg.get("host", "")).strip():
            return "目标地址不能为空。"
        cls = DEVICE_KINDS[self.kind.currentData()]
        if cls.byte_stream:
            from ..comm.framing import FramingConfig
            try:
                FramingConfig.from_config(cfg)
            except ValueError as e:
                return f"分帧配置：{e}"
        try:
            "".encode(str(cfg.get("encoding", "utf-8") or "utf-8"))
        except LookupError:
            return f"不认识的文本编码 {cfg.get('encoding')!r}（可用 utf-8 / ascii / gbk / gb18030 / latin-1）。"
        return ""

    def result(self) -> tuple[str, str, dict, bool]:
        cfg = {k: _read_widget(w) for k, w in self._fields.items()}
        return self.name.text().strip(), self.kind.currentData(), cfg, self.enabled.isChecked()


# ------------------------------------------------------------------ 数据点
class DataPointDialog(_BaseDialog):
    """一个 Modbus 数据点。地址同时显示参考地址，字节序同时显示实际字节示例。"""

    def __init__(self, parent=None, point: DataPoint | None = None,
                 used_names: list[str] | None = None, variables: list[str] | None = None) -> None:
        super().__init__(parent, "数据点", 600)
        p = point or DataPoint(name="point")
        self._used = [n for n in (used_names or []) if n != p.name]
        form = QFormLayout(self.body)
        self.name = QLineEdit(p.name)
        self.area = QComboBox()
        for key, info in AREA_INFO.items():
            self.area.addItem(f"{info['label']}  参考地址 {info['ref_base']}…", key)
        self.area.setCurrentIndex(max(0, self.area.findData(p.area)))
        self.address = QSpinBox(); self.address.setRange(0, 65535); self.address.setValue(int(p.address))
        self.ref = QLabel("")
        self.dtype = QComboBox()
        for key, label in DTYPE_LABELS.items():
            self.dtype.addItem(f"{key} — {label}", key)
        self.dtype.setCurrentIndex(max(0, self.dtype.findData(p.dtype)))
        self.count = QSpinBox(); self.count.setRange(1, 2000); self.count.setValue(int(p.count))
        self.length = QSpinBox(); self.length.setRange(0, 250); self.length.setValue(int(p.length))
        self.direction = QComboBox()
        for key, label in DIRECTION_LABELS.items():
            self.direction.addItem(label, key)
        self.direction.setCurrentIndex(max(0, self.direction.findData(p.direction)))
        self.poll_ms = QSpinBox(); self.poll_ms.setRange(0, 600000); self.poll_ms.setValue(int(p.poll_ms))
        self.scale = QDoubleSpinBox(); self.scale.setRange(-1e6, 1e6); self.scale.setDecimals(6)
        self.scale.setValue(float(p.scale))
        self.layout_box = QComboBox()
        for key, label in LAYOUT_LABELS.items():
            self.layout_box.addItem(label, key)
        self.layout_box.setCurrentIndex(max(0, self.layout_box.findData(p.layout)))
        self.layout_hint = _hint("")
        self.encoding = QLineEdit(p.encoding)
        self.variable = QComboBox(); self.variable.setEditable(True)
        self.variable.addItems([""] + list(variables or []))
        self.variable.setCurrentText(p.variable)
        self.description = QLineEdit(p.description)
        self.enabled = QCheckBox(tr("启用")); self.enabled.setChecked(bool(p.enabled))

        rows = [("名称", self.name), ("数据区", self.area), ("地址（协议地址，从 0 开始）", self.address),
                ("参考地址", self.ref), ("数据类型", self.dtype), ("数量", self.count),
                ("字符串字节数", self.length), ("读写方向", self.direction),
                ("轮询周期（毫秒，0=不轮询）", self.poll_ms), ("缩放系数", self.scale),
                ("字节序 / 字序", self.layout_box), ("字节示例", self.layout_hint),
                ("字符串编码", self.encoding), ("绑定全局变量", self.variable),
                ("说明", self.description), ("", self.enabled)]
        for label, w in rows:
            form.addRow(tr(label), w)
        form.addRow(_hint("内部一律使用协议地址（报文里那个数，从 0 开始）；参考地址是 40001 那一套写法。"
                          "32 位数据占两个寄存器：ABCD 不交换、CDAB 交换寄存器顺序、"
                          "BADC 交换寄存器内两个字节、DCBA 两者都交换。"))
        for w in (self.area, self.address, self.dtype, self.layout_box):
            sig = w.valueChanged if isinstance(w, QSpinBox) else w.currentIndexChanged
            sig.connect(lambda *_: self._refresh())
        self._refresh()

    def _refresh(self) -> None:
        area, addr = self.area.currentData(), self.address.value()
        try:
            self.ref.setText(reference_text(area, addr))
        except MapError as e:
            self.ref.setText(str(e))
        self.layout_hint.setText(layout_example(self.dtype.currentData(), self.layout_box.currentData()))
        bits = AREA_INFO.get(area, {}).get("bits", False)
        self.count.setEnabled(bits)
        self.length.setEnabled(self.dtype.currentData() == "string")
        self.layout_box.setEnabled(not bits)

    def validate(self) -> str:
        if self.name.text().strip() in self._used:
            return f"已经有一个数据点叫「{self.name.text().strip()}」了。"
        try:
            self.result()
        except MapError as e:
            return str(e)
        return ""

    def result(self) -> DataPoint:
        return DataPoint(name=self.name.text().strip(), area=self.area.currentData(),
                         address=self.address.value(), dtype=self.dtype.currentData(),
                         count=self.count.value(), length=self.length.value(),
                         direction=self.direction.currentData(), poll_ms=self.poll_ms.value(),
                         scale=self.scale.value(), layout=self.layout_box.currentData(),
                         encoding=self.encoding.text().strip() or "ascii",
                         variable=self.variable.currentText().strip(),
                         description=self.description.text(), enabled=self.enabled.isChecked())


# ------------------------------------------------------------------ 接收规则
class ReceiveRuleDialog(_BaseDialog):
    """触发绑定：什么报文 / 哪个数据点变化会触发哪条流程。"""

    def __init__(self, parent, rule: ReceiveRule, devices: list[tuple[str, str]],
                 flows: list[str], parse_rules: list[str] | None = None,
                 points: dict[str, list[str]] | None = None) -> None:
        super().__init__(parent, "触发规则", 620)
        self._points = points or {}
        form = QFormLayout(self.body)
        self.name = QLineEdit(rule.name)
        self.device = QComboBox()
        self.device.addItem(tr("（任意设备）"), "")
        for dev_id, label in devices:
            self.device.addItem(label, dev_id)
        self.device.setCurrentIndex(max(0, self.device.findData(rule.device)))
        self.source = QComboBox()
        for key, label in RULE_SOURCE_LABELS.items():
            self.source.addItem(label, key)
        self.source.setCurrentIndex(max(0, self.source.findData(rule.source)))
        self.match = QComboBox()
        self.pattern = QLineEdit(rule.pattern)
        self.point = QComboBox(); self.point.setEditable(True)
        self.address = QSpinBox(); self.address.setRange(0, 65535); self.address.setValue(int(rule.address))
        self.value = QDoubleSpinBox(); self.value.setRange(-1e9, 1e9); self.value.setDecimals(3)
        self.value.setValue(float(rule.value))
        self.value2 = QDoubleSpinBox(); self.value2.setRange(-1e9, 1e9); self.value2.setDecimals(3)
        self.value2.setValue(float(rule.value2))
        self.action = QComboBox()
        for key, label in ACTION_LABELS.items():
            self.action.addItem(label, key)
        self.action.setCurrentIndex(max(0, self.action.findData(rule.action)))
        self.flow = QComboBox(); self.flow.addItems(flows or ["main"]); self.flow.setCurrentText(rule.flow)
        self.variable = QLineEdit(rule.variable)
        self.parse = QComboBox(); self.parse.addItem("", "")
        for name in parse_rules or []:
            self.parse.addItem(name, name)
        self.parse.setCurrentText(rule.parse)
        self.request_field = QLineEdit(rule.request_field)
        self.busy_policy = QComboBox()
        for key, label in BUSY_POLICY_LABELS.items():
            self.busy_policy.addItem(label, key)
        self.busy_policy.setCurrentIndex(max(0, self.busy_policy.findData(rule.busy_policy)))
        self.max_queue = QSpinBox(); self.max_queue.setRange(1, 1000); self.max_queue.setValue(int(rule.max_queue))
        self.enabled = QCheckBox(tr("启用")); self.enabled.setChecked(bool(rule.enabled))

        self._rows = [("名称", self.name), ("设备", self.device), ("触发来源", self.source),
                      ("条件", self.match), ("文本内容", self.pattern), ("数据点", self.point),
                      ("寄存器地址（没有数据点时）", self.address), ("比较值", self.value),
                      ("区间上界", self.value2), ("动作", self.action), ("流程", self.flow),
                      ("全局变量", self.variable), ("解析规则", self.parse),
                      ("请求编号字段", self.request_field), ("流程忙时", self.busy_policy),
                      ("排队上限", self.max_queue), ("", self.enabled)]
        self._labels: dict[QWidget, QLabel] = {}
        for label, w in self._rows:
            lbl = QLabel(tr(label))
            form.addRow(lbl, w)
            self._labels[w] = lbl
        form.addRow(_hint("触发位一直保持为 1 时**不会**重复触发：必须先回到不满足条件，"
                          "才会再次产生一次触发。"))
        self.source.currentIndexChanged.connect(lambda _: self._sync())
        self.match.currentIndexChanged.connect(lambda _: self._sync_visibility())
        self.action.currentIndexChanged.connect(lambda _: self._sync_visibility())
        self.device.currentIndexChanged.connect(lambda _: self._sync_points())
        self._sync(initial=rule.match)

    def _sync(self, initial: str = "") -> None:
        want = initial or self.match.currentData() or ""
        self.match.blockSignals(True)
        self.match.clear()
        table = TEXT_MATCH_LABELS if self.source.currentData() == "message" else DATA_MATCH_LABELS
        for key, label in table.items():
            self.match.addItem(label, key)
        idx = self.match.findData(want)
        self.match.setCurrentIndex(idx if idx >= 0 else 0)
        self.match.blockSignals(False)
        self._sync_points()
        self._sync_visibility()

    def _sync_points(self) -> None:
        current = self.point.currentText()
        self.point.clear()
        self.point.addItems([""] + self._points.get(self.device.currentData() or "", []))
        self.point.setCurrentText(current)

    def _show(self, w: QWidget, flag: bool) -> None:
        w.setVisible(flag)
        self._labels[w].setVisible(flag)

    def _sync_visibility(self) -> None:
        is_msg = self.source.currentData() == "message"
        match = self.match.currentData() or ""
        self._show(self.pattern, is_msg and match != "any")
        self._show(self.point, not is_msg)
        self._show(self.address, not is_msg)
        self._show(self.value, (not is_msg) and match in ("equals", "not_equals", "greater",
                                                          "less", "in_range"))
        self._show(self.value2, (not is_msg) and match == "in_range")
        trigger = self.action.currentData() == "trigger_flow"
        self._show(self.flow, trigger)
        self._show(self.busy_policy, trigger)
        self._show(self.max_queue, trigger)
        self._show(self.request_field, trigger)
        self._show(self.parse, trigger and is_msg)
        self._show(self.variable, not trigger)

    def validate(self) -> str:
        if not self.name.text().strip():
            return "规则名称不能为空。"
        if self.source.currentData() == "message" and self.match.currentData() == "regex":
            import re
            try:
                re.compile(self.pattern.text())
            except re.error as e:
                return f"正则有误：{e}"
        if self.source.currentData() == "message" and self.match.currentData() != "any" \
                and not self.pattern.text():
            return "文本内容不能为空（或把条件改成“收到任何报文”）。"
        if self.source.currentData() == "datapoint" and not self.point.currentText().strip() \
                and self.address.value() == 0 and self.match.currentData() not in ("rising", "changed"):
            return "请填数据点名称，或填寄存器地址。"
        if self.action.currentData() == "set_variable" and not self.variable.text().strip():
            return "动作是“写入全局变量”时，必须填变量名。"
        return ""

    def result(self) -> ReceiveRule:
        return ReceiveRule(id="", name=self.name.text().strip(), device=self.device.currentData() or "",
                           source=self.source.currentData(), match=self.match.currentData(),
                           pattern=self.pattern.text(), point=self.point.currentText().strip(),
                           address=self.address.value(), value=self.value.value(),
                           value2=self.value2.value(), action=self.action.currentData(),
                           flow=self.flow.currentText(), variable=self.variable.text().strip(),
                           parse=self.parse.currentText().strip(),
                           request_field=self.request_field.text().strip(),
                           busy_policy=self.busy_policy.currentData(),
                           max_queue=self.max_queue.value(), enabled=self.enabled.isChecked())


# ------------------------------------------------------------------ 发送规则
class SendRuleDialog(_BaseDialog):
    """流程结束时发什么、发给谁。"""

    def __init__(self, parent, rule: SendRule, devices: list[tuple[str, str]], flows: list[str],
                 format_rules: list[str] | None = None, points: dict[str, list[str]] | None = None) -> None:
        super().__init__(parent, "发送规则", 640)
        self._regs = list(rule.registers or [])
        self._points_cfg = list(rule.points or [])
        form = QFormLayout(self.body)
        self.name = QLineEdit(rule.name)
        self.device = QComboBox()
        for dev_id, label in devices:
            self.device.addItem(label, dev_id)
        self.device.setCurrentIndex(max(0, self.device.findData(rule.device)))
        self.flow = QComboBox(); self.flow.addItem(tr("（任意流程）"), "")
        for f in flows:
            self.flow.addItem(f, f)
        self.flow.setCurrentIndex(max(0, self.flow.findData(rule.flow)))
        self.when = QComboBox()
        for key, label in WHEN_LABELS.items():
            self.when.addItem(label, key)
        self.when.setCurrentIndex(max(0, self.when.findData(rule.when)))
        self.target = QComboBox()
        for key, label in SEND_TARGET_LABELS.items():
            self.target.addItem(label, key)
        self.target.setCurrentIndex(max(0, self.target.findData(rule.target)))
        self.format = QComboBox(); self.format.addItem("", "")
        for name in format_rules or []:
            self.format.addItem(name, name)
        self.format.setCurrentText(rule.format)
        self.template = QLineEdit(rule.template)
        self.template.setToolTip("{status} {ok} {ng} {run_id} {flow} {duration_ms} {request_id} "
                                 "{out.名字} {var.名字} {field.名字} {node[节点名].端口}")
        self.points = QPlainTextEdit(json.dumps(self._points_cfg, ensure_ascii=False, indent=1)
                                     if self._points_cfg else "")
        self.points.setPlaceholderText('[{"point": "x", "source": "out.width"}]')
        self.points.setMaximumHeight(80)
        self.registers = QPlainTextEdit(json.dumps(self._regs, ensure_ascii=False, indent=1)
                                        if self._regs else "")
        self.registers.setPlaceholderText('[{"address": 10, "expr": "{out.count}", "kind": "int16"}]')
        self.registers.setMaximumHeight(80)
        self.enabled = QCheckBox(tr("启用")); self.enabled.setChecked(bool(rule.enabled))
        for label, w in [("名称", self.name), ("设备", self.device), ("流程", self.flow),
                         ("发送条件", self.when), ("发给谁", self.target),
                         ("格式化规则", self.format), ("文本模板（没有格式化规则时用）", self.template),
                         ("写数据点", self.points), ("写寄存器（按裸地址，老方案）", self.registers),
                         ("", self.enabled)]:
            form.addRow(tr(label), w)
        form.addRow(_hint("执行顺序：先写数据点、再写寄存器、最后发报文。" + SOURCE_HELP))
        hint_points = points or {}
        self.device.currentIndexChanged.connect(
            lambda _: self.points.setToolTip("该设备的数据点：" +
                                             ("、".join(hint_points.get(self.device.currentData() or "", []))
                                              or "（没有配置数据点）")))
        self.device.currentIndexChanged.emit(self.device.currentIndex())

    def validate(self) -> str:
        if not self.name.text().strip():
            return "规则名称不能为空。"
        if self.device.currentData() in (None, ""):
            return "请选择一台设备。"
        for widget, label, key in ((self.points, "写数据点", "point"),
                                   (self.registers, "写寄存器", "address")):
            text = widget.toPlainText().strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                return f"{label} 的 JSON 有误：{e.msg}（第 {e.lineno} 行第 {e.colno} 列）"
            if not isinstance(parsed, list) or any(not isinstance(x, dict) for x in parsed):
                return f"{label} 应该是一个由对象组成的列表。"
            if any(key not in x for x in parsed):
                return f"{label} 的每一项都要有 {key!r} 字段。"
        if not self.format.currentText().strip() and not self.template.text() \
                and not self.points.toPlainText().strip() and not self.registers.toPlainText().strip():
            return "这条规则什么都不发：请至少填格式化规则、文本模板、数据点或寄存器中的一项。"
        return ""

    def result(self) -> SendRule:
        def _json(widget):
            text = widget.toPlainText().strip()
            return json.loads(text) if text else []
        return SendRule(id="", name=self.name.text().strip(), device=self.device.currentData(),
                        flow=self.flow.currentData() or "", when=self.when.currentData(),
                        template=self.template.text(), format=self.format.currentText().strip(),
                        registers=_json(self.registers), points=_json(self.points),
                        target=self.target.currentData(), enabled=self.enabled.isChecked())


# ------------------------------------------------------------------ 解析 / 格式化
class _FieldTableDialog(_BaseDialog):
    """带一张字段表的对话框（解析规则与格式化规则共用表格的增删改）。"""

    columns: tuple[str, ...] = ()

    def _make_table(self) -> QTableWidget:
        t = QTableWidget(0, len(self.columns))
        t.setHorizontalHeaderLabels([tr(c) for c in self.columns])
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setAlternatingRowColors(True)
        return t

    def _bar(self, add, remove, up=None) -> QHBoxLayout:
        bar = QHBoxLayout()
        for text, slot in (("添加字段", add), ("删除字段", remove)):
            b = QPushButton(tr(text))
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        return bar


class ParseRuleDialog(_FieldTableDialog):
    """解析规则：把报文拆成命名字段，供触发参数与流程使用。"""

    columns = ("字段名", "下标/分组/路径/起始", "长度", "类型", "缩放", "必填", "默认值")

    def __init__(self, parent, rule: ParseRule | None = None, used_names: list[str] | None = None) -> None:
        super().__init__(parent, "解析规则", 720)
        r = rule or ParseRule(name="parse")
        self._used = [n for n in (used_names or []) if n != r.name]
        lay = QVBoxLayout(self.body)
        form = QFormLayout()
        self.name = QLineEdit(r.name)
        self.kind = QComboBox()
        for key, label in PARSE_LABELS.items():
            self.kind.addItem(label, key)
        self.kind.setCurrentIndex(max(0, self.kind.findData(r.kind)))
        self.separator = QLineEdit(r.separator)
        self.pattern = QLineEdit(r.pattern)
        self.encoding = QLineEdit(r.encoding)
        self.strip = QCheckBox(tr("去掉首尾空白")); self.strip.setChecked(bool(r.strip))
        self.to_variables = QCheckBox(tr("同时写入全局变量"))
        self.to_variables.setChecked(bool(r.to_variables))
        for label, w in [("名称", self.name), ("解析方式", self.kind), ("分隔符", self.separator),
                         ("正则表达式", self.pattern), ("文本编码", self.encoding),
                         ("", self.strip), ("", self.to_variables)]:
            form.addRow(tr(label), w)
        lay.addLayout(form)
        self.table = self._make_table()
        lay.addLayout(self._bar(self._add, self._remove))
        lay.addWidget(self.table)
        lay.addWidget(_hint("按分隔符切分时第二列是**第几段**（0 开始）；正则分组时填分组名或分组号；"
                            "JSON 时填点号路径（如 req.items[0].id）；定长切片时填起始字节。"))
        for f in r.fields:
            self._append(f)
        if not r.fields:
            self._append(ParseField(name="request_id", index=1, dtype="int"))

    def _append(self, f: ParseField) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        locator = {"delimited": str(f.index), "regex": f.group, "json": f.path,
                   "fixed_fields": str(f.start)}.get(self.kind.currentData(), str(f.index))
        dtype = QComboBox()
        dtype.addItems(list(VALUE_TYPES) + list(SCALAR_FORMATS))
        dtype.setCurrentText(f.dtype)
        required = QCheckBox(); required.setChecked(bool(f.required))
        for col, value in enumerate([f.name, locator, str(f.length), None, str(f.scale), None,
                                     "" if f.default is None else str(f.default)]):
            if value is None:
                continue
            self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.setCellWidget(row, 3, dtype)
        self.table.setCellWidget(row, 5, required)

    def _add(self) -> None:
        self._append(ParseField(name=f"field{self.table.rowCount() + 1}"))

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _cell(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item else ""

    def validate(self) -> str:
        if not self.name.text().strip():
            return "规则名称不能为空。"
        if self.name.text().strip() in self._used:
            return f"已经有一条规则叫「{self.name.text().strip()}」了。"
        if self.kind.currentData() == "regex":
            import re
            try:
                re.compile(self.pattern.text())
            except re.error as e:
                return f"正则有误：{e}"
        if self.table.rowCount() == 0:
            return "至少要有一个字段。"
        names = []
        for row in range(self.table.rowCount()):
            name = self._cell(row, 0)
            if not name:
                return f"第 {row + 1} 行：字段名不能为空。"
            if name in names:
                return f"字段名 {name!r} 重复。"
            names.append(name)
            if self.kind.currentData() in ("delimited", "fixed_fields"):
                loc = self._cell(row, 1)
                if not loc.lstrip("-").isdigit():
                    return f"第 {row + 1} 行：{'下标' if self.kind.currentData() == 'delimited' else '起始字节'}要填数字。"
        try:
            self.result()
        except (ValueError, TypeError) as e:
            return str(e)
        return ""

    def result(self) -> ParseRule:
        kind = self.kind.currentData()
        fields = []
        for row in range(self.table.rowCount()):
            loc = self._cell(row, 1)
            dtype_w = self.table.cellWidget(row, 3)
            req_w = self.table.cellWidget(row, 5)
            default = self._cell(row, 6)
            fields.append(ParseField(
                name=self._cell(row, 0),
                index=int(loc) if kind == "delimited" and loc.lstrip("-").isdigit() else 0,
                group=loc if kind == "regex" else "",
                path=loc if kind == "json" else "",
                start=int(loc) if kind == "fixed_fields" and loc.lstrip("-").isdigit() else 0,
                length=int(self._cell(row, 2) or 0),
                dtype=dtype_w.currentText() if dtype_w else "string",
                scale=float(self._cell(row, 4) or 1.0),
                required=bool(req_w.isChecked()) if req_w else True,
                default=default or None))
        return ParseRule(name=self.name.text().strip(), kind=kind, separator=self.separator.text(),
                         pattern=self.pattern.text(), encoding=self.encoding.text().strip() or "utf-8",
                         strip=self.strip.isChecked(), to_variables=self.to_variables.isChecked(),
                         fields=fields)


class FormatRuleDialog(_FieldTableDialog):
    """格式化规则：把流程结果拼成文本、JSON 或二进制。"""

    columns = ("字段名", "取值来源", "类型", "小数位", "缩放", "宽度", "对齐", "默认值")

    def __init__(self, parent, rule: FormatRule | None = None, used_names: list[str] | None = None,
                 published: list[str] | None = None, variables: list[str] | None = None) -> None:
        super().__init__(parent, "格式化规则", 760)
        r = rule or FormatRule(name="format")
        self._used = [n for n in (used_names or []) if n != r.name]
        self._suggest = (list(SOURCE_SPECIALS) + [f"out.{n}" for n in (published or [])]
                         + [f"var.{n}" for n in (variables or [])])
        lay = QVBoxLayout(self.body)
        form = QFormLayout()
        self.name = QLineEdit(r.name)
        self.kind = QComboBox()
        for key, label in FORMAT_LABELS.items():
            self.kind.addItem(label, key)
        self.kind.setCurrentIndex(max(0, self.kind.findData(r.kind)))
        self.template = QLineEdit(r.template)
        self.separator = QLineEdit(r.separator)
        self.prefix = QLineEdit(r.prefix)
        self.suffix = QLineEdit(r.suffix)
        self.terminator = QLineEdit(r.terminator)
        self.encoding = QLineEdit(r.encoding)
        self.missing = QLineEdit(r.missing)
        for label, w in [("名称", self.name), ("格式", self.kind),
                         ("文本模板（填了就优先用它）", self.template), ("分隔符", self.separator),
                         ("前缀", self.prefix), ("后缀", self.suffix), ("结束符", self.terminator),
                         ("文本编码", self.encoding), ("取不到值时填", self.missing)]:
            form.addRow(tr(label), w)
        lay.addLayout(form)
        self.table = self._make_table()
        lay.addLayout(self._bar(self._add, self._remove))
        lay.addWidget(self.table)
        lay.addWidget(_hint(SOURCE_HELP))
        for f in r.fields:
            self._append(f)
        if not r.fields:
            self._append(FormatField(name="status", source="status"))

    def _append(self, f: FormatField) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        source = QComboBox(); source.setEditable(True)
        source.addItems([""] + self._suggest)
        source.setCurrentText(f.source)
        dtype = QComboBox()
        dtype.addItems(list(VALUE_TYPES) + list(SCALAR_FORMATS))
        dtype.setCurrentText(f.dtype)
        align = QComboBox(); align.addItems(["left", "right"]); align.setCurrentText(f.align)
        for col, value in enumerate([f.name, None, None, str(f.decimals), str(f.scale),
                                     str(f.width), None, "" if f.default is None else str(f.default)]):
            if value is None:
                continue
            self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.setCellWidget(row, 1, source)
        self.table.setCellWidget(row, 2, dtype)
        self.table.setCellWidget(row, 6, align)

    def _add(self) -> None:
        self._append(FormatField(name=f"field{self.table.rowCount() + 1}"))

    def _remove(self) -> None:
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _cell(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item else ""

    def validate(self) -> str:
        if not self.name.text().strip():
            return "规则名称不能为空。"
        if self.name.text().strip() in self._used:
            return f"已经有一条规则叫「{self.name.text().strip()}」了。"
        if not self.template.text() and self.table.rowCount() == 0:
            return "既没有文本模板，也没有字段。"
        for row in range(self.table.rowCount()):
            if not self._cell(row, 0):
                return f"第 {row + 1} 行：字段名不能为空。"
            src = self.table.cellWidget(row, 1)
            if src is not None and not src.currentText().strip():
                return f"第 {row + 1} 行：取值来源不能为空。"
            for col, label in ((3, "小数位"), (4, "缩放"), (5, "宽度")):
                text = self._cell(row, col)
                if text:
                    try:
                        float(text)
                    except ValueError:
                        return f"第 {row + 1} 行：{label} 要填数字。"
        if self.kind.currentData() == "binary":
            for row in range(self.table.rowCount()):
                dtype = self.table.cellWidget(row, 2)
                if dtype is not None and dtype.currentText() in ("int", "float"):
                    return (f"第 {row + 1} 行：二进制格式要指明确定宽度的类型"
                            f"（int16 / uint16 / int32 / float32 …），不能用 int / float。")
        try:
            self.result()
        except (ValueError, TypeError) as e:
            return str(e)
        return ""

    def result(self) -> FormatRule:
        fields = []
        for row in range(self.table.rowCount()):
            src = self.table.cellWidget(row, 1)
            dtype = self.table.cellWidget(row, 2)
            align = self.table.cellWidget(row, 6)
            fields.append(FormatField(
                name=self._cell(row, 0), source=src.currentText().strip() if src else "",
                dtype=dtype.currentText() if dtype else "string",
                decimals=int(float(self._cell(row, 3) or 3)),
                scale=float(self._cell(row, 4) or 1.0),
                width=int(float(self._cell(row, 5) or 0)),
                align=align.currentText() if align else "left",
                default=self._cell(row, 7)))
        return FormatRule(name=self.name.text().strip(), kind=self.kind.currentData(),
                          template=self.template.text(), separator=self.separator.text(),
                          prefix=self.prefix.text(), suffix=self.suffix.text(),
                          terminator=self.terminator.text(),
                          encoding=self.encoding.text().strip() or "utf-8",
                          missing=self.missing.text(), fields=fields)


# ------------------------------------------------------------------ 检测握手
class HandshakeDialog(_BaseDialog):
    """检测握手：把一台设备和一条流程绑成完整的请求—执行—回传—确认闭环。"""

    def __init__(self, parent, config: HandshakeConfig | None = None,
                 devices: list[tuple[str, str]] | None = None, flows: list[str] | None = None,
                 format_rules: list[str] | None = None, points: dict[str, list[str]] | None = None,
                 used_names: list[str] | None = None) -> None:
        super().__init__(parent, "检测握手", 700)
        c = config or HandshakeConfig()
        self._used = [n for n in (used_names or []) if n != c.name]
        self._points = points or {}
        lay = QVBoxLayout(self.body)
        head = QFormLayout()
        self.name = QLineEdit(c.name)
        self.device = QComboBox()
        for dev_id, label in devices or []:
            self.device.addItem(label, dev_id)
        self.device.setCurrentIndex(max(0, self.device.findData(c.device)))
        self.flow = QComboBox(); self.flow.addItems(flows or ["main"]); self.flow.setCurrentText(c.flow)
        self.mode = QComboBox()
        self.mode.addItem(tr("寄存器信号（Modbus / MC / S7）"), "register")
        self.mode.addItem(tr("文本报文（TCP / 串口 / UDP）"), "text")
        self.mode.setCurrentIndex(max(0, self.mode.findData(c.mode)))
        self.enabled = QCheckBox(tr("启用")); self.enabled.setChecked(bool(c.enabled))
        for label, w in [("名称", self.name), ("设备", self.device), ("流程", self.flow),
                         ("模式", self.mode), ("", self.enabled)]:
            head.addRow(tr(label), w)
        lay.addLayout(head)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)
        # --- 寄存器信号 ---
        self.reg_page = QWidget()
        reg = QFormLayout(self.reg_page)
        self._signals: dict[str, QComboBox] = {}
        for key, label in [("ready", "就绪（Ready）"), ("busy", "忙（Busy）"), ("done", "完成（Done）"),
                           ("result", "判定结果（OK/NG）"), ("error_code", "错误码"),
                           ("request_id", "请求编号（对端写）"), ("done_id", "完成编号"),
                           ("ack", "对端确认位")]:
            box = QComboBox(); box.setEditable(True)
            self._signals[key] = box
            box.setCurrentText(getattr(c, key))
            reg.addRow(tr(label), box)
        self.ack_value = QSpinBox(); self.ack_value.setRange(-32768, 65535); self.ack_value.setValue(int(c.ack_value))
        self.ok_value = QSpinBox(); self.ok_value.setRange(-32768, 65535); self.ok_value.setValue(int(c.ok_value))
        self.ng_value = QSpinBox(); self.ng_value.setRange(-32768, 65535); self.ng_value.setValue(int(c.ng_value))
        for label, w in [("确认位的值", self.ack_value), ("OK 写入值", self.ok_value),
                         ("NG 写入值", self.ng_value)]:
            reg.addRow(tr(label), w)
        reg.addRow(_hint("填数据点名称，或直接填地址（如 40011 / 4x10）。留空表示不使用该信号。"))
        self.tabs.addTab(self.reg_page, tr("寄存器信号"))
        # --- 文本报文 ---
        self.text_page = QWidget()
        txt = QFormLayout(self.text_page)
        self._formats: dict[str, QComboBox] = {}
        for key, label in [("accept_format", "受理回复（可留空）"), ("result_format", "结果回复"),
                           ("busy_format", "忙回复"), ("error_format", "错误回复")]:
            box = QComboBox(); box.addItem("", "")
            for name in format_rules or []:
                box.addItem(name, name)
            box.setCurrentText(getattr(c, key))
            self._formats[key] = box
            txt.addRow(tr(label), box)
        txt.addRow(_hint("回复一定发给**发起请求的那个**客户端；格式化规则里可以用 "
                         "{request_id}、{error_code} 和 out.名字。"))
        self.tabs.addTab(self.text_page, tr("文本报文"))
        # --- 策略 ---
        pol_page = QWidget()
        pol = QFormLayout(pol_page)
        self.require_ack = QCheckBox(tr("需要对端确认后才复位")); self.require_ack.setChecked(bool(c.require_ack))
        self.auto_reset_s = QDoubleSpinBox(); self.auto_reset_s.setRange(0, 3600); self.auto_reset_s.setDecimals(1)
        self.auto_reset_s.setValue(float(c.auto_reset_s))
        self.run_timeout_s = QDoubleSpinBox(); self.run_timeout_s.setRange(0, 3600); self.run_timeout_s.setDecimals(1)
        self.run_timeout_s.setValue(float(c.run_timeout_s))
        self.busy_policy = QComboBox()
        for key, label in BUSY_POLICY_LABELS.items():
            self.busy_policy.addItem(label, key)
        self.busy_policy.setCurrentIndex(max(0, self.busy_policy.findData(c.busy_policy)))
        self.max_queue = QSpinBox(); self.max_queue.setRange(1, 100); self.max_queue.setValue(int(c.max_queue))
        self.dedup_window_s = QDoubleSpinBox(); self.dedup_window_s.setRange(0, 600); self.dedup_window_s.setDecimals(1)
        self.dedup_window_s.setValue(float(c.dedup_window_s))
        self.dedup_action = QComboBox()
        for key, label in DEDUP_LABELS.items():
            self.dedup_action.addItem(label, key)
        self.dedup_action.setCurrentIndex(max(0, self.dedup_action.findData(c.dedup_action)))
        self.ng_error_code = QCheckBox(tr("判定 NG 时也写错误码"))
        self.ng_error_code.setChecked(bool(c.ng_error_code))
        for label, w in [("", self.require_ack), ("等确认超时自动复位（秒，0=一直等）", self.auto_reset_s),
                         ("流程超时（秒，0=不判超时）", self.run_timeout_s),
                         ("流程忙时", self.busy_policy), ("排队上限", self.max_queue),
                         ("重复请求判定窗口（秒，0=不去重）", self.dedup_window_s),
                         ("重复请求怎么办", self.dedup_action), ("", self.ng_error_code)]:
            pol.addRow(tr(label), w)
        pol.addRow(_hint("超时只上报错误码，**不会**重发触发或重复写入——那可能让对端执行两次动作。"
                         "断线重连后，断线前那次任务的结果会被丢弃。错误码含义："
                         + "、".join(f"{k}={v}" for k, v in ERROR_TEXT.items())))
        self.tabs.addTab(pol_page, tr("策略"))
        self.mode.currentIndexChanged.connect(lambda _: self._sync())
        self.device.currentIndexChanged.connect(lambda _: self._sync_points())
        self._sync()

    def _sync(self) -> None:
        register = self.mode.currentData() == "register"
        self.tabs.setTabEnabled(0, register)
        self.tabs.setTabEnabled(1, not register)
        self.tabs.setCurrentIndex(0 if register else 1)
        self._sync_points()

    def _sync_points(self) -> None:
        names = self._points.get(self.device.currentData() or "", [])
        for key, box in self._signals.items():
            current = box.currentText()
            box.clear()
            box.addItems([""] + names)
            box.setCurrentText(current)

    def validate(self) -> str:
        if not self.name.text().strip():
            return "握手名称不能为空。"
        if self.name.text().strip() in self._used:
            return f"已经有一个握手叫「{self.name.text().strip()}」了。"
        if self.device.currentData() in (None, ""):
            return "请选择一台设备。"
        if self.mode.currentData() == "register":
            if not any(box.currentText().strip() for box in self._signals.values()):
                return "寄存器模式至少要配一个信号（通常是 完成 Done）。"
            if self.require_ack.isChecked() and not self._signals["ack"].currentText().strip():
                return "勾选了“需要对端确认”，必须指定确认位。"
        else:
            if not self._formats["result_format"].currentText().strip():
                return "文本模式必须指定结果回复用的格式化规则。"
        return ""

    def result(self) -> HandshakeConfig:
        return HandshakeConfig(
            id="", name=self.name.text().strip(), device=self.device.currentData(),
            flow=self.flow.currentText(), mode=self.mode.currentData(),
            enabled=self.enabled.isChecked(),
            **{k: v.currentText().strip() for k, v in self._signals.items()},
            ack_value=self.ack_value.value(), ok_value=self.ok_value.value(),
            ng_value=self.ng_value.value(),
            **{k: v.currentText().strip() for k, v in self._formats.items()},
            require_ack=self.require_ack.isChecked(), auto_reset_s=self.auto_reset_s.value(),
            run_timeout_s=self.run_timeout_s.value(), busy_policy=self.busy_policy.currentData(),
            max_queue=self.max_queue.value(), dedup_window_s=self.dedup_window_s.value(),
            dedup_action=self.dedup_action.currentData(),
            ng_error_code=self.ng_error_code.isChecked())

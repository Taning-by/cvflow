"""Communication panel: devices, receive rules, send rules, monitor and manual send."""
from __future__ import annotations

import json
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..comm.manager import DEVICE_KIND_LABELS, DEVICE_KINDS, REGISTER_MATCHES, TEXT_MATCHES, CommManager, ReceiveRule, SendRule
from .i18n import tr


# ------------------------------------------------------------------ dialogs
class DeviceDialog(QDialog):
    def __init__(self, parent=None, name: str = "", kind: str = "tcp_server", config: dict | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Communication device"))
        self.setMinimumWidth(520)
        self._fields: dict[str, QWidget] = {}
        lay = QVBoxLayout(self)
        top = QFormLayout()
        self.name = QLineEdit(name or "plc")
        self.kind = QComboBox()
        for k in sorted(DEVICE_KINDS):
            self.kind.addItem(DEVICE_KIND_LABELS.get(k, k), k)
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        self.kind.setEnabled(not name)
        top.addRow(tr("Name"), self.name)
        top.addRow(tr("Kind"), self.kind)
        lay.addLayout(top)
        self._form_host = QWidget()
        lay.addWidget(self._form_host)
        self._config = dict(config or {})
        self.kind.currentIndexChanged.connect(lambda _: self._build_form())
        self._build_form()
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _build_form(self) -> None:
        old = self._form_host.layout()
        if old is not None:
            while old.count():
                w = old.takeAt(0).widget()
                if w:
                    w.deleteLater()
            QWidget().setLayout(old)
        form = QFormLayout(self._form_host)
        self._fields.clear()
        cls = DEVICE_KINDS[self.kind.currentData()]
        for key, kind, label, default, desc in cls.config_schema:
            val = self._config.get(key, default)
            if kind == "int":
                w = QSpinBox(); w.setRange(-1, 2_000_000_000); w.setValue(int(val))
            elif kind == "float":
                w = QDoubleSpinBox(); w.setRange(0, 1e9); w.setDecimals(3); w.setValue(float(val))
            elif kind == "bool":
                w = QCheckBox(); w.setChecked(bool(val))
            elif kind.startswith("enum:"):
                w = QComboBox(); w.addItems(kind[5:].split(",")); w.setCurrentText(str(val))
            else:
                w = QLineEdit(str(val))
            if desc:
                w.setToolTip(tr(desc))
            self._fields[key] = w
            form.addRow(tr(label), w)

    def result(self) -> tuple[str, str, dict]:
        cfg = {}
        for key, w in self._fields.items():
            if isinstance(w, QSpinBox) or isinstance(w, QDoubleSpinBox):
                cfg[key] = w.value()
            elif isinstance(w, QCheckBox):
                cfg[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                cfg[key] = w.currentText()
            else:
                cfg[key] = w.text()
        return self.name.text().strip(), self.kind.currentData(), cfg


class ReceiveRuleDialog(QDialog):
    def __init__(self, parent, rule: ReceiveRule, devices: list[str], flows: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Receive rule"))
        form = QFormLayout(self)
        self.name = QLineEdit(rule.name)
        self.device = QComboBox(); self.device.addItems([""] + devices); self.device.setCurrentText(rule.device)
        self.match = QComboBox(); self.match.addItems(list(TEXT_MATCHES) + list(REGISTER_MATCHES)); self.match.setCurrentText(rule.match)
        self.pattern = QLineEdit(rule.pattern)
        self.address = QSpinBox(); self.address.setRange(0, 65535); self.address.setValue(int(rule.address))
        self.value = QSpinBox(); self.value.setRange(-32768, 65535); self.value.setValue(int(rule.value))
        self.action = QComboBox(); self.action.addItems(["trigger_flow", "set_variable"]); self.action.setCurrentText(rule.action)
        self.flow = QComboBox(); self.flow.addItems(flows); self.flow.setCurrentText(rule.flow)
        self.variable = QLineEdit(rule.variable)
        self.enabled = QCheckBox(); self.enabled.setChecked(rule.enabled)
        for lbl, w in [("Name", self.name), ("Device (empty = any)", self.device), ("Match", self.match),
                       ("Text pattern", self.pattern), ("Register address", self.address), ("Register value", self.value),
                       ("Action", self.action), ("Flow", self.flow), ("Variable", self.variable), ("Enabled", self.enabled)]:
            form.addRow(tr(lbl), w)
        hint = QLabel(tr("Text matches apply to TCP/UDP/serial frames; register_* matches apply to Modbus devices "
                         "(rising = 0→non-zero)."))
        hint.setWordWrap(True); hint.setStyleSheet("color:#888")
        form.addRow(hint)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result(self) -> ReceiveRule:
        return ReceiveRule(name=self.name.text(), device=self.device.currentText(), match=self.match.currentText(),
                           pattern=self.pattern.text(), address=self.address.value(), value=self.value.value(),
                           action=self.action.currentText(), flow=self.flow.currentText(),
                           variable=self.variable.text(), enabled=self.enabled.isChecked())


class SendRuleDialog(QDialog):
    def __init__(self, parent, rule: SendRule, devices: list[str], flows: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Send rule"))
        form = QFormLayout(self)
        self.name = QLineEdit(rule.name)
        self.device = QComboBox(); self.device.addItems(devices); self.device.setCurrentText(rule.device)
        self.flow = QComboBox(); self.flow.addItems([""] + flows); self.flow.setCurrentText(rule.flow)
        self.when = QComboBox(); self.when.addItems(["always", "ok", "ng", "error"]); self.when.setCurrentText(rule.when)
        self.template = QLineEdit(rule.template)
        self.template.setToolTip("{status} {ok} {ng} {run_id} {flow} {duration_ms} {out.name} {var.name} {node[Node Name].port}")
        self.registers = QPlainTextEdit(json.dumps(rule.registers, indent=1) if rule.registers else "")
        self.registers.setPlaceholderText(tr('Modbus only, JSON list: [{"address": 10, "expr": "{out.count}", "kind": "int16"}]'))
        self.registers.setMaximumHeight(90)
        self.enabled = QCheckBox(); self.enabled.setChecked(rule.enabled)
        for lbl, w in [("Name", self.name), ("Device", self.device), ("Flow (empty = any)", self.flow), ("When", self.when),
                       ("Text template", self.template), ("Register writes", self.registers), ("Enabled", self.enabled)]:
            form.addRow(tr(lbl), w)
        self.err = QLabel(""); self.err.setStyleSheet("color:#ff8a65")
        form.addRow(self.err)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._ok); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _ok(self) -> None:
        try:
            self._regs = json.loads(self.registers.toPlainText()) if self.registers.toPlainText().strip() else []
        except json.JSONDecodeError as e:
            self.err.setText(tr("registers JSON: ") + e.msg)
            return
        self.accept()

    def result(self) -> SendRule:
        return SendRule(name=self.name.text(), device=self.device.currentText(), flow=self.flow.currentText(),
                        when=self.when.currentText(), template=self.template.text(), registers=self._regs,
                        enabled=self.enabled.isChecked())


# ------------------------------------------------------------------ panel
class CommPanel(QWidget):
    config_changed = Signal()

    def __init__(self, manager: CommManager, flows_provider: Callable[[], list[str]], parent=None) -> None:
        super().__init__(parent)
        self.mgr = manager
        self._flows = flows_provider
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        tabs = QTabWidget()
        lay.addWidget(tabs)

        # devices
        dev_w = QWidget(); dl = QVBoxLayout(dev_w); dl.setContentsMargins(2, 2, 2, 2)
        bar = QHBoxLayout()
        for text, slot in [("Add", self._add_device), ("Edit", self._edit_device), ("Remove", self._remove_device),
                           ("Connect", self._connect), ("Disconnect", self._disconnect), ("Connect all", self._connect_all),
                           ("测试连接", self._test_device)]:
            b = QPushButton(tr(text)); b.clicked.connect(slot); bar.addWidget(b)
        bar.addStretch(1)
        self.dev_table = QTableWidget(0, 6)
        self.dev_table.setHorizontalHeaderLabels([tr(h) for h in ("Name", "Kind", "Status", "RX", "TX", "Last error")])
        self.dev_table.horizontalHeader().setStretchLastSection(True)
        self.dev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.dev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.dev_table.setAlternatingRowColors(True)
        self.dev_table.verticalHeader().setVisible(False)
        dl.addLayout(bar); dl.addWidget(self.dev_table)
        tabs.addTab(dev_w, tr("Devices"))

        # receive rules
        rx_w = QWidget(); rl = QVBoxLayout(rx_w); rl.setContentsMargins(2, 2, 2, 2)
        bar = QHBoxLayout()
        for text, slot in [("Add", self._add_rx), ("Edit", self._edit_rx), ("Remove", self._remove_rx)]:
            b = QPushButton(tr(text)); b.clicked.connect(slot); bar.addWidget(b)
        bar.addStretch(1)
        self.rx_table = QTableWidget(0, 6)
        self.rx_table.setHorizontalHeaderLabels([tr(h) for h in ("Name", "Device", "Match", "Pattern / address", "Action", "Flow / variable")])
        self.rx_table.horizontalHeader().setStretchLastSection(True)
        self.rx_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rx_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rx_table.setAlternatingRowColors(True)
        self.rx_table.verticalHeader().setVisible(False)
        self.rx_table.doubleClicked.connect(lambda _: self._edit_rx())
        rl.addLayout(bar); rl.addWidget(self.rx_table)
        tabs.addTab(rx_w, tr("Receive rules (triggers)"))

        # send rules
        tx_w = QWidget(); tl = QVBoxLayout(tx_w); tl.setContentsMargins(2, 2, 2, 2)
        bar = QHBoxLayout()
        for text, slot in [("Add", self._add_tx), ("Edit", self._edit_tx), ("Remove", self._remove_tx)]:
            b = QPushButton(tr(text)); b.clicked.connect(slot); bar.addWidget(b)
        bar.addStretch(1)
        self.tx_table = QTableWidget(0, 5)
        self.tx_table.setHorizontalHeaderLabels([tr(h) for h in ("Name", "Device", "Flow", "When", "Template / registers")])
        self.tx_table.horizontalHeader().setStretchLastSection(True)
        self.tx_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tx_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tx_table.setAlternatingRowColors(True)
        self.tx_table.verticalHeader().setVisible(False)
        self.tx_table.doubleClicked.connect(lambda _: self._edit_tx())
        tl.addLayout(bar); tl.addWidget(self.tx_table)
        tabs.addTab(tx_w, tr("Send rules (results)"))

        # monitor
        mon_w = QWidget(); ml = QVBoxLayout(mon_w); ml.setContentsMargins(2, 2, 2, 2)
        self.monitor = QListWidget()
        row = QHBoxLayout()
        self.send_dev = QComboBox()
        self.send_text = QLineEdit(); self.send_text.setPlaceholderText(tr("text to send (\\n etc. allowed)"))
        send_btn = QPushButton(tr("Send")); send_btn.clicked.connect(self._manual_send)
        self.send_text.returnPressed.connect(self._manual_send)
        clear = QPushButton(tr("Clear")); clear.clicked.connect(self.monitor.clear)
        row.addWidget(self.send_dev); row.addWidget(self.send_text, 1); row.addWidget(send_btn); row.addWidget(clear)
        ml.addWidget(self.monitor); ml.addLayout(row)
        tabs.addTab(mon_w, tr("Monitor"))
        self.refresh()

    # ---- refresh ----
    def set_manager(self, mgr: CommManager) -> None:
        self.mgr = mgr
        self.refresh()

    def refresh(self) -> None:
        devs = list(self.mgr.devices.values())
        self.dev_table.setRowCount(len(devs))
        for r, d in enumerate(devs):
            info = d.info()
            status = tr("connected") if d.connected else tr("disconnected")
            cells = [d.name, DEVICE_KIND_LABELS.get(d.kind, d.kind), status, str(info.get("rx", 0)), str(info.get("tx", 0)), d.last_error]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(("●  " + text) if c == 2 else text)
                if c == 2:
                    it.setForeground(QColor("#3ccf6e") if d.connected else QColor("#8d95a3"))
                self.dev_table.setItem(r, c, it)
        self.rx_table.setRowCount(len(self.mgr.receive_rules))
        for r, rule in enumerate(self.mgr.receive_rules):
            pat = rule.pattern if rule.match in TEXT_MATCHES else f"addr {rule.address}" + (f" == {rule.value}" if rule.match == "register_equals" else "")
            tgt = rule.flow if rule.action == "trigger_flow" else f"var {rule.variable}"
            for c, text in enumerate([rule.name + ("" if rule.enabled else tr(" (off)")), rule.device or "*", rule.match, pat, rule.action, tgt]):
                self.rx_table.setItem(r, c, QTableWidgetItem(text))
        self.tx_table.setRowCount(len(self.mgr.send_rules))
        for r, rule in enumerate(self.mgr.send_rules):
            payload = rule.template if not rule.registers else json.dumps(rule.registers)
            for c, text in enumerate([rule.name + ("" if rule.enabled else tr(" (off)")), rule.device, rule.flow or "*", rule.when, payload]):
                self.tx_table.setItem(r, c, QTableWidgetItem(text))
        cur = self.send_dev.currentText()
        self.send_dev.clear()
        self.send_dev.addItems([d.name for d in devs])
        if cur:
            self.send_dev.setCurrentText(cur)

    def on_event(self, event: str, payload: dict) -> None:
        if event in ("comm_received", "comm_sent", "comm_error", "comm_connected", "comm_disconnected"):
            dev = payload.get("device", "")
            if event == "comm_received":
                line = f"◀ {dev}: {payload.get('text', '')!r}"
            elif event == "comm_sent":
                line = f"▶ {dev}: {payload.get('text', '')!r}"
            elif event == "comm_error":
                line = f"✖ {dev}: {payload.get('error', '')}"
            else:
                line = f"● {dev}: {event.replace('comm_', '')}"
            self.monitor.addItem(line)
            if self.monitor.count() > 500:
                self.monitor.takeItem(0)
            self.monitor.scrollToBottom()
            self.refresh()

    # ---- devices ----
    def _selected_device(self):
        row = self.dev_table.currentRow()
        if row < 0:
            return None
        return self.mgr.devices.get(self.dev_table.item(row, 0).text())

    def _add_device(self) -> None:
        dlg = DeviceDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name, kind, cfg = dlg.result()
            if name:
                self.mgr.add_device(name, kind, cfg)
                self.refresh(); self.config_changed.emit()

    def _edit_device(self) -> None:
        d = self._selected_device()
        if d is None:
            return
        dlg = DeviceDialog(self, d.name, d.kind, d.config)
        if dlg.exec() == QDialog.Accepted:
            name, kind, cfg = dlg.result()
            was = d.connected
            self.mgr.add_device(name, kind, cfg)
            if was:
                try:
                    self.mgr.devices[name].connect()
                except Exception:
                    pass
            self.refresh(); self.config_changed.emit()

    def _remove_device(self) -> None:
        d = self._selected_device()
        if d is not None:
            self.mgr.remove_device(d.name)
            self.refresh(); self.config_changed.emit()

    def _connect(self) -> None:
        d = self._selected_device()
        if d is not None:
            try:
                d.connect()
            except Exception as e:
                d._error(str(e))
            self.refresh()

    def _disconnect(self) -> None:
        d = self._selected_device()
        if d is not None:
            d.disconnect()
            self.refresh()

    def _connect_all(self) -> None:
        self.mgr.connect_all()
        self.refresh()

    def _test_device(self) -> None:
        d = self._selected_device()
        if d is None:
            QMessageBox.information(self, "测试连接", "请先在列表中选择一个设备。")
            return
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            ok, msg = self.mgr.test_connection(d.name)
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh()
        (QMessageBox.information if ok else QMessageBox.warning)(self, f"测试连接 – {d.name}", msg)

    # ---- rules ----
    def _add_rx(self) -> None:
        dlg = ReceiveRuleDialog(self, ReceiveRule(), list(self.mgr.devices), self._flows())
        if dlg.exec() == QDialog.Accepted:
            self.mgr.add_receive_rule(dlg.result()); self.refresh(); self.config_changed.emit()

    def _edit_rx(self) -> None:
        row = self.rx_table.currentRow()
        if row < 0:
            return
        dlg = ReceiveRuleDialog(self, self.mgr.receive_rules[row], list(self.mgr.devices), self._flows())
        if dlg.exec() == QDialog.Accepted:
            self.mgr.receive_rules[row] = dlg.result(); self.refresh(); self.config_changed.emit()

    def _remove_rx(self) -> None:
        row = self.rx_table.currentRow()
        if row >= 0:
            del self.mgr.receive_rules[row]; self.refresh(); self.config_changed.emit()

    def _add_tx(self) -> None:
        dlg = SendRuleDialog(self, SendRule(), list(self.mgr.devices), self._flows())
        if dlg.exec() == QDialog.Accepted:
            self.mgr.add_send_rule(dlg.result()); self.refresh(); self.config_changed.emit()

    def _edit_tx(self) -> None:
        row = self.tx_table.currentRow()
        if row < 0:
            return
        dlg = SendRuleDialog(self, self.mgr.send_rules[row], list(self.mgr.devices), self._flows())
        if dlg.exec() == QDialog.Accepted:
            self.mgr.send_rules[row] = dlg.result(); self.refresh(); self.config_changed.emit()

    def _remove_tx(self) -> None:
        row = self.tx_table.currentRow()
        if row >= 0:
            del self.mgr.send_rules[row]; self.refresh(); self.config_changed.emit()

    def _manual_send(self) -> None:
        dev = self.send_dev.currentText()
        text = self.send_text.text()
        if dev and text:
            self.mgr.send(dev, text.encode().decode("unicode_escape"))

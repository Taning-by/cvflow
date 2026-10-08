"""通信面板：设备、数据点、触发、发送、解析/格式化、检测握手、调试。

界面线程**不做任何阻塞通信**：收发与轮询都在通信层自己的后台线程里，面板靠一个 250 ms 的
定时器批量拉取（设备状态、握手状态、调试日志），所以高频通信不会把界面拖卡。日志有容量
上限，可以暂停滚动、按方向与关键字筛选、导出 CSV。
"""
from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..comm.handshake import HandshakeConfig, STATE_LABELS
from ..comm.manager import (ACTION_LABELS, DATA_MATCH_LABELS, LOG_DIRECTIONS, RULE_SOURCE_LABELS,
                            SEND_TARGET_LABELS, TEXT_MATCH_LABELS, WHEN_LABELS, CommManager,
                            ReceiveRule, SendRule)
from ..comm.modbus import _PointHost
from ..comm.modbus_map import AREA_INFO, DIRECTION_LABELS
from .comm_dialogs import (DataPointDialog, DeviceDialog, FormatRuleDialog, HandshakeDialog,
                           ParseRuleDialog, ReceiveRuleDialog, SendRuleDialog)
from .i18n import tr
from .theme import C, mono_font

REFRESH_MS = 250            # 批量刷新间隔：高频通信下界面也只按这个节奏重画


def _table(headers: list[str], stretch_last: bool = True) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels([tr(h) for h in headers])
    t.horizontalHeader().setStretchLastSection(stretch_last)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    return t


def _bar(*buttons: tuple[str, Callable]) -> tuple[QHBoxLayout, dict[str, QPushButton]]:
    row = QHBoxLayout()
    made: dict[str, QPushButton] = {}
    for text, slot in buttons:
        b = QPushButton(tr(text))
        b.clicked.connect(slot)
        row.addWidget(b)
        made[text] = b
    row.addStretch(1)
    return row, made


def _page(layout_cls=QVBoxLayout) -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    lay = layout_cls(w)
    lay.setContentsMargins(4, 4, 4, 4)
    return w, lay


def _fmt_time(ts: float) -> str:
    if not ts:
        return "—"
    return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int(ts % 1 * 1000):03d}"


def _ago(ts: float) -> str:
    if not ts:
        return "—"
    delta = time.time() - ts
    if delta < 1:
        return "刚刚"
    if delta < 60:
        return f"{delta:.0f} 秒前"
    if delta < 3600:
        return f"{delta / 60:.0f} 分钟前"
    return _fmt_time(ts)


class CommPanel(QWidget):
    config_changed = Signal()

    def __init__(self, manager: CommManager, flows_provider: Callable[[], list[str]], parent=None) -> None:
        super().__init__(parent)
        self.mgr = manager
        self._flows = flows_provider
        self._log_seq = 0
        self._monitor_device = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        self._build_devices()
        self._build_points()
        self._build_receive()
        self._build_send()
        self._build_rules()
        self._build_handshakes()
        self._build_debug()

        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.refresh()

    # ============================================================ 设备
    def _build_devices(self) -> None:
        page, lay = _page()
        bar, _ = _bar(("添加", self._add_device), ("编辑", self._edit_device),
                      ("复制", self._copy_device), ("删除", self._remove_device),
                      ("启用/禁用", self._toggle_device), ("连接", self._connect),
                      ("断开", self._disconnect), ("全部连接", self._connect_all),
                      ("测试连接", self._test_device), ("检查配置", self._validate))
        self.dev_table = _table(["名称", "类型", "状态", "对端", "收", "发", "最近通信", "最后错误"])
        self.dev_table.doubleClicked.connect(lambda _: self._edit_device())
        lay.addLayout(bar)
        lay.addWidget(self.dev_table)
        self.tabs.addTab(page, tr("设备"))

    def _selected_device(self):
        row = self.dev_table.currentRow()
        if row < 0:
            return None
        item = self.dev_table.item(row, 0)
        return self.mgr.resolve(item.data(Qt.UserRole) or item.text()) if item else None

    def _device_choices(self) -> list[tuple[str, str]]:
        return [(d.id, d.name) for d in self.mgr.devices.values()]

    def _point_choices(self) -> dict[str, list[str]]:
        return {d.id: (d.points.names() if isinstance(d, _PointHost) else [])
                for d in self.mgr.devices.values()}

    def _add_device(self) -> None:
        dlg = DeviceDialog(self, used_names=list(self.mgr.devices))
        if dlg.exec() == QDialog.Accepted:
            name, kind, cfg, enabled = dlg.result()
            try:
                self.mgr.add_device(name, kind, cfg, enabled=enabled)
            except Exception as e:
                QMessageBox.warning(self, tr("添加设备"), str(e))
                return
            self._changed()

    def _edit_device(self) -> None:
        dev = self._selected_device()
        if dev is None:
            return
        dlg = DeviceDialog(self, dev.name, dev.kind, dev.config, dev.enabled, dev.id,
                           used_names=list(self.mgr.devices))
        if dlg.exec() != QDialog.Accepted:
            return
        name, _kind, cfg, enabled = dlg.result()
        try:
            if name != dev.name:
                self.mgr.rename_device(dev.id, name)
            self.mgr.update_device(dev.id, config=cfg, enabled=enabled)
        except Exception as e:
            QMessageBox.warning(self, tr("编辑设备"), str(e))
        self._changed()

    def _copy_device(self) -> None:
        dev = self._selected_device()
        if dev is None:
            return
        try:
            self.mgr.duplicate_device(dev.id)
        except Exception as e:
            QMessageBox.warning(self, tr("复制设备"), str(e))
            return
        self._changed()

    def _remove_device(self) -> None:
        dev = self._selected_device()
        if dev is None:
            return
        refs = self.mgr.references(dev.id)
        if refs:
            listed = "\n".join(f"· {r}" for r in refs[:12])
            more = f"\n…… 以及另外 {len(refs) - 12} 处" if len(refs) > 12 else ""
            ans = QMessageBox.question(
                self, tr("删除设备"),
                f"「{dev.name}」还被下面这些地方引用：\n\n{listed}{more}\n\n"
                f"删除后这些引用会失效（界面上会标成红色）。仍要删除吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
        self.mgr.remove_device(dev.id)
        self._changed()

    def _toggle_device(self) -> None:
        dev = self._selected_device()
        if dev is not None:
            self.mgr.set_device_enabled(dev.id, not dev.enabled)
            self._changed()

    def _connect(self) -> None:
        dev = self._selected_device()
        if dev is None:
            return
        ok, msg = self.mgr.connect(dev.id)
        if not ok:
            QMessageBox.warning(self, tr("连接"), msg)
        self.refresh()

    def _disconnect(self) -> None:
        dev = self._selected_device()
        if dev is not None:
            self.mgr.disconnect(dev.id)
            self.refresh()

    def _connect_all(self) -> None:
        errors = self.mgr.connect_all()
        if errors:
            QMessageBox.warning(self, tr("全部连接"), "\n".join(errors))
        self.refresh()

    def _validate(self) -> None:
        """配置自检：引用了不存在的设备/流程/规则、数据点非法，都在这里一次性列出来。

        和 `cvflow serve` 启动时打印的是同一份检查——产线上那条路也会在启动时看到。
        """
        problems = self.mgr.validate()
        if not problems:
            summary = (f"通信配置没有问题。\n\n设备 {len(self.mgr.devices)} 台"
                       f"（启用 {sum(1 for d in self.mgr.devices.values() if d.enabled)} 台）、"
                       f"触发规则 {len(self.mgr.receive_rules)} 条、发送规则 {len(self.mgr.send_rules)} 条、"
                       f"解析规则 {len(self.mgr.parse_rules)} 条、格式化规则 {len(self.mgr.format_rules)} 条、"
                       f"检测握手 {len(self.mgr.handshakes)} 个。")
            QMessageBox.information(self, tr("检查配置"), summary)
            return
        listed = "\n".join(f"· {t}" for t in problems[:20])
        more = f"\n…… 以及另外 {len(problems) - 20} 处" if len(problems) > 20 else ""
        QMessageBox.warning(self, tr("检查配置"),
                            f"发现 {len(problems)} 处问题：\n\n{listed}{more}")

    def _test_device(self) -> None:
        dev = self._selected_device()
        if dev is None:
            QMessageBox.information(self, tr("测试连接"), tr("请先在列表中选择一个设备。"))
            return
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            ok, msg = self.mgr.test_connection(dev.id)
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh()
        (QMessageBox.information if ok else QMessageBox.warning)(self, f"测试连接 – {dev.name}", msg)

    # ============================================================ 数据点
    def _build_points(self) -> None:
        page, lay = _page()
        top = QHBoxLayout()
        top.addWidget(QLabel(tr("设备")))
        self.point_device = QComboBox()
        self.point_device.currentIndexChanged.connect(lambda _: self._refresh_points())
        top.addWidget(self.point_device, 1)
        lay.addLayout(top)
        bar, _ = _bar(("添加", self._add_point), ("编辑", self._edit_point),
                      ("删除", self._remove_point), ("读取一次", self._read_point))
        self.point_table = _table(["名称", "数据区", "协议地址", "参考地址", "类型", "方向",
                                   "轮询(ms)", "字节序", "绑定变量", "当前值"])
        self.point_table.doubleClicked.connect(lambda _: self._edit_point())
        lay.addLayout(bar)
        lay.addWidget(self.point_table)
        lay.addWidget(QLabel(tr("数据点给地址起名字：流程、触发规则和检测握手都按名字引用，改地址不用改流程。")))
        self.tabs.addTab(page, tr("数据点"))

    def _point_host(self):
        dev = self.mgr.resolve(self.point_device.currentData() or "")
        return dev if isinstance(dev, _PointHost) else None

    def _selected_point(self):
        host = self._point_host()
        row = self.point_table.currentRow()
        if host is None or row < 0:
            return None, None
        item = self.point_table.item(row, 0)
        return host, (host.points.get(item.text()) if item else None)

    def _add_point(self) -> None:
        host = self._point_host()
        if host is None:
            QMessageBox.information(self, tr("数据点"), tr("请先选择一台支持数据点的设备（Modbus / MC / S7）。"))
            return
        dlg = DataPointDialog(self, None, host.points.names(), self.mgr.variables.names())
        if dlg.exec() == QDialog.Accepted:
            try:
                host.points.add(dlg.result())
            except Exception as e:
                QMessageBox.warning(self, tr("数据点"), str(e))
                return
            self._changed(self.point_table)

    def _edit_point(self) -> None:
        host, point = self._selected_point()
        if host is None or point is None:
            return
        dlg = DataPointDialog(self, point, host.points.names(), self.mgr.variables.names())
        if dlg.exec() == QDialog.Accepted:
            try:
                host.points.replace(host.points.points.index(point), dlg.result())
            except Exception as e:
                QMessageBox.warning(self, tr("数据点"), str(e))
                return
            self._changed()

    def _remove_point(self) -> None:
        host, point = self._selected_point()
        if host is not None and point is not None:
            host.points.remove(point.name)
            self._changed()

    def _read_point(self) -> None:
        host, point = self._selected_point()
        if host is None or point is None:
            return
        try:
            value = host.read_point(point.name)
        except Exception as e:
            QMessageBox.warning(self, tr("读取数据点"), f"{point.name}：{e}")
            return
        self._refresh_points()
        QMessageBox.information(self, tr("读取数据点"), f"{point.name} = {value}")

    # ============================================================ 触发规则
    def _build_receive(self) -> None:
        page, lay = _page()
        bar, _ = _bar(("添加", self._add_rx), ("编辑", self._edit_rx), ("删除", self._remove_rx),
                      ("启用/禁用", self._toggle_rx))
        self.rx_table = _table(["名称", "设备", "来源", "条件", "匹配内容", "动作", "目标", "忙时"])
        self.rx_table.doubleClicked.connect(lambda _: self._edit_rx())
        lay.addLayout(bar)
        lay.addWidget(self.rx_table)
        self.tabs.addTab(page, tr("触发"))

    def _rx_dialog(self, rule: ReceiveRule) -> ReceiveRuleDialog:
        return ReceiveRuleDialog(self, rule, self._device_choices(), self._flows(),
                                 list(self.mgr.parse_rules), self._point_choices())

    def _add_rx(self) -> None:
        dlg = self._rx_dialog(ReceiveRule())
        if dlg.exec() == QDialog.Accepted:
            self.mgr.add_receive_rule(dlg.result())
            self._changed(self.rx_table)

    def _edit_rx(self) -> None:
        row = self.rx_table.currentRow()
        if row < 0:
            return
        dlg = self._rx_dialog(self.mgr.receive_rules[row])
        if dlg.exec() == QDialog.Accepted:
            rule = dlg.result()
            rule.id = self.mgr.receive_rules[row].id
            self.mgr.receive_rules[row] = rule
            self._changed()

    def _remove_rx(self) -> None:
        row = self.rx_table.currentRow()
        if row >= 0:
            del self.mgr.receive_rules[row]
            self._changed()

    def _toggle_rx(self) -> None:
        row = self.rx_table.currentRow()
        if row >= 0:
            rule = self.mgr.receive_rules[row]
            rule.enabled = not rule.enabled
            self._changed()

    # ============================================================ 发送规则
    def _build_send(self) -> None:
        page, lay = _page()
        bar, _ = _bar(("添加", self._add_tx), ("编辑", self._edit_tx), ("删除", self._remove_tx),
                      ("启用/禁用", self._toggle_tx))
        self.tx_table = _table(["名称", "设备", "流程", "条件", "发给谁", "载荷"])
        self.tx_table.doubleClicked.connect(lambda _: self._edit_tx())
        lay.addLayout(bar)
        lay.addWidget(self.tx_table)
        self.tabs.addTab(page, tr("发送"))

    def _tx_dialog(self, rule: SendRule) -> SendRuleDialog:
        return SendRuleDialog(self, rule, self._device_choices(), self._flows(),
                              list(self.mgr.format_rules), self._point_choices())

    def _add_tx(self) -> None:
        dlg = self._tx_dialog(SendRule())
        if dlg.exec() == QDialog.Accepted:
            self.mgr.add_send_rule(dlg.result())
            self._changed(self.tx_table)

    def _edit_tx(self) -> None:
        row = self.tx_table.currentRow()
        if row < 0:
            return
        dlg = self._tx_dialog(self.mgr.send_rules[row])
        if dlg.exec() == QDialog.Accepted:
            rule = dlg.result()
            rule.id = self.mgr.send_rules[row].id
            self.mgr.send_rules[row] = rule
            self._changed()

    def _remove_tx(self) -> None:
        row = self.tx_table.currentRow()
        if row >= 0:
            del self.mgr.send_rules[row]
            self._changed()

    def _toggle_tx(self) -> None:
        row = self.tx_table.currentRow()
        if row >= 0:
            rule = self.mgr.send_rules[row]
            rule.enabled = not rule.enabled
            self._changed()

    # ============================================================ 解析 / 格式化
    def _build_rules(self) -> None:
        page, lay = _page()
        split = QSplitter(Qt.Horizontal)
        parse_box = QGroupBox(tr("解析规则（报文 → 字段）"))
        pl = QVBoxLayout(parse_box)
        pbar, _ = _bar(("添加", self._add_parse), ("编辑", self._edit_parse), ("删除", self._remove_parse))
        self.parse_table = _table(["名称", "方式", "字段"])
        self.parse_table.doubleClicked.connect(lambda _: self._edit_parse())
        pl.addLayout(pbar)
        pl.addWidget(self.parse_table)
        fmt_box = QGroupBox(tr("格式化规则（结果 → 报文）"))
        fl = QVBoxLayout(fmt_box)
        fbar, _ = _bar(("添加", self._add_format), ("编辑", self._edit_format), ("删除", self._remove_format))
        self.format_table = _table(["名称", "格式", "字段"])
        self.format_table.doubleClicked.connect(lambda _: self._edit_format())
        fl.addLayout(fbar)
        fl.addWidget(self.format_table)
        split.addWidget(parse_box)
        split.addWidget(fmt_box)
        lay.addWidget(split)
        self.tabs.addTab(page, tr("解析/格式"))

    def _published_names(self) -> list[str]:
        names: list[str] = []
        for runner in self.mgr.runners.values():
            for node in runner.graph.nodes.values():
                if node.type_id == "output.publish":
                    names += [str(v) for k, v in node.values.items() if k.startswith("name_") and v]
        return sorted(set(names))

    def _add_parse(self) -> None:
        dlg = ParseRuleDialog(self, None, list(self.mgr.parse_rules))
        if dlg.exec() == QDialog.Accepted:
            self.mgr.add_parse_rule(dlg.result())
            self._changed(self.parse_table)

    def _edit_parse(self) -> None:
        row = self.parse_table.currentRow()
        if row < 0:
            return
        name = self.parse_table.item(row, 0).text()
        dlg = ParseRuleDialog(self, self.mgr.parse_rules.get(name), list(self.mgr.parse_rules))
        if dlg.exec() == QDialog.Accepted:
            rule = dlg.result()
            if rule.name != name:
                self.mgr.parse_rules.pop(name, None)
            self.mgr.add_parse_rule(rule)
            self._changed()

    def _remove_parse(self) -> None:
        row = self.parse_table.currentRow()
        if row >= 0:
            self.mgr.parse_rules.pop(self.parse_table.item(row, 0).text(), None)
            self._changed()

    def _add_format(self) -> None:
        dlg = FormatRuleDialog(self, None, list(self.mgr.format_rules), self._published_names(),
                               self.mgr.variables.names())
        if dlg.exec() == QDialog.Accepted:
            self.mgr.add_format_rule(dlg.result())
            self._changed(self.format_table)

    def _edit_format(self) -> None:
        row = self.format_table.currentRow()
        if row < 0:
            return
        name = self.format_table.item(row, 0).text()
        dlg = FormatRuleDialog(self, self.mgr.format_rules.get(name), list(self.mgr.format_rules),
                               self._published_names(), self.mgr.variables.names())
        if dlg.exec() == QDialog.Accepted:
            rule = dlg.result()
            if rule.name != name:
                self.mgr.format_rules.pop(name, None)
            self.mgr.add_format_rule(rule)
            self._changed()

    def _remove_format(self) -> None:
        row = self.format_table.currentRow()
        if row >= 0:
            self.mgr.format_rules.pop(self.format_table.item(row, 0).text(), None)
            self._changed()

    # ============================================================ 检测握手
    def _build_handshakes(self) -> None:
        page, lay = _page()
        bar, _ = _bar(("添加", self._add_hs), ("编辑", self._edit_hs), ("删除", self._remove_hs))
        self.hs_table = _table(["名称", "设备", "流程", "模式", "状态", "请求编号", "错误码",
                                "已接受", "已完成", "忙拒绝", "超时", "说明"])
        self.hs_table.doubleClicked.connect(lambda _: self._edit_hs())
        lay.addLayout(bar)
        lay.addWidget(self.hs_table)
        lay.addWidget(QLabel(tr("握手把设备与流程绑成闭环：接受请求 → 执行 → 先写结果再写完成 → 对端确认后复位。")))
        self.tabs.addTab(page, tr("握手"))

    def _hs_dialog(self, config: HandshakeConfig | None) -> HandshakeDialog:
        return HandshakeDialog(self, config, self._device_choices(), self._flows(),
                               list(self.mgr.format_rules), self._point_choices(),
                               [h.name for h in self.mgr.handshakes])

    def _add_hs(self) -> None:
        dlg = self._hs_dialog(None)
        if dlg.exec() == QDialog.Accepted:
            self.mgr.add_handshake(dlg.result())
            self._changed(self.hs_table)

    def _edit_hs(self) -> None:
        row = self.hs_table.currentRow()
        if row < 0:
            return
        old = self.mgr.handshakes[row]
        dlg = self._hs_dialog(old.config)
        if dlg.exec() == QDialog.Accepted:
            cfg = dlg.result()
            cfg.id = old.config.id or old.id
            old.shutdown()
            from ..comm.handshake import Handshake
            self.mgr.handshakes[row] = Handshake(cfg, self.mgr)
            self._changed()

    def _remove_hs(self) -> None:
        row = self.hs_table.currentRow()
        if row >= 0:
            self.mgr.handshakes.pop(row).shutdown()
            self._changed()

    # ============================================================ 调试
    def _build_debug(self) -> None:
        page, lay = _page()
        # 手动发送
        send_box = QGroupBox(tr("手动发送"))
        sl = QHBoxLayout(send_box)
        self.send_dev = QComboBox()
        self.send_mode = QComboBox()
        self.send_mode.addItem(tr("文本"), "text")
        self.send_mode.addItem(tr("十六进制"), "hex")
        self.send_peer = QComboBox()
        self.send_peer.setEditable(False)
        self.send_text = QLineEdit()
        self.send_text.setPlaceholderText(tr("要发送的内容（文本可写 \\n、\\r\\n；十六进制写 02 41 42 03）"))
        self.send_text.returnPressed.connect(self._manual_send)
        send_btn = QPushButton(tr("发送"))
        send_btn.clicked.connect(self._manual_send)
        self.send_raw = QCheckBox(tr("不自动加结束符"))
        for w in (QLabel(tr("设备")), self.send_dev, self.send_mode, QLabel(tr("对端")), self.send_peer):
            sl.addWidget(w)
        sl.addWidget(self.send_text, 1)
        sl.addWidget(self.send_raw)
        sl.addWidget(send_btn)
        lay.addWidget(send_box)

        # Modbus 手动读写与轮询监视
        mb_box = QGroupBox(tr("Modbus 手动读写 / 轮询监视"))
        ml = QHBoxLayout(mb_box)
        self.mb_dev = QComboBox()
        self.mb_dev.currentIndexChanged.connect(lambda _: self._sync_mb_points())
        self.mb_point = QComboBox()
        self.mb_value = QLineEdit()
        self.mb_value.setPlaceholderText(tr("要写入的值"))
        read_btn = QPushButton(tr("读取"))
        read_btn.clicked.connect(self._mb_read)
        write_btn = QPushButton(tr("写入"))
        write_btn.clicked.connect(self._mb_write)
        self.mb_watch = QCheckBox(tr("持续监视"))
        self.mb_result = QLabel("—")
        self.mb_result.setFont(mono_font())
        for w in (QLabel(tr("设备")), self.mb_dev, QLabel(tr("数据点")), self.mb_point,
                  self.mb_value, read_btn, write_btn, self.mb_watch):
            ml.addWidget(w)
        ml.addWidget(self.mb_result, 1)
        lay.addWidget(mb_box)

        # 日志
        tool = QHBoxLayout()
        self.log_filter_dir = QComboBox()
        self.log_filter_dir.addItem(tr("全部方向"), "")
        for key, label in LOG_DIRECTIONS.items():
            self.log_filter_dir.addItem(label, key)
        self.log_filter_dir.currentIndexChanged.connect(lambda _: self._reload_log())
        self.log_filter_dev = QComboBox()
        self.log_filter_dev.currentIndexChanged.connect(lambda _: self._reload_log())
        self.log_filter_text = QLineEdit()
        self.log_filter_text.setPlaceholderText(tr("按内容筛选"))
        self.log_filter_text.textChanged.connect(lambda _: self._reload_log())
        self.log_pause = QCheckBox(tr("暂停滚动"))
        self.log_stop = QCheckBox(tr("停止记录"))
        self.log_stop.toggled.connect(self._set_log_paused)
        clear_btn = QPushButton(tr("清空"))
        clear_btn.clicked.connect(self._clear_log)
        export_btn = QPushButton(tr("导出 CSV"))
        export_btn.clicked.connect(self._export_log)
        self.log_count = QLabel("")
        for w in (self.log_filter_dir, self.log_filter_dev, self.log_filter_text,
                  self.log_pause, self.log_stop, clear_btn, export_btn):
            tool.addWidget(w)
        tool.addWidget(self.log_count, 1)
        lay.addLayout(tool)
        self.log_table = _table(["时间", "方向", "设备", "对端", "字节", "文本", "十六进制", "说明"])
        self.log_table.setFont(mono_font())
        self.log_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Interactive)
        lay.addWidget(self.log_table, 1)
        self.tabs.addTab(page, tr("调试"))

    def _set_log_paused(self, flag: bool) -> None:
        self.mgr.log.paused = bool(flag)

    def _manual_send(self) -> None:
        dev = self.mgr.resolve(self.send_dev.currentData() or self.send_dev.currentText())
        text = self.send_text.text()
        if dev is None or not text:
            return
        peer = self.send_peer.currentData() or ""
        if self.send_mode.currentData() == "hex":
            from ..comm.framing import parse_hex
            try:
                data = parse_hex(text)
            except ValueError as e:
                QMessageBox.warning(self, tr("手动发送"), str(e))
                return
        else:
            from ..comm.framing import unescape
            data = unescape(text)
        ok = dev.send(data, peer=peer or None, raw=self.send_raw.isChecked() or
                      self.send_mode.currentData() == "hex")
        if not ok:
            QMessageBox.warning(self, tr("手动发送"), f"{dev.name}：{dev.last_error or '发送失败'}")
        self._tick()

    def _sync_mb_points(self) -> None:
        dev = self.mgr.resolve(self.mb_dev.currentData() or "")
        current = self.mb_point.currentText()
        self.mb_point.clear()
        if isinstance(dev, _PointHost):
            self.mb_point.addItems(dev.points.names())
        self.mb_point.setCurrentText(current)

    def _mb_read(self) -> None:
        dev = self.mgr.resolve(self.mb_dev.currentData() or "")
        name = self.mb_point.currentText()
        if not isinstance(dev, _PointHost) or not name:
            return
        try:
            value = dev.read_point(name)
        except Exception as e:
            self.mb_result.setText(f"读取失败：{e}")
            self.mb_result.setStyleSheet(f"color:{C['ng']}")
            return
        self.mb_result.setText(f"{name} = {value}   （{_fmt_time(time.time())}）")
        self.mb_result.setStyleSheet(f"color:{C['text']}")

    def _mb_write(self) -> None:
        dev = self.mgr.resolve(self.mb_dev.currentData() or "")
        name = self.mb_point.currentText()
        if not isinstance(dev, _PointHost) or not name:
            return
        raw = self.mb_value.text().strip()
        point = dev.points.get(name)
        value: object = raw
        if point is not None and point.dtype != "string":
            try:
                value = float(raw) if "." in raw else int(raw, 0)
            except ValueError:
                if point.dtype == "bool":
                    value = raw.lower() in ("1", "true", "on", "ok")
                else:
                    self.mb_result.setText(f"{raw!r} 不是数字")
                    self.mb_result.setStyleSheet(f"color:{C['ng']}")
                    return
        try:
            dev.write_point(name, value)
        except Exception as e:
            self.mb_result.setText(f"写入失败：{e}")
            self.mb_result.setStyleSheet(f"color:{C['ng']}")
            return
        self.mb_result.setText(f"已写入 {name} = {value}")
        self.mb_result.setStyleSheet(f"color:{C['ok']}")

    def _clear_log(self) -> None:
        self.mgr.log.clear()
        self._log_seq = 0
        self.log_table.setRowCount(0)
        self.log_count.setText("")

    def _export_log(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr("导出通信日志"), "comm-log.csv",
                                              tr("CSV 文件 (*.csv)"))
        if not path:
            return
        try:
            n = self.mgr.log.export(path)
        except OSError as e:
            QMessageBox.warning(self, tr("导出通信日志"), str(e))
            return
        QMessageBox.information(self, tr("导出通信日志"), f"已导出 {n} 条记录到\n{path}")

    def _log_matches(self, entry: dict) -> bool:
        want_dir = self.log_filter_dir.currentData() or ""
        if want_dir and entry["dir"] != want_dir:
            return False
        want_dev = self.log_filter_dev.currentData() or ""
        if want_dev and entry["device"] != want_dev:
            return False
        needle = self.log_filter_text.text().strip()
        if needle:
            blob = f"{entry['text']} {entry['hex']} {entry['note']} {entry['peer']}"
            return needle.lower() in blob.lower()
        return True

    def _reload_log(self) -> None:
        self.log_table.setRowCount(0)
        self._log_seq = 0
        self._append_log(self.mgr.log.entries(limit=self.mgr.log.capacity))

    def _append_log(self, entries: list[dict]) -> None:
        for e in entries:
            self._log_seq = max(self._log_seq, e["seq"])
            if not self._log_matches(e):
                continue
            row = self.log_table.rowCount()
            if row >= self.mgr.log.capacity:
                self.log_table.removeRow(0)
                row -= 1
            self.log_table.insertRow(row)
            cells = [_fmt_time(e["time"]), LOG_DIRECTIONS.get(e["dir"], e["dir"]), e["device"],
                     e["peer"], str(e["size"] or ""), e["text"], e["hex"], e["note"]]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 1:
                    item.setForeground(QColor({"rx": C["accent"], "tx": C["ok"],
                                               "err": C["ng"]}.get(e["dir"], C["muted"])))
                self.log_table.setItem(row, col, item)
        if entries and not self.log_pause.isChecked():
            self.log_table.scrollToBottom()

    # ============================================================ 刷新
    def set_manager(self, mgr: CommManager) -> None:
        self.mgr = mgr
        self._log_seq = 0
        self.log_table.setRowCount(0)
        self.refresh()

    def _changed(self, select_last: QTableWidget | None = None) -> None:
        self.refresh()
        if select_last is not None and select_last.rowCount():
            select_last.setCurrentCell(select_last.rowCount() - 1, 0)
        self.config_changed.emit()

    def _tick(self) -> None:
        """定时批量刷新。面板不可见时只拉日志，不重画表格。"""
        new = self.mgr.log.entries(since_seq=self._log_seq)
        if new:
            self._append_log(new)
            self.log_count.setText(
                f"{self.log_table.rowCount()} 条显示 / 共 {len(self.mgr.log.entries())} 条"
                + (f"，已丢弃 {self.mgr.log.dropped} 条" if self.mgr.log.dropped else ""))
        if not self.isVisible():
            return
        current = self.tabs.currentIndex()
        if current == 0:
            self._refresh_devices()
        elif current == 1:
            self._refresh_points()
        elif current == 5:
            self._refresh_handshakes()
        if self.mb_watch.isChecked() and current == 6:
            self._mb_read()

    def refresh(self) -> None:
        self._refresh_devices()
        self._refresh_combos()
        self._refresh_points()
        self._refresh_rules()
        self._refresh_handshakes()

    def _refresh_devices(self) -> None:
        rows = self.mgr.device_rows()
        self.dev_table.setRowCount(len(rows))
        for r, info in enumerate(rows):
            if info["connected"]:
                status, color = "● 已连接", C["ok"]
            elif not info["enabled"]:
                status, color = "⊘ 已禁用", C["muted"]
            elif info["manual_stop"]:
                status, color = "○ 已手动断开", C["muted"]
            elif info["error"]:
                status, color = "✖ 未连接", C["ng"]
            else:
                status, color = "○ 未连接", C["muted"]
            cells = [info["name"], info["kind_label"] + info["status_note"], status, info["peer"],
                     str(info["rx"]), str(info["tx"]), _ago(info["last_comm"]), info["error"]]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.UserRole, info["id"])
                    item.setToolTip(f"标识 {info['id']}（改名不影响引用）")
                if c == 2:
                    item.setForeground(QColor(color))
                if c == 7 and text:
                    item.setForeground(QColor(C["ng"]))
                    item.setToolTip(text)
                self.dev_table.setItem(r, c, item)

    def _refresh_combos(self) -> None:
        for box, keep in ((self.point_device, True), (self.send_dev, True), (self.mb_dev, True)):
            current = box.currentData()
            box.blockSignals(True)
            box.clear()
            for dev in self.mgr.devices.values():
                if box is self.mb_dev and not isinstance(dev, _PointHost):
                    continue
                if box is self.point_device and not isinstance(dev, _PointHost):
                    continue
                box.addItem(dev.name, dev.id)
            idx = box.findData(current)
            if idx >= 0:
                box.setCurrentIndex(idx)
            box.blockSignals(False)
        dev = self.mgr.resolve(self.send_dev.currentData() or "")
        current_peer = self.send_peer.currentData()
        self.send_peer.blockSignals(True)
        self.send_peer.clear()
        self.send_peer.addItem(tr("（默认）"), "")
        if dev is not None:
            for pid in dev.peers:
                self.send_peer.addItem(pid, pid)
        idx = self.send_peer.findData(current_peer)
        if idx >= 0:
            self.send_peer.setCurrentIndex(idx)
        self.send_peer.blockSignals(False)
        current_dev = self.log_filter_dev.currentData()
        self.log_filter_dev.blockSignals(True)
        self.log_filter_dev.clear()
        self.log_filter_dev.addItem(tr("全部设备"), "")
        for d in self.mgr.devices.values():
            self.log_filter_dev.addItem(d.name, d.name)
        idx = self.log_filter_dev.findData(current_dev)
        if idx >= 0:
            self.log_filter_dev.setCurrentIndex(idx)
        self.log_filter_dev.blockSignals(False)
        self._sync_mb_points()

    def _refresh_points(self) -> None:
        host = self._point_host()
        points = host.points.points if host is not None else []
        self.point_table.setRowCount(len(points))
        for r, p in enumerate(points):
            pv = host.point_value(p.name) if host is not None else None
            if pv is None or not pv.valid:
                value = "—"
            elif pv.stale:
                value = f"{pv.value}（已过期）"
            else:
                value = str(pv.value)
            cells = [p.name + ("" if p.enabled else tr("（已禁用）")),
                     AREA_INFO[p.area]["label"], str(p.address), str(p.reference),
                     p.dtype + (f"×{p.count}" if p.count > 1 else "")
                     + (f"[{p.length}]" if p.dtype == "string" else ""),
                     DIRECTION_LABELS[p.direction], str(p.poll_ms or ""), p.layout,
                     p.variable, value]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 7:
                    item.setToolTip(p.layout_hint())
                if c == 9 and pv is not None and (pv.stale or not pv.valid):
                    item.setForeground(QColor(C["warn"] if pv.stale else C["muted"]))
                    if pv.error:
                        item.setToolTip(pv.error)
                self.point_table.setItem(r, c, item)

    def _refresh_rules(self) -> None:
        self.rx_table.setRowCount(len(self.mgr.receive_rules))
        for r, rule in enumerate(self.mgr.receive_rules):
            if rule.source == "message":
                match = TEXT_MATCH_LABELS.get(rule.match, rule.match)
                pattern = rule.pattern if rule.match != "any" else "—"
            else:
                match = DATA_MATCH_LABELS.get(rule.match, rule.match)
                pattern = rule.point or f"地址 {rule.address}"
                if rule.match in ("equals", "not_equals", "greater", "less"):
                    pattern += f" {rule.value:g}"
                elif rule.match == "in_range":
                    pattern += f" [{rule.value:g}, {rule.value2:g}]"
            target = rule.flow if rule.action == "trigger_flow" else f"变量 {rule.variable}"
            cells = [rule.name + ("" if rule.enabled else tr("（已停用）")),
                     self.mgr.device_label(rule.device) if rule.device else "全部",
                     RULE_SOURCE_LABELS.get(rule.source, rule.source), match, pattern,
                     ACTION_LABELS.get(rule.action, rule.action), target,
                     rule.busy_policy if rule.action == "trigger_flow" else ""]
            self._fill_row(self.rx_table, r, cells, rule.enabled,
                           dangling=bool(rule.device) and self.mgr.resolve(rule.device) is None)
        self.tx_table.setRowCount(len(self.mgr.send_rules))
        for r, rule in enumerate(self.mgr.send_rules):
            payload = []
            if rule.points:
                payload.append(f"{len(rule.points)} 个数据点")
            if rule.registers:
                payload.append(f"{len(rule.registers)} 个寄存器")
            if rule.format:
                payload.append(f"格式化 {rule.format}")
            elif rule.template:
                payload.append(rule.template)
            cells = [rule.name + ("" if rule.enabled else tr("（已停用）")),
                     self.mgr.device_label(rule.device), rule.flow or "全部",
                     WHEN_LABELS.get(rule.when, rule.when),
                     SEND_TARGET_LABELS.get(rule.target, rule.target), " + ".join(payload)]
            self._fill_row(self.tx_table, r, cells, rule.enabled,
                           dangling=self.mgr.resolve(rule.device) is None)
        self.parse_table.setRowCount(len(self.mgr.parse_rules))
        for r, rule in enumerate(self.mgr.parse_rules.values()):
            from ..comm.parsing import PARSE_LABELS
            self._fill_row(self.parse_table, r, [rule.name, PARSE_LABELS.get(rule.kind, rule.kind),
                                                 "、".join(f.name for f in rule.fields)], True)
        self.format_table.setRowCount(len(self.mgr.format_rules))
        for r, rule in enumerate(self.mgr.format_rules.values()):
            from ..comm.parsing import FORMAT_LABELS
            detail = rule.template or "、".join(f.name for f in rule.fields)
            self._fill_row(self.format_table, r,
                           [rule.name, FORMAT_LABELS.get(rule.kind, rule.kind), detail], True)

    def _fill_row(self, table: QTableWidget, row: int, cells: list[str], enabled: bool,
                  dangling: bool = False) -> None:
        for c, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if not enabled:
                item.setForeground(QColor(C["muted"]))
            if dangling:
                item.setForeground(QColor(C["ng"]))
                item.setToolTip(tr("引用的设备不存在了"))
            table.setItem(row, c, item)

    def _refresh_handshakes(self) -> None:
        self.hs_table.setRowCount(len(self.mgr.handshakes))
        for r, hs in enumerate(self.mgr.handshakes):
            st = hs.status_dict()
            cells = [st["name"] + ("" if st["enabled"] else tr("（已停用）")), st["device"],
                     st["flow"], "寄存器" if st["mode"] == "register" else "文本",
                     STATE_LABELS.get(st["state"], st["state"]), str(st["request_id"]),
                     f"{st['error_code']} {st['error_text']}".strip(),
                     str(st["accepted"]), str(st["completed"]), str(st["rejected_busy"]),
                     str(st["timeout"]), st["status"]]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 4:
                    item.setForeground(QColor({"ready": C["ok"], "busy": C["warn"],
                                               "offline": C["muted"]}.get(st["state"], C["text"])))
                if c == 6 and st["error_code"]:
                    item.setForeground(QColor(C["ng"]))
                self.hs_table.setItem(r, c, item)

    # ---- 事件（由主窗口转到界面线程）----
    def on_event(self, event: str, payload: dict) -> None:
        """通信事件只触发一次轻量更新；具体内容由定时器批量拉取，避免高频通信刷爆界面。"""
        if event in ("comm_connected", "comm_disconnected"):
            self._refresh_devices()
            self._refresh_combos()

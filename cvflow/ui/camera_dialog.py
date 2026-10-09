"""相机管理对话框：像 VisionMaster 一样搜索网络上的 GigE/USB 相机，支持按 IP 添加、连接测试、强制 IP、加入流程。"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from ..camera import discovery, enumerate_cameras, test_camera
from ..camera.discovery import GigEDevice

_COLS = ["来源", "名称", "型号", "序列号", "IP 地址", "MAC", "厂商", "网段"]


class ForceIpDialog(QDialog):
    def __init__(self, parent, dev: GigEDevice) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"强制 IP – {dev.display_name}")
        form = QFormLayout(self)
        iface = dev.interface_ip or (discovery.local_interfaces() or ["192.168.1.1"])[0]
        base = ".".join(iface.split(".")[:3])
        self.ip = QLineEdit(f"{base}.{int(dev.ip.split('.')[-1]) if dev.ip else 100}")
        self.subnet = QLineEdit("255.255.255.0")
        self.gateway = QLineEdit(f"{base}.1")
        form.addRow("相机 MAC", QLabel(dev.mac))
        form.addRow("本机网卡", QLabel(iface))
        form.addRow("新 IP", self.ip)
        form.addRow("子网掩码", self.subnet)
        form.addRow("网关", self.gateway)
        hint = QLabel("强制 IP 是临时的，相机断电后恢复原设置；要永久修改请在厂商软件里设置持久 IP。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888")
        form.addRow(hint)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)


class CameraDialog(QDialog):
    add_requested = Signal(str, str, str, str)   # kind, source, camera name, cti

    def __init__(self, parent=None, enumerator: Callable | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("相机管理")
        self.resize(980, 520)
        self._enumerate = enumerator or enumerate_cameras
        self._devices: list[GigEDevice] = []
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.cti = QLineEdit(str(QSettings().value("cti", "") or ""))
        self.cti.setPlaceholderText("GenTL 驱动 (.cti)，仅 GenICam 方式取图需要")
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_cti)
        top.addWidget(QLabel("GenTL："))
        top.addWidget(self.cti, 1)
        top.addWidget(browse)
        lay.addLayout(top)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(lambda _: self._add_to_flow())
        lay.addWidget(self.table, 1)

        bar = QHBoxLayout()
        for text, slot in [("搜索相机", self._search), ("按 IP 添加…", self._add_by_ip), ("连接测试", self._test),
                           ("强制 IP…", self._force_ip), ("添加到流程", self._add_to_flow)]:
            b = QPushButton(text)
            if text == "添加到流程":
                b.setObjectName("primary")
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        bar.addWidget(close)
        lay.addLayout(bar)

        self.status = QLabel(self._sdk_status())
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#aaa")
        lay.addWidget(self.status)

    # ---- helpers ----
    @staticmethod
    def _sdk_status() -> str:
        try:
            from ..camera.hik_cam import sdk_available
            hik = "可用" if sdk_available() else "未找到（安装 MVS 后可用海康相机取图）"
        except Exception:
            hik = "未找到"
        try:
            import harvesters  # noqa: F401
            gen = "可用"
        except ImportError:
            gen = "未安装（pip install harvesters）"
        return f"海康 MVS SDK：{hik}　|　GenICam/harvesters：{gen}　|　搜索使用 GigE Vision 发现协议，不依赖 SDK"

    def _browse_cti(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 GenTL 驱动", self.cti.text(), "GenTL producer (*.cti)")
        if path:
            self.cti.setText(path)
            QSettings().setValue("cti", path)

    def _selected(self) -> GigEDevice | None:
        row = self.table.currentRow()
        return self._devices[row] if 0 <= row < len(self._devices) else None

    def _fill(self, devices: list[GigEDevice]) -> None:
        self._devices = list(devices)
        self.table.setRowCount(len(devices))
        for r, d in enumerate(devices):
            src = "+".join(d.extra.get("sources", [d.source]))
            net = "可达" if d.reachable else "不在本机网段（需强制 IP）"
            for c, text in enumerate([src, d.display_name, d.model, d.serial, d.ip, d.mac, d.manufacturer, net]):
                it = QTableWidgetItem(text)
                if c == 7 and not d.reachable:
                    it.setForeground(Qt.red)
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        for c, w in enumerate((70, 110, 150, 120, 120, 140, 100)):
            self.table.setColumnWidth(c, max(self.table.columnWidth(c), w))
        if devices:
            self.table.selectRow(0)

    # ---- actions ----
    def _search(self) -> None:
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            devs = self._enumerate(timeout=1.0, cti=self.cti.text().strip() or None)
        except Exception as e:
            devs = []
            QMessageBox.warning(self, "搜索相机", str(e))
        finally:
            QApplication.restoreOverrideCursor()
        self._fill(devs)
        self.status.setText(f"找到 {len(devs)} 台相机　|　{self._sdk_status()}")

    def _add_by_ip(self) -> None:
        ip, ok = QInputDialog.getText(self, "按 IP 添加相机", "相机 IP 地址：")
        if not ok or not ip.strip():
            return
        ip = ip.strip()
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            devs = self._enumerate(timeout=1.0, targets=[ip], cti=None)
        except Exception:
            devs = []
        finally:
            QApplication.restoreOverrideCursor()
        found = [d for d in devs if d.ip == ip]
        if not found:
            QMessageBox.warning(self, "按 IP 添加相机", f"{ip} 没有响应 GigE Vision 发现请求。\n可以继续手动添加，取图时再用 SDK 按 IP 连接。")
            found = [GigEDevice(ip=ip, mac="", source="manual", user_name=ip, reachable=True)]
        self._fill(self._devices + found)
        self.table.selectRow(len(self._devices) - 1)

    def _test(self) -> None:
        dev = self._selected()
        if dev is None:
            return
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            if dev.ip:
                ok, msg = discovery.test_connection(dev.ip)
            else:
                ok, msg = test_camera("hik", dev.serial)
        finally:
            QApplication.restoreOverrideCursor()
        (QMessageBox.information if ok else QMessageBox.warning)(self, "连接测试", msg)

    def _force_ip(self) -> None:
        dev = self._selected()
        if dev is None or not dev.mac:
            QMessageBox.information(self, "强制 IP", "请选择一台通过搜索找到的 GigE 相机（需要 MAC 地址）。")
            return
        dlg = ForceIpDialog(self, dev)
        if dlg.exec() != QDialog.Accepted:
            return
        ok = discovery.force_ip(dev.mac, dlg.ip.text().strip(), dlg.subnet.text().strip(), dlg.gateway.text().strip())
        QMessageBox.information(self, "强制 IP", "已发送强制 IP 命令，正在重新搜索。" if ok else "相机没有确认强制 IP 命令，请检查网线和网卡。")
        self._search()

    def _add_to_flow(self) -> None:
        dev = self._selected()
        if dev is None:
            return
        cti = self.cti.text().strip()
        index = str(dev.extra.get("index", 0))
        # 按这台设备实际是被哪条路枚举出来的决定取图方式：厂商 SDK 优先，其次通用 GenTL。
        # 不能只看某个 SDK 装没装——装了海康 SDK 不代表手上这台就是海康相机。
        sources = dev.extra.get("sources") or [dev.source]
        if "mindvision" in sources:
            kind, source = "mindvision", dev.serial or index
        elif "hik" in sources:
            kind, source = "hik", dev.serial or dev.ip
        elif "genicam" in sources or cti:
            kind, source = "genicam", dev.serial or index
        else:
            kind, source = "hik", dev.serial or dev.ip
            QMessageBox.information(self, "添加到流程",
                                    "这台相机只被 GigE 广播发现，没有任何 SDK 枚举到它。已按海康方式添加，"
                                    "取图前请安装 MVS SDK，或填写 GenTL 驱动改用 GenICam 方式。")
        name = dev.user_name or dev.serial or dev.ip or "cam"
        self.add_requested.emit(kind, source, name, cti if kind == "genicam" else "")
        self.accept()

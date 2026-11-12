"""帮助 → 提交 Bug：写几句话，一键发到开发者邮箱。"""
from __future__ import annotations

import smtplib
import threading
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QPushButton, QVBoxLayout)

from .. import bug_report
from ..bug_report import BugReport, SmtpConfig
from .theme import C, M, mono_font, ui_font


def describe_send_error(e: BaseException, cfg: SmtpConfig) -> str:
    if isinstance(e, smtplib.SMTPAuthenticationError):
        return "发信邮箱登录失败（授权码不对或已失效）"
    if isinstance(e, smtplib.SMTPException):
        return f"邮件服务器拒绝了：{e}"
    if isinstance(e, OSError):
        return f"连不上邮件服务器 {cfg.host}:{cfg.port}（这台电脑可能没有外网）：{e}"
    return f"{type(e).__name__}: {e}"


class BugReportDialog(QDialog):
    #: 后台线程发完之后回到界面线程：空字符串表示成功，否则是错误说明
    _finished = Signal(str)

    def __init__(self, parent=None,
                 load_config: Callable[[], SmtpConfig | None] = bug_report.load_config,
                 sender: Callable[[BugReport, SmtpConfig], None] = bug_report.send) -> None:
        super().__init__(parent)
        self.setWindowTitle("提交 Bug")
        self.resize(620, 520)
        self._sender = sender
        self._thread: threading.Thread | None = None
        self._report: BugReport | None = None
        try:
            self._cfg, cfg_error = load_config(), ""
        except ValueError as e:
            self._cfg, cfg_error = None, str(e)

        lay = QVBoxLayout(self)
        lay.setSpacing(M["gap"])
        to = self._cfg.to if self._cfg else bug_report.DEFAULT_RECIPIENT
        intro = QLabel(f"报告会发到 {to}。")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        lay.addWidget(QLabel("问题描述"))
        self.description = QPlainTextEdit()
        self.description.setPlaceholderText("出了什么问题？做了哪些操作？能不能稳定复现？")
        lay.addWidget(self.description, 2)

        row = QHBoxLayout()
        row.addWidget(QLabel("联系方式"))
        self.contact = QLineEdit()
        self.contact.setPlaceholderText("邮箱或电话，选填，方便回复你")
        row.addWidget(self.contact, 1)
        lay.addLayout(row)

        self.cb_system = QCheckBox("附带软件和系统信息")
        self.cb_system.setChecked(True)
        self.cb_log = QCheckBox("附带最近的日志")
        self.cb_log.setChecked(True)
        self.btn_preview = QPushButton("查看要发送的内容")
        self.btn_preview.setCheckable(True)
        self.btn_preview.toggled.connect(self._toggle_preview)
        opts = QHBoxLayout()
        opts.addWidget(self.cb_system)
        opts.addWidget(self.cb_log)
        opts.addStretch(1)
        opts.addWidget(self.btn_preview)
        lay.addLayout(opts)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(mono_font(M["font_s"]))
        self.preview.hide()
        lay.addWidget(self.preview, 3)
        for w in (self.cb_system, self.cb_log):
            w.toggled.connect(self._refresh_preview)
        self.description.textChanged.connect(self._refresh_preview)
        self.contact.textChanged.connect(self._refresh_preview)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setFont(ui_font(M["font_s"]))
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)

        self.btn_save = QPushButton("另存为文件…")
        self.btn_save.setToolTip("发不出去时，存成 zip 再想办法转交")
        self.btn_save.clicked.connect(self._save)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_send = QPushButton("发送")
        self.btn_send.setObjectName("primary")
        self.btn_send.setDefault(True)
        self.btn_send.clicked.connect(self._send)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_save)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_send)
        lay.addLayout(buttons)

        self._finished.connect(self._on_finished)
        if self._cfg is None:
            self.btn_send.setEnabled(False)
            self._set_status(cfg_error or "这个版本没有配置发信邮箱，只能另存为文件。", "warn")

    # ---- 内容 ----
    def build_report(self) -> BugReport:
        return BugReport.collect(self.description.toPlainText(), self.contact.text(),
                                 include_system=self.cb_system.isChecked(), include_log=self.cb_log.isChecked())

    def _toggle_preview(self, on: bool) -> None:
        self.preview.setVisible(on)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self.preview.isHidden():
            return
        r = self.build_report()
        text = f"标题：{r.subject}\n\n{r.body()}"
        if r.include_log:
            text += "\n\n—— 附件 log.txt ——\n" + (r.log_text or "（没有日志）")
        self.preview.setPlainText(text)

    def _set_status(self, text: str, color: str = "muted") -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color:{C[color]}")

    # ---- 发送 ----
    def _send(self) -> None:
        if self._cfg is None or self.sending:
            return
        if not self.description.toPlainText().strip():
            self._set_status("先写几句问题描述。", "warn")
            self.description.setFocus()
            return
        self._report = self.build_report()
        for w in (self.btn_send, self.btn_save, self.description, self.contact, self.cb_system, self.cb_log):
            w.setEnabled(False)
        self._set_status("正在发送…")
        report, cfg = self._report, self._cfg

        def work() -> None:
            try:
                self._sender(report, cfg)
                self._finished.emit("")
            except Exception as e:  # noqa: BLE001 - 任何失败都要回到界面上说清楚
                self._finished.emit(describe_send_error(e, cfg))

        self._thread = threading.Thread(target=work, name="bug-report", daemon=True)
        self._thread.start()

    @property
    def sending(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _on_finished(self, error: str) -> None:
        for w in (self.btn_save, self.description, self.contact, self.cb_system, self.cb_log):
            w.setEnabled(True)
        self.btn_send.setEnabled(self._cfg is not None)
        if error:
            self._set_status(f"发送失败：{error}。可以点「另存为文件」存下来，再想办法发出去。", "ng")
            return
        self._set_status("已发送，谢谢！", "ok")
        QMessageBox.information(self, "提交 Bug", "已发送，谢谢！")
        self.accept()

    # ---- 另存 ----
    def _save(self) -> None:
        default = str(Path.home() / f"CVFlow-bug-{time.strftime('%Y%m%d-%H%M%S')}.zip")
        path, _ = QFileDialog.getSaveFileName(self, "另存为", default, "Zip (*.zip)")
        if not path:
            return
        try:
            self.build_report().save_zip(path)
        except OSError as e:
            self._set_status(f"保存失败：{e}", "ng")
            return
        self._set_status(f"已保存到 {path}", "ok")

    def reject(self) -> None:
        if self.sending:                       # 发到一半别关：关了也收不到结果
            return
        super().reject()

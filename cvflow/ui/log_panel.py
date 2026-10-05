"""Log panel fed by the standard logging module (thread-safe through a Qt signal)."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from .i18n import tr
from .theme import C, M, mono_font

_COLORS = {"DEBUG": C["faint"], "INFO": C["text"], "WARNING": C["warn"], "ERROR": C["ng"], "CRITICAL": "#8F1D2B"}


class _Emitter(QObject):
    record = Signal(str, str)


class _Handler(logging.Handler):
    def __init__(self, emitter: _Emitter) -> None:
        super().__init__()
        self._e = emitter
        self.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._e.record.emit(record.levelname, self.format(record))
        except RuntimeError:
            pass


class LogPanel(QWidget):
    MAX_LINES = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(M["gap"], M["gap"], M["gap"], M["gap"])
        lay.setSpacing(M["gap"])
        bar = QHBoxLayout()
        bar.setSpacing(M["gap"])
        lvl = QLabel(tr("Level"))
        lvl.setObjectName("muted")
        self.level = QComboBox()
        self.level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.level.setCurrentText("INFO")
        self.level.setFixedHeight(M["ctl_h"])
        self.level.setMinimumWidth(110)
        self.level.currentTextChanged.connect(self._set_level)
        clear = QPushButton(tr("Clear"))
        clear.setFixedHeight(M["ctl_h"])
        bar.addWidget(lvl)
        bar.addWidget(self.level)
        bar.addStretch(1)
        bar.addWidget(clear)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(mono_font())
        self.text.setMaximumBlockCount(self.MAX_LINES)
        clear.clicked.connect(self.text.clear)
        lay.addLayout(bar)
        lay.addWidget(self.text)
        self._emitter = _Emitter()
        self._emitter.record.connect(self.append)
        self._handler = _Handler(self._emitter)
        self._handler.setLevel(logging.INFO)
        logging.getLogger("cvflow").addHandler(self._handler)
        logging.getLogger("cvflow").setLevel(logging.DEBUG)

    def _set_level(self, name: str) -> None:
        self._handler.setLevel(getattr(logging, name))

    def append(self, level: str, msg: str) -> None:
        color = _COLORS.get(level, C["text"])
        safe = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.text.appendHtml(f"<span style='color:{color}'>{safe}</span>")

    def detach(self) -> None:
        logging.getLogger("cvflow").removeHandler(self._handler)

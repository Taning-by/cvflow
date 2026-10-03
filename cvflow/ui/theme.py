"""Dark theme and shared colours."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from ..core.node import NodeStatus
from ..core.types import DataType

STATUS_COLORS = {
    NodeStatus.IDLE: "#777777",
    NodeStatus.RUNNING: "#42a5f5",
    NodeStatus.OK: "#43a047",
    NodeStatus.NG: "#e53935",
    NodeStatus.ERROR: "#fb8c00",
    NodeStatus.SKIPPED: "#555555",
}

DTYPE_COLORS = {
    DataType.IMAGE: "#4fc3f7",
    DataType.INT: "#aed581", DataType.FLOAT: "#aed581", DataType.BOOL: "#c5e1a5",
    DataType.STRING: "#ffb74d",
    DataType.RECT: "#ba68c8", DataType.POINT: "#ba68c8", DataType.CIRCLE: "#ba68c8", DataType.LINE: "#ba68c8",
    DataType.LIST: "#90a4ae", DataType.DICT: "#90a4ae", DataType.CONTOURS: "#90a4ae", DataType.TENSOR: "#90a4ae",
    DataType.ANY: "#e0e0e0",
}

STYLESHEET = """
QMainWindow, QDialog { background: #2b2b2b; }
QWidget { color: #dddddd; font-size: 12px; }
QDockWidget::title { background: #353535; padding: 4px; }
QToolBar { background: #333333; border: none; spacing: 4px; }
QMenuBar, QMenu { background: #333333; }
QMenu::item:selected { background: #505050; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QTreeWidget, QTableWidget, QListWidget {
    background: #1e1e1e; border: 1px solid #444; selection-background-color: #3d6185; }
QPushButton { background: #444; border: 1px solid #555; padding: 3px 10px; border-radius: 3px; }
QPushButton:hover { background: #505050; }
QPushButton:checked { background: #2e7d32; }
QPushButton:disabled { color: #777; }
QHeaderView::section { background: #353535; border: none; padding: 3px; }
QTabBar::tab { background: #353535; padding: 5px 10px; }
QTabBar::tab:selected { background: #454545; }
QStatusBar { background: #333333; }
QScrollArea { border: none; }
QGroupBox { border: 1px solid #444; margin-top: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 6px; }
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#2b2b2b"))
    pal.setColor(QPalette.WindowText, QColor("#dddddd"))
    pal.setColor(QPalette.Base, QColor("#1e1e1e"))
    pal.setColor(QPalette.AlternateBase, QColor("#262626"))
    pal.setColor(QPalette.Text, QColor("#dddddd"))
    pal.setColor(QPalette.Button, QColor("#444444"))
    pal.setColor(QPalette.ButtonText, QColor("#dddddd"))
    pal.setColor(QPalette.Highlight, QColor("#3d6185"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#333333"))
    pal.setColor(QPalette.ToolTipText, QColor("#dddddd"))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)

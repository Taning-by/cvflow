"""视觉规范：配色、字体、全局样式表，以及程序内绘制的矢量图标。"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication

from ..core.node import NodeStatus
from ..core.types import DataType

# ---- 配色 ----
C = {
    "bg": "#1b1e24",          # 窗口底色
    "panel": "#232730",       # 面板
    "panel2": "#2a2f3a",      # 面板内卡片 / 表头
    "canvas": "#171a1f",      # 画布
    "border": "#343a46",
    "border2": "#414857",
    "text": "#e4e7ec",
    "muted": "#8d95a3",
    "accent": "#4f8cff",
    "accent2": "#3b6fd6",
    "ok": "#3ccf6e",
    "ng": "#ff5c5c",
    "warn": "#ffb020",
    "sel": "#ffd54f",
}

STATUS_COLORS = {
    NodeStatus.IDLE: "#6b7280",
    NodeStatus.RUNNING: "#4f8cff",
    NodeStatus.OK: C["ok"],
    NodeStatus.NG: C["ng"],
    NodeStatus.ERROR: C["warn"],
    NodeStatus.SKIPPED: "#4b5160",
}

DTYPE_COLORS = {
    DataType.IMAGE: "#56c3f5",
    DataType.INT: "#a6d86b", DataType.FLOAT: "#a6d86b", DataType.BOOL: "#c7e28d",
    DataType.STRING: "#ffb86b",
    DataType.RECT: "#c792ea", DataType.POINT: "#c792ea", DataType.CIRCLE: "#c792ea", DataType.LINE: "#c792ea",
    DataType.LIST: "#9aa3ad", DataType.DICT: "#9aa3ad", DataType.CONTOURS: "#9aa3ad", DataType.TENSOR: "#9aa3ad",
    DataType.ANY: "#d7dbe2",
}

STYLESHEET = f"""
* {{ outline: none; }}
QMainWindow, QDialog, QWidget {{ background: {C['bg']}; color: {C['text']}; }}
QMainWindow::separator {{ background: {C['bg']}; width: 5px; height: 5px; }}
QMainWindow::separator:hover {{ background: {C['accent2']}; }}
QToolTip {{ background: {C['panel2']}; color: {C['text']}; border: 1px solid {C['border2']}; padding: 5px 8px; border-radius: 4px; }}

QMenuBar {{ background: {C['bg']}; padding: 2px 6px; border-bottom: 1px solid {C['border']}; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {C['panel2']}; }}
QMenu {{ background: {C['panel']}; border: 1px solid {C['border2']}; padding: 6px; }}
QMenu::item {{ padding: 6px 28px 6px 14px; border-radius: 4px; }}
QMenu::item:selected {{ background: {C['accent2']}; color: white; }}
QMenu::separator {{ height: 1px; background: {C['border']}; margin: 5px 8px; }}

QToolBar {{ background: {C['panel']}; border: none; border-bottom: 1px solid {C['border']}; padding: 4px 8px; spacing: 4px; }}
QToolBar::separator {{ width: 1px; background: {C['border2']}; margin: 6px 6px; }}
QToolButton {{ background: transparent; color: {C['text']}; border: 1px solid transparent; border-radius: 6px; padding: 5px 10px; }}
QToolButton:hover {{ background: {C['panel2']}; border-color: {C['border2']}; }}
QToolButton:pressed {{ background: {C['border']}; }}
QToolButton:checked {{ background: #3a2a2a; border-color: {C['ng']}; color: #ffb3b3; }}
QToolButton#primary {{ background: {C['accent2']}; color: white; font-weight: 600; }}
QToolButton#primary:hover {{ background: {C['accent']}; }}

QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
QDockWidget::title {{ background: {C['panel']}; padding: 7px 12px; border-bottom: 1px solid {C['border']}; font-weight: 600; color: {C['muted']}; text-align: left; }}
QDockWidget > QWidget {{ background: {C['panel']}; }}

QTabWidget::pane {{ border: none; background: {C['panel']}; }}
QTabBar {{ background: {C['panel']}; }}
QTabBar::tab {{ background: transparent; color: {C['muted']}; padding: 8px 16px; border-bottom: 2px solid transparent; margin-right: 2px; }}
QTabBar::tab:selected {{ color: {C['text']}; border-bottom: 2px solid {C['accent']}; }}
QTabBar::tab:hover {{ color: {C['text']}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {C['bg']}; color: {C['text']}; border: 1px solid {C['border2']}; border-radius: 5px; padding: 4px 7px;
    selection-background-color: {C['accent2']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {C['accent']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {C['muted']}; background: {C['panel']}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 14px; background: transparent; border: none; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid {C['muted']}; margin-right: 6px; }}
QComboBox QAbstractItemView {{ background: {C['panel']}; border: 1px solid {C['border2']}; selection-background-color: {C['accent2']}; padding: 4px; }}

QPushButton {{ background: {C['panel2']}; color: {C['text']}; border: 1px solid {C['border2']}; border-radius: 6px; padding: 5px 14px; }}
QPushButton:hover {{ background: #323846; border-color: #4a5264; }}
QPushButton:pressed {{ background: {C['border']}; }}
QPushButton:disabled {{ color: #5f6775; border-color: {C['border']}; }}
QPushButton:checked {{ background: {C['accent2']}; border-color: {C['accent']}; color: white; }}
QPushButton:flat {{ background: transparent; border: none; color: {C['accent']}; padding: 2px 4px; }}
QPushButton#primary {{ background: {C['accent2']}; border-color: {C['accent']}; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: {C['accent']}; }}
QPushButton#danger {{ background: #4a2a2a; border-color: {C['ng']}; color: #ffb3b3; }}

QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {C['border2']}; border-radius: 4px; background: {C['bg']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; image: url("__CHECK__"); }}
QCheckBox::indicator:hover {{ border-color: {C['accent']}; }}

QSlider::groove:horizontal {{ height: 4px; background: {C['border2']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -5px 0; background: {C['text']}; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: white; }}

QTreeWidget, QTableWidget, QListWidget, QTreeView, QTableView {{
    background: {C['bg']}; alternate-background-color: #1f232a; border: 1px solid {C['border']}; border-radius: 6px;
    gridline-color: {C['border']}; selection-background-color: {C['accent2']}; selection-color: white; }}
QTreeWidget::item, QListWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {C['panel2']}; }}
QTreeWidget::branch {{ background: transparent; }}
QHeaderView::section {{ background: {C['panel2']}; color: {C['muted']}; border: none; border-right: 1px solid {C['border']}; border-bottom: 1px solid {C['border']}; padding: 5px 8px; font-weight: 600; }}
QTableWidget QTableCornerButton::section {{ background: {C['panel2']}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['border2']}; min-height: 24px; border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: #525a6b; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C['border2']}; min-width: 24px; border-radius: 4px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; border: none; width: 0; height: 0; }}

QStatusBar {{ background: {C['panel']}; border-top: 1px solid {C['border']}; color: {C['muted']}; }}
QStatusBar::item {{ border: none; }}
QScrollArea {{ border: none; background: {C['panel']}; }}
QScrollArea > QWidget > QWidget {{ background: {C['panel']}; }}
QGroupBox {{ border: 1px solid {C['border']}; border-radius: 6px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {C['muted']}; }}
QSplitter::handle {{ background: {C['bg']}; }}
QLabel#muted {{ color: {C['muted']}; }}
QLabel#chip {{ padding: 2px 10px; border-radius: 9px; background: {C['panel2']}; color: {C['text']}; }}
QFrame#card {{ background: {C['panel2']}; border: 1px solid {C['border']}; border-radius: 8px; }}
QFrame[frameShape="4"] {{ color: {C['border']}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(C["bg"]))
    pal.setColor(QPalette.WindowText, QColor(C["text"]))
    pal.setColor(QPalette.Base, QColor(C["bg"]))
    pal.setColor(QPalette.AlternateBase, QColor("#1f232a"))
    pal.setColor(QPalette.Text, QColor(C["text"]))
    pal.setColor(QPalette.Button, QColor(C["panel2"]))
    pal.setColor(QPalette.ButtonText, QColor(C["text"]))
    pal.setColor(QPalette.Highlight, QColor(C["accent2"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor(C["panel2"]))
    pal.setColor(QPalette.ToolTipText, QColor(C["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(C["muted"]))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#5f6775"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5f6775"))
    app.setPalette(pal)
    font = QFont()
    font.setFamilies(["Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei", "Segoe UI", "Sans"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET.replace("__CHECK__", _check_icon_path()))


def _check_icon_path() -> str:
    """把勾选图标画成临时 PNG（Qt 样式表只能引用文件或资源，不能内嵌 SVG）。"""
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "cvflow_check.png")
    if not os.path.exists(path):
        make_icon("check", "#ffffff", 14).pixmap(14, 14).save(path, "PNG")
    return path.replace("\\", "/")


# ---- 矢量图标（无需图片文件） ----
def make_icon(name: str, color: str = "#e4e7ec", size: int = 20) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    col = QColor(color)
    pen = QPen(col, 1.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    s = size
    m = s * 0.18
    if name == "play":
        p.setBrush(col); p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([QPointF(m + 1, m), QPointF(s - m, s / 2), QPointF(m + 1, s - m)]))
    elif name == "stop":
        p.setBrush(col); p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(m, m, s - 2 * m, s - 2 * m), 2, 2)
    elif name == "record":
        p.setBrush(col); p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(m, m, s - 2 * m, s - 2 * m))
    elif name == "new":
        path = QPainterPath()
        path.moveTo(m + 1, m); path.lineTo(s * 0.58, m); path.lineTo(s - m, s * 0.36); path.lineTo(s - m, s - m); path.lineTo(m + 1, s - m); path.closeSubpath()
        p.drawPath(path); p.drawLine(QPointF(s * 0.58, m), QPointF(s * 0.58, s * 0.36)); p.drawLine(QPointF(s * 0.58, s * 0.36), QPointF(s - m, s * 0.36))
    elif name == "open":
        path = QPainterPath()
        path.moveTo(m, s * 0.3); path.lineTo(s * 0.42, s * 0.3); path.lineTo(s * 0.5, s * 0.4); path.lineTo(s - m, s * 0.4); path.lineTo(s - m, s - m); path.lineTo(m, s - m); path.closeSubpath()
        p.drawPath(path)
    elif name == "save":
        p.drawRoundedRect(QRectF(m, m, s - 2 * m, s - 2 * m), 2, 2)
        p.drawRect(QRectF(s * 0.34, m, s * 0.32, s * 0.25)); p.drawRect(QRectF(s * 0.3, s * 0.6, s * 0.4, s * 0.22))
    elif name == "plus":
        p.drawLine(QPointF(s / 2, m), QPointF(s / 2, s - m)); p.drawLine(QPointF(m, s / 2), QPointF(s - m, s / 2))
    elif name == "minus":
        p.drawLine(QPointF(m, s / 2), QPointF(s - m, s / 2))
    elif name == "fit":
        for (x, y, dx, dy) in ((m, m, 1, 1), (s - m, m, -1, 1), (m, s - m, 1, -1), (s - m, s - m, -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * s * 0.22, y)); p.drawLine(QPointF(x, y), QPointF(x, y + dy * s * 0.22))
    elif name == "camera":
        p.drawRoundedRect(QRectF(m, s * 0.32, s - 2 * m, s * 0.48), 3, 3)
        p.drawEllipse(QRectF(s * 0.37, s * 0.42, s * 0.26, s * 0.26)); p.drawLine(QPointF(s * 0.36, s * 0.32), QPointF(s * 0.42, m + 2)); p.drawLine(QPointF(s * 0.42, m + 2), QPointF(s * 0.58, m + 2)); p.drawLine(QPointF(s * 0.58, m + 2), QPointF(s * 0.64, s * 0.32))
    elif name == "refresh":
        path = QPainterPath(); path.arcMoveTo(QRectF(m, m, s - 2 * m, s - 2 * m), 40); path.arcTo(QRectF(m, m, s - 2 * m, s - 2 * m), 40, 280)
        p.drawPath(path); p.drawLine(QPointF(s - m - 1, m + 2), QPointF(s - m - 1, s * 0.42)); p.drawLine(QPointF(s - m - 1, s * 0.42), QPointF(s * 0.62, s * 0.42))
    elif name == "plug":
        p.drawLine(QPointF(s * 0.35, m), QPointF(s * 0.35, s * 0.38)); p.drawLine(QPointF(s * 0.65, m), QPointF(s * 0.65, s * 0.38))
        p.drawRoundedRect(QRectF(s * 0.25, s * 0.38, s * 0.5, s * 0.3), 3, 3); p.drawLine(QPointF(s / 2, s * 0.68), QPointF(s / 2, s - m))
    elif name == "check":
        p.drawLine(QPointF(m, s * 0.52), QPointF(s * 0.42, s - m)); p.drawLine(QPointF(s * 0.42, s - m), QPointF(s - m, m + 1))
    elif name == "dot":
        p.setBrush(col); p.setPen(Qt.NoPen); p.drawEllipse(QRectF(s * 0.3, s * 0.3, s * 0.4, s * 0.4))
    p.end()
    return QIcon(pm)


def app_icon(size: int = 64) -> QIcon:
    """程序图标：圆角方块 + 三个相连的节点。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(C["accent2"]))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    pen = QPen(QColor("white"), size * 0.07)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    a, b, c = QPointF(size * 0.25, size * 0.35), QPointF(size * 0.55, size * 0.65), QPointF(size * 0.78, size * 0.3)
    p.drawLine(a, b); p.drawLine(b, c)
    p.setBrush(QColor("white")); p.setPen(Qt.NoPen)
    for pt in (a, b, c):
        p.drawEllipse(pt, size * 0.11, size * 0.11)
    p.end()
    return QIcon(pm)


def chip_style(color: str, fg: str = "white") -> str:
    return f"padding: 2px 10px; border-radius: 9px; background: {color}; color: {fg}; font-weight: 600;"

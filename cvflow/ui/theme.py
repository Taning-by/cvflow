"""视觉规范：设计令牌（配色、间距、控件尺寸）、字体、全局样式表，以及程序内绘制的矢量图标。

设计方向是"浅色精密仪器工作台"：灰白界面、清晰的结构边界、克制的蓝绿色强调色。
所有颜色集中在 ``C`` 里，所有尺寸集中在 ``M`` 里，改一处即可全局生效。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (QColor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import QApplication, QLabel, QSizePolicy

from ..core.node import NodeStatus
from ..core.types import DataType

# ---- 配色令牌 ----
# 旧代码里的键名全部保留，取值换成浅色，另外补充语义色。
C = {
    "bg": "#F3F5F7",          # 应用底色
    "panel": "#FFFFFF",       # 工作面板
    "panel2": "#EDF1F4",      # 面板内的次级底（表头、分组条、胶囊）
    "panel3": "#F7F9FA",      # 更浅的次级底（隔行、输入框禁用态）
    "canvas": "#F7F8FA",      # 流程画布
    "border": "#DCE3E8",      # 面板边界
    "border2": "#C6D0D8",     # 控件边界（对比更强）
    "text": "#1F2933",        # 主要文字
    "muted": "#586572",       # 次要文字
    "faint": "#8A97A3",       # 第三级文字（占位符、禁用）
    "accent": "#176B75",      # 品牌强调色
    "accent2": "#115159",     # 强调色按下态
    "accent_bg": "#E8F3F4",   # 选中浅底色
    "ok": "#23704F",
    "ok_bg": "#E4F1EB",
    "warn": "#945D00",
    "warn_bg": "#FAF0DE",
    "ng": "#C43D4B",
    "ng_bg": "#FAE9EB",
    "sel": "#176B75",         # 兼容旧名：选中色＝品牌色
    "viewer_bg": "#2B3036",   # 图像查看器底色（中性深灰，只用于图像区域）
    "roi": "#FF8A1E",         # 图像上 ROI 的默认颜色，与品牌色分开，用户可改
}

# ---- 尺寸令牌（逻辑像素，4/8 栅格） ----
M = {
    "gap": 8,            # 基础间距
    "gap_s": 4,
    "gap_l": 12,
    "radius": 6,         # 面板/卡片圆角
    "radius_s": 4,       # 控件圆角
    "ctl_h": 30,         # 常用控件高度
    "font": 13,          # 正文字号
    "font_s": 12,        # 次要文字
    "font_title": 14,    # 面板标题
    "label_w": 88,       # 参数标签列宽
}

# 状态色：这五种状态在界面上永远同时带文字/图形，不只靠颜色区分。
STATUS_COLORS = {
    NodeStatus.IDLE: "#8A97A3",
    NodeStatus.RUNNING: C["accent"],
    NodeStatus.OK: C["ok"],
    NodeStatus.NG: C["ng"],
    NodeStatus.ERROR: C["warn"],
    NodeStatus.SKIPPED: "#9AA6B2",
}

# 状态的文字与字形：执行成功(OK)、检测不合格(NG)、算法执行失败(错误)是三件事。
STATUS_TEXT = {
    NodeStatus.IDLE: "待运行", NodeStatus.RUNNING: "运行中", NodeStatus.OK: "OK",
    NodeStatus.NG: "NG", NodeStatus.ERROR: "错误", NodeStatus.SKIPPED: "跳过",
}
STATUS_GLYPH = {
    NodeStatus.IDLE: "○", NodeStatus.RUNNING: "◐", NodeStatus.OK: "✓",
    NodeStatus.NG: "✕", NodeStatus.ERROR: "!", NodeStatus.SKIPPED: "–",
}

# 端口/连线的数据类型色：在浅色底上保证对比度。
DTYPE_COLORS = {
    DataType.IMAGE: "#1B6FA8",
    DataType.INT: "#3F7A34", DataType.FLOAT: "#3F7A34", DataType.BOOL: "#4F8A3F",
    DataType.STRING: "#9A5B1E",
    DataType.RECT: "#7044A0", DataType.POINT: "#7044A0", DataType.CIRCLE: "#7044A0", DataType.LINE: "#7044A0",
    DataType.LIST: "#5A6774", DataType.DICT: "#5A6774", DataType.CONTOURS: "#5A6774", DataType.TENSOR: "#5A6774",
    DataType.ANY: "#78848F",
}

FONT_FAMILIES = ["Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC",
                 "WenQuanYi Micro Hei", "Segoe UI", "Sans"]
MONO_FAMILIES = ["Cascadia Mono", "Consolas", "SF Mono", "DejaVu Sans Mono", "WenQuanYi Micro Hei Mono", "Monospace"]

STYLESHEET = f"""
* {{ outline: none; }}
QMainWindow, QDialog {{ background: {C['bg']}; color: {C['text']}; }}
QWidget {{ color: {C['text']}; font-size: {M['font']}px; }}
QMainWindow::separator {{ background: {C['bg']}; width: 6px; height: 6px; }}
QMainWindow::separator:hover {{ background: {C['accent_bg']}; }}
QToolTip {{ background: {C['panel']}; color: {C['text']}; border: 1px solid {C['border2']};
            padding: 5px 8px; border-radius: {M['radius_s']}px; }}

QMenuBar {{ background: {C['panel']}; padding: 2px 6px; border-bottom: 1px solid {C['border']}; }}
QMenuBar::item {{ padding: 5px 10px; border-radius: {M['radius_s']}px; }}
QMenuBar::item:selected {{ background: {C['accent_bg']}; color: {C['accent']}; }}
QMenu {{ background: {C['panel']}; border: 1px solid {C['border2']}; padding: 5px; }}
QMenu::item {{ padding: 6px 28px 6px 14px; border-radius: {M['radius_s']}px; }}
QMenu::item:selected {{ background: {C['accent_bg']}; color: {C['accent']}; }}
QMenu::item:disabled {{ color: {C['faint']}; }}
QMenu::separator {{ height: 1px; background: {C['border']}; margin: 5px 8px; }}

QToolBar {{ background: {C['panel']}; border: none; border-bottom: 1px solid {C['border']};
            padding: 5px 8px; spacing: {M['gap_s']}px; }}
QToolBar::separator {{ width: 1px; background: {C['border']}; margin: 5px 7px; }}
QToolButton {{ background: transparent; color: {C['text']}; border: 1px solid transparent;
               border-radius: {M['radius_s']}px; padding: 5px 9px; }}
QToolButton:hover {{ background: {C['panel2']}; border-color: {C['border']}; }}
QToolButton:pressed {{ background: {C['border']}; }}
QToolButton:disabled {{ color: {C['faint']}; }}
QToolButton:checked {{ background: {C['accent_bg']}; border-color: {C['accent']}; color: {C['accent']}; }}
QToolButton#primary {{ background: {C['accent']}; color: #FFFFFF; border-color: {C['accent']}; font-weight: 600; }}
QToolButton#primary:hover {{ background: {C['accent2']}; border-color: {C['accent2']}; }}
QToolButton#primary:disabled {{ background: {C['panel2']}; color: {C['faint']}; border-color: {C['border']}; }}
QToolButton#runmode:checked {{ background: {C['ng_bg']}; border-color: {C['ng']}; color: {C['ng']}; font-weight: 600; }}

QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; color: {C['muted']}; }}
QDockWidget::title {{ background: {C['bg']}; padding: 6px 10px; border: none;
                      font-size: {M['font_s']}px; font-weight: 600; color: {C['muted']}; text-align: left; }}
QDockWidget > QWidget {{ background: {C['panel']}; border: 1px solid {C['border']};
                         border-radius: {M['radius']}px; }}

QTabWidget::pane {{ border: none; background: {C['panel']}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {C['muted']}; padding: 7px 14px;
                border-bottom: 2px solid transparent; margin-right: 2px; }}
QTabBar::tab:selected {{ color: {C['accent']}; border-bottom: 2px solid {C['accent']}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {C['text']}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {C['panel']}; color: {C['text']}; border: 1px solid {C['border2']};
    border-radius: {M['radius_s']}px; padding: 4px 8px; min-height: 20px;
    selection-background-color: {C['accent']}; selection-color: #FFFFFF; }}
QPlainTextEdit, QTextEdit {{ padding: 6px 8px; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {C['muted']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {C['accent']}; }}
QLineEdit[invalid="true"], QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"] {{
    border-color: {C['ng']}; background: {C['ng_bg']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{
    color: {C['faint']}; background: {C['panel3']}; border-color: {C['border']}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 15px; background: transparent; border: none; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; border-left: 4px solid transparent;
    border-right: 4px solid transparent; border-top: 5px solid {C['muted']}; margin-right: 7px; }}
QComboBox QAbstractItemView {{ background: {C['panel']}; border: 1px solid {C['border2']};
    selection-background-color: {C['accent_bg']}; selection-color: {C['accent']}; padding: 4px; }}

QPushButton {{ background: {C['panel']}; color: {C['text']}; border: 1px solid {C['border2']};
               border-radius: {M['radius_s']}px; padding: 5px 14px; min-height: 20px; }}
QPushButton:hover {{ background: {C['panel2']}; border-color: {C['muted']}; }}
QPushButton:pressed {{ background: {C['border']}; }}
QPushButton:disabled {{ color: {C['faint']}; border-color: {C['border']}; background: {C['panel3']}; }}
QPushButton:checked {{ background: {C['accent_bg']}; border-color: {C['accent']}; color: {C['accent']}; }}
QPushButton:flat {{ background: transparent; border: none; color: {C['accent']}; padding: 3px 4px; }}
QPushButton#primary {{ background: {C['accent']}; border-color: {C['accent']}; color: #FFFFFF; font-weight: 600; }}
QPushButton#primary:hover {{ background: {C['accent2']}; border-color: {C['accent2']}; }}
QPushButton#danger {{ background: {C['ng_bg']}; border-color: {C['ng']}; color: {C['ng']}; }}
QPushButton#compact {{ padding: 4px 8px; }}
QPushButton#linklike {{ background: transparent; border: none; color: {C['accent']}; padding: 2px 6px;
                        text-align: left; }}
QPushButton#linklike:hover {{ color: {C['accent2']}; text-decoration: underline; }}

QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {C['border2']};
                        border-radius: {M['radius_s']}px; background: {C['panel']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; image: url("__CHECK__"); }}
QCheckBox::indicator:hover {{ border-color: {C['accent']}; }}
QCheckBox:disabled {{ color: {C['faint']}; }}

QSlider::groove:horizontal {{ height: 4px; background: {C['border']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 13px; height: 13px; margin: -5px 0; background: {C['panel']};
                              border: 1px solid {C['accent']}; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {C['accent_bg']}; }}

QTreeWidget, QTableWidget, QListWidget, QTreeView, QTableView {{
    background: {C['panel']}; alternate-background-color: {C['panel3']};
    border: 1px solid {C['border']}; border-radius: {M['radius']}px;
    gridline-color: {C['border']}; selection-background-color: {C['accent_bg']}; selection-color: {C['text']}; }}
QTreeWidget::item, QListWidget::item, QTableWidget::item {{ padding: 3px 2px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background: {C['panel2']}; }}
QTreeWidget::item:selected, QListWidget::item:selected, QTableWidget::item:selected {{
    background: {C['accent_bg']}; color: {C['text']}; }}
QTreeWidget::branch {{ background: transparent; }}
QHeaderView::section {{ background: {C['panel2']}; color: {C['muted']}; border: none;
    border-right: 1px solid {C['border']}; border-bottom: 1px solid {C['border']};
    padding: 5px 8px; font-size: {M['font_s']}px; font-weight: 600; }}
QTableWidget QTableCornerButton::section {{ background: {C['panel2']}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #C3CDD6; min-height: 26px; border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: {C['muted']}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #C3CDD6; min-width: 26px; border-radius: 4px; }}
QScrollBar::handle:horizontal:hover {{ background: {C['muted']}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none; border: none; width: 0; height: 0; }}

QStatusBar {{ background: {C['panel']}; border-top: 1px solid {C['border']}; color: {C['muted']}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {C['muted']}; }}
QScrollArea {{ border: none; background: {C['panel']}; }}
QScrollArea > QWidget > QWidget {{ background: {C['panel']}; }}
QGroupBox {{ border: 1px solid {C['border']}; border-radius: {M['radius']}px; margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {C['muted']}; }}
QSplitter::handle {{ background: {C['bg']}; }}
QSplitter::handle:horizontal {{ width: 6px; }}
QSplitter::handle:vertical {{ height: 6px; }}
QSplitter::handle:hover {{ background: {C['accent_bg']}; }}
QLabel#muted {{ color: {C['muted']}; font-size: {M['font_s']}px; }}
QLabel#chip {{ padding: 2px 10px; border-radius: 9px; background: {C['panel2']}; color: {C['muted']}; }}
QFrame#card {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: {M['radius']}px; }}
QFrame#paneHeader {{ background: {C['panel']}; border: none; border-bottom: 1px solid {C['border']}; }}
QWidget#pane {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: {M['radius']}px; }}
QFrame[frameShape="4"] {{ color: {C['border']}; }}
QFrame[frameShape="5"] {{ color: {C['border']}; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(C["bg"]))
    pal.setColor(QPalette.WindowText, QColor(C["text"]))
    pal.setColor(QPalette.Base, QColor(C["panel"]))
    pal.setColor(QPalette.AlternateBase, QColor(C["panel3"]))
    pal.setColor(QPalette.Text, QColor(C["text"]))
    pal.setColor(QPalette.Button, QColor(C["panel"]))
    pal.setColor(QPalette.ButtonText, QColor(C["text"]))
    pal.setColor(QPalette.Highlight, QColor(C["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ToolTipBase, QColor(C["panel"]))
    pal.setColor(QPalette.ToolTipText, QColor(C["text"]))
    pal.setColor(QPalette.PlaceholderText, QColor(C["faint"]))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(C["faint"]))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(C["faint"]))
    app.setPalette(pal)
    app.setFont(ui_font())
    app.setStyleSheet(STYLESHEET.replace("__CHECK__", _check_icon_path()))


def ui_font(size: int | None = None, bold: bool = False) -> QFont:
    """界面字体：按逻辑像素给字号，随显示缩放一起放大。"""
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPixelSize(int(size or M["font"]))
    if bold:
        f.setWeight(QFont.DemiBold)
    return f


def mono_font(size: int | None = None) -> QFont:
    """等宽字体：只给代码、路径、报文这类内容用。"""
    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setStyleHint(QFont.Monospace)
    f.setFixedPitch(True)
    f.setPixelSize(int(size or M["font_s"]))
    return f


def tabular(font: QFont) -> QFont:
    """等宽数字（tabular figures）：数值列对齐，但正文仍用比例字体。"""
    f = QFont(font)
    try:
        f.setFeature(QFont.Tag("tnum"), 1)
    except Exception:       # 老版本 Qt 没有字体特性接口，退化为普通数字
        pass
    return f


def _check_icon_path() -> str:
    """把勾选图标画成临时 PNG（Qt 样式表只能引用文件或资源，不能内嵌 SVG）。"""
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "cvflow_check_light.png")
    if not os.path.exists(path):
        make_icon("check", "#FFFFFF", 14).pixmap(14, 14).save(path, "PNG")
    return path.replace("\\", "/")


# ---- 矢量图标（无需图片文件） ----
def make_icon(name: str, color: str | None = None, size: int = 18) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    col = QColor(color or C["muted"])
    pen = QPen(col, 1.6)
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
    elif name == "loop":                      # 连续运行：循环箭头
        path = QPainterPath()
        path.arcMoveTo(QRectF(m, m, s - 2 * m, s - 2 * m), 120)
        path.arcTo(QRectF(m, m, s - 2 * m, s - 2 * m), 120, 300)
        p.drawPath(path)
        tip = QPointF(s * 0.5, m)
        p.setBrush(col); p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF([tip, QPointF(tip.x() - s * 0.16, tip.y() - s * 0.02),
                                 QPointF(tip.x() - s * 0.06, tip.y() + s * 0.16)]))
    elif name == "new":
        path = QPainterPath()
        path.moveTo(m + 1, m); path.lineTo(s * 0.58, m); path.lineTo(s - m, s * 0.36)
        path.lineTo(s - m, s - m); path.lineTo(m + 1, s - m); path.closeSubpath()
        p.drawPath(path)
        p.drawLine(QPointF(s * 0.58, m), QPointF(s * 0.58, s * 0.36))
        p.drawLine(QPointF(s * 0.58, s * 0.36), QPointF(s - m, s * 0.36))
    elif name == "open":
        path = QPainterPath()
        path.moveTo(m, s * 0.3); path.lineTo(s * 0.42, s * 0.3); path.lineTo(s * 0.5, s * 0.4)
        path.lineTo(s - m, s * 0.4); path.lineTo(s - m, s - m); path.lineTo(m, s - m); path.closeSubpath()
        p.drawPath(path)
    elif name == "save":
        p.drawRoundedRect(QRectF(m, m, s - 2 * m, s - 2 * m), 2, 2)
        p.drawRect(QRectF(s * 0.34, m, s * 0.32, s * 0.25))
        p.drawRect(QRectF(s * 0.3, s * 0.6, s * 0.4, s * 0.22))
    elif name == "plus":
        p.drawLine(QPointF(s / 2, m), QPointF(s / 2, s - m)); p.drawLine(QPointF(m, s / 2), QPointF(s - m, s / 2))
    elif name == "minus":
        p.drawLine(QPointF(m, s / 2), QPointF(s - m, s / 2))
    elif name == "fit":                       # 四角定位标记：也是本软件的视觉母题
        for (x, y, dx, dy) in ((m, m, 1, 1), (s - m, m, -1, 1), (m, s - m, 1, -1), (s - m, s - m, -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * s * 0.24, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * s * 0.24))
    elif name == "one_to_one":
        p.setFont(ui_font(int(s * 0.62), True))
        p.drawText(QRectF(0, 0, s, s), Qt.AlignCenter, "1:1")
    elif name == "zoom":
        p.drawEllipse(QRectF(m, m, s * 0.5, s * 0.5))
        p.drawLine(QPointF(s * 0.62, s * 0.62), QPointF(s - m, s - m))
    elif name == "search":
        p.drawEllipse(QRectF(m, m, s * 0.48, s * 0.48))
        p.drawLine(QPointF(s * 0.6, s * 0.6), QPointF(s - m, s - m))
    elif name == "split_h":                   # 左右分栏
        p.drawRoundedRect(QRectF(m, m + 1, s - 2 * m, s - 2 * m - 2), 2, 2)
        p.drawLine(QPointF(s / 2, m + 1), QPointF(s / 2, s - m - 1))
    elif name == "split_v":                   # 上下分栏
        p.drawRoundedRect(QRectF(m, m + 1, s - 2 * m, s - 2 * m - 2), 2, 2)
        p.drawLine(QPointF(m, s / 2), QPointF(s - m, s / 2))
    elif name == "expand":
        p.drawLine(QPointF(m, s * 0.42), QPointF(m, m)); p.drawLine(QPointF(m, m), QPointF(s * 0.42, m))
        p.drawLine(QPointF(s - m, s * 0.58), QPointF(s - m, s - m))
        p.drawLine(QPointF(s - m, s - m), QPointF(s * 0.58, s - m))
        p.drawLine(QPointF(m, m), QPointF(s * 0.42, s * 0.42))
        p.drawLine(QPointF(s - m, s - m), QPointF(s * 0.58, s * 0.58))
    elif name == "camera":
        p.drawRoundedRect(QRectF(m, s * 0.32, s - 2 * m, s * 0.48), 3, 3)
        p.drawEllipse(QRectF(s * 0.37, s * 0.42, s * 0.26, s * 0.26))
        p.drawLine(QPointF(s * 0.36, s * 0.32), QPointF(s * 0.42, m + 2))
        p.drawLine(QPointF(s * 0.42, m + 2), QPointF(s * 0.58, m + 2))
        p.drawLine(QPointF(s * 0.58, m + 2), QPointF(s * 0.64, s * 0.32))
    elif name == "refresh":
        path = QPainterPath()
        path.arcMoveTo(QRectF(m, m, s - 2 * m, s - 2 * m), 40)
        path.arcTo(QRectF(m, m, s - 2 * m, s - 2 * m), 40, 280)
        p.drawPath(path)
        p.drawLine(QPointF(s - m - 1, m + 2), QPointF(s - m - 1, s * 0.42))
        p.drawLine(QPointF(s - m - 1, s * 0.42), QPointF(s * 0.62, s * 0.42))
    elif name == "plug":
        p.drawLine(QPointF(s * 0.35, m), QPointF(s * 0.35, s * 0.38))
        p.drawLine(QPointF(s * 0.65, m), QPointF(s * 0.65, s * 0.38))
        p.drawRoundedRect(QRectF(s * 0.25, s * 0.38, s * 0.5, s * 0.3), 3, 3)
        p.drawLine(QPointF(s / 2, s * 0.68), QPointF(s / 2, s - m))
    elif name == "check":
        p.drawLine(QPointF(m, s * 0.52), QPointF(s * 0.42, s - m))
        p.drawLine(QPointF(s * 0.42, s - m), QPointF(s - m, m + 1))
    elif name == "dot":
        p.setBrush(col); p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(s * 0.3, s * 0.3, s * 0.4, s * 0.4))
    elif name == "chevron_down":
        p.drawLine(QPointF(s * 0.28, s * 0.4), QPointF(s / 2, s * 0.62))
        p.drawLine(QPointF(s / 2, s * 0.62), QPointF(s * 0.72, s * 0.4))
    elif name == "chevron_right":
        p.drawLine(QPointF(s * 0.4, s * 0.28), QPointF(s * 0.62, s / 2))
        p.drawLine(QPointF(s * 0.62, s / 2), QPointF(s * 0.4, s * 0.72))
    elif name == "roi":                       # 矩形 + 四角标记
        p.drawRect(QRectF(s * 0.3, s * 0.3, s * 0.4, s * 0.4))
        for (x, y, dx, dy) in ((m, m, 1, 1), (s - m, m, -1, 1), (m, s - m, 1, -1), (s - m, s - m, -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + dx * s * 0.18, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + dy * s * 0.18))
    p.end()
    return QIcon(pm)


def app_icon(size: int = 64) -> QIcon:
    """程序图标：品牌色圆角方块 + ROI 四角定位标记 + 流程节点连线。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(C["accent"]))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    pen = QPen(QColor(255, 255, 255, 150), size * 0.05)
    pen.setCapStyle(Qt.FlatCap)
    p.setPen(pen)
    k, e = size * 0.18, size * 0.17
    for (x, y, dx, dy) in ((k, k, 1, 1), (size - k, k, -1, 1), (k, size - k, 1, -1), (size - k, size - k, -1, -1)):
        p.drawLine(QPointF(x, y), QPointF(x + dx * e, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + dy * e))
    pen = QPen(QColor("white"), size * 0.07)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    a, b, c = QPointF(size * 0.3, size * 0.4), QPointF(size * 0.5, size * 0.6), QPointF(size * 0.7, size * 0.38)
    p.drawLine(a, b); p.drawLine(b, c)
    p.setBrush(QColor("white")); p.setPen(Qt.NoPen)
    for pt in (a, b, c):
        p.drawEllipse(pt, size * 0.085, size * 0.085)
    p.end()
    return QIcon(pm)


class ElidedLabel(QLabel):
    """放不下时用省略号显示的标签。

    普通 QLabel 的最小宽度等于整行文字的宽度，放在面板标题栏里会把整个面板的最小宽度
    撑到"文字有多长就有多宽"，用户就再也拖不动面板之间的分隔条了。这个标签把最小宽度
    压到很小，宽度不够时自己省略。``text()`` 仍返回完整文字，提示信息与自动化测试照常工作。

    只用于纯文本；富文本（带标签的）按原样显示，不做省略。
    """

    def __init__(self, text: str = "", mode: Qt.TextElideMode = Qt.ElideRight, parent=None) -> None:
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        # 横向用 Preferred：有地方时按文字宽度排布，地方不够时能一路缩到
        # 下面 minimumSizeHint 给的那个很小的值（Ignored 会让它直接被压成 0）。
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text or ""
        self._elide()

    def text(self) -> str:
        return self._full

    def setElideMode(self, mode: Qt.TextElideMode) -> None:
        self._mode = mode
        self._elide()

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), 24), hint.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        if "<" in self._full:          # 富文本按原样显示：省略是按纯文本算的，会把标签截坏
            super().setText(self._full)
            return
        width = max(0, self.width() - 2)
        shown = QFontMetrics(self.font()).elidedText(self._full, self._mode, width) if width else self._full
        super().setText(shown)


def chip_style(color: str, fg: str = "white") -> str:
    return (f"padding: 2px 10px; border-radius: 9px; background: {color}; color: {fg}; "
            f"font-size: {M['font_s']}px; font-weight: 600;")


def soft_chip_style(fg: str, bg: str, border: str | None = None, selector: str = "") -> str:
    """浅底 + 深色文字的状态胶囊，用于状态栏与横幅。

    ``selector`` 传入 ``QLabel#xxx`` 时生成带选择器的规则：状态栏里的 QLabel 有全局规则，
    不带选择器的内联样式会被盖掉。
    """
    body = (f"padding: 2px 10px; border-radius: 9px; background: {bg}; color: {fg}; "
            f"border: 1px solid {border or bg}; font-size: {M['font_s']}px; font-weight: 600;")
    return f"{selector} {{ {body} }}" if selector else body

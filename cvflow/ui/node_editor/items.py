"""节点、端口、连线的图形项。

流程图形语言（产品辨识度的主要来源）：
* 节点＝浅色卡片，左侧 4px 分类色书脊，标题右侧是状态胶囊（字形＋文字＋颜色，三重编码）。
* 端口统一为圆点，位置与原来一致；必填端口实心，可选端口空心，不靠颜色区分。
* 底栏左边是参数摘要（扫一眼就知道这个节点在做什么），右边是耗时或错误。
* 选中＝细品牌色边框 + 浅底 + 四角定位标记，呼应图像上的 ROI。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetricsF, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem

from ...core.graph import Link
from ...core.node import Node, NodeStatus, Port
from ..i18n import tr
from ..theme import C, DTYPE_COLORS, STATUS_COLORS, STATUS_GLYPH, STATUS_TEXT, ui_font

NODE_W = 184          # 与既有方案文件里的节点间距保持一致，不改变旧流程的疏密
HEADER_H = 28
ROW_H = 22
PORT_R = 5.0
FOOT_H = 22
RADIUS = 6
SPINE_W = 4.0
CORNER = 7.0          # 选中时四角定位标记的长度

_PILL_STATES = (NodeStatus.OK, NodeStatus.NG, NodeStatus.ERROR, NodeStatus.RUNNING)


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", port: Port, is_output: bool) -> None:
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, node_item)
        self.node_item = node_item
        self.port = port
        self.is_output = is_output
        self.color = QColor(DTYPE_COLORS.get(port.dtype, C["faint"]))
        # 必填端口实心、可选端口空心：形状本身带信息，不只靠颜色。
        self.setBrush(QBrush(QColor(C["panel"]) if port.optional else self.color))
        self.setPen(QPen(self.color, 1.8))
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.setToolTip(f"{port.label}（{port.name}）: {port.dtype.value}" + ("，可选" if port.optional else "")
                        + (f"\n{tr(port.description)}" if port.description else ""))

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))

    def hoverEnterEvent(self, event) -> None:
        self.setScale(1.4)
        self.setPen(QPen(QColor(C["accent"]), 2))

    def hoverLeaveEvent(self, event) -> None:
        self.setScale(1.0)
        self.setPen(QPen(self.color, 1.8))

    def mousePressEvent(self, event) -> None:
        self.scene().start_link(self, event.scenePos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self.scene().update_temp_link(event.scenePos())

    def mouseReleaseEvent(self, event) -> None:
        self.scene().finish_link(event.scenePos())


class NodeItem(QGraphicsItem):
    def __init__(self, node: Node) -> None:
        super().__init__()
        self.node = node
        self.inputs: dict[str, PortItem] = {}
        self.outputs: dict[str, PortItem] = {}
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable |
                      QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(1)
        rows = max(len(node.inputs), len(node.outputs), 1)
        self._body_top = HEADER_H + 5
        self._h = self._body_top + rows * ROW_H + 5 + FOOT_H
        for i, p in enumerate(node.inputs):
            it = PortItem(self, p, False)
            it.setPos(0, self._body_top + i * ROW_H + ROW_H / 2)
            self.inputs[p.name] = it
        for i, p in enumerate(node.outputs):
            it = PortItem(self, p, True)
            it.setPos(NODE_W, self._body_top + i * ROW_H + ROW_H / 2)
            self.outputs[p.name] = it
        self.setPos(node.position[0], node.position[1])
        self.setToolTip(f"{tr(node.label)} [{node.type_id}]\n{tr(node.description)}")

    def boundingRect(self) -> QRectF:
        pad = PORT_R + CORNER
        return QRectF(-pad, -pad, NODE_W + 2 * pad, self._h + 2 * pad)

    def shape(self) -> QPainterPath:
        """命中区域只有卡片本身：外扩的四角标记不参与点击，免得盖住邻近节点的端口。"""
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, NODE_W, self._h), RADIUS, RADIUS)
        return path

    # ---- 参数摘要：底栏里一行能扫到的关键设置 ----
    def _summary(self) -> str:
        bits: list[str] = []
        for p in self.node.params:
            if p.advanced or p.kind in ("code", "json", "rect", "list"):
                continue
            try:
                v = self.node.get(p.name)
            except Exception:
                continue
            if v is None or v == "":
                continue
            if p.kind == "bool":
                if not v:
                    continue
                bits.append(tr(p.label))
            elif p.kind == "float":
                bits.append(f"{tr(p.label)} {float(v):g}")
            elif p.kind in ("file", "dir"):
                bits.append(str(v).replace("\\", "/").rsplit("/", 1)[-1])
            else:
                bits.append(f"{tr(p.label)} {v}")
            if len(bits) == 2:
                break
        return " · ".join(bits)

    def paint(self, painter, option, widget=None) -> None:
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        body = QRectF(0, 0, NODE_W, self._h)
        node = self.node
        enabled = node.enabled
        st = node.status
        selected = self.isSelected()
        status_col = QColor(STATUS_COLORS.get(st, C["faint"]))

        path = QPainterPath()
        path.addRoundedRect(body, RADIUS, RADIUS)
        # 极轻的投影：只用来把卡片从画布上托起，不做装饰
        painter.fillPath(QPainterPath(path).translated(0, 1.5), QColor(31, 41, 51, 22))
        painter.fillPath(path, QColor(C["panel"] if enabled else C["panel3"]))

        # 标题区底色：选中时用品牌浅底
        head = QPainterPath()
        head.addRoundedRect(QRectF(0, 0, NODE_W, HEADER_H), RADIUS, RADIUS)
        head.addRect(QRectF(0, HEADER_H - RADIUS, NODE_W, RADIUS))
        painter.fillPath(head.simplified(), QColor(C["accent_bg"] if selected else C["panel2"]))

        # 左侧分类色书脊
        spine = QPainterPath()
        spine.addRoundedRect(QRectF(0, 0, SPINE_W * 2, self._h), RADIUS, RADIUS)
        spine.addRect(QRectF(SPINE_W, 0, SPINE_W, self._h))
        cat = QColor(node.color)
        if not enabled:
            cat.setAlpha(90)
        painter.fillPath(spine.simplified().intersected(path), cat)

        # 标题文字
        painter.setPen(QPen(QColor(C["text"] if enabled else C["faint"])))
        f = ui_font(13, True)
        painter.setFont(f)
        pill_w = self._pill_width(st)
        name_rect = QRectF(SPINE_W + 8, 0, NODE_W - SPINE_W - 18 - pill_w, HEADER_H)
        name = QFontMetricsF(f).elidedText(node.name, Qt.ElideMiddle, name_rect.width())
        painter.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, name)
        self._draw_pill(painter, QRectF(NODE_W - 8 - pill_w, (HEADER_H - 15) / 2, pill_w, 15), st, status_col, enabled)

        # 端口标签
        painter.setFont(ui_font(12))
        painter.setPen(QPen(QColor(C["muted"] if enabled else C["faint"])))
        for i, p in enumerate(node.inputs):
            y = self._body_top + i * ROW_H
            painter.drawText(QRectF(SPINE_W + 8, y, NODE_W / 2 - 8, ROW_H), Qt.AlignVCenter | Qt.AlignLeft, p.label)
        for i, p in enumerate(node.outputs):
            y = self._body_top + i * ROW_H
            painter.drawText(QRectF(NODE_W / 2, y, NODE_W / 2 - 10, ROW_H), Qt.AlignVCenter | Qt.AlignRight, p.label)

        # 底栏：左＝参数摘要，右＝耗时 / 错误 / 已禁用
        foot_y = self._h - FOOT_H
        painter.setPen(QPen(QColor(C["border"])))
        painter.drawLine(QPointF(SPINE_W + 1, foot_y), QPointF(NODE_W - 1, foot_y))
        painter.setFont(ui_font(11))
        if not enabled:
            right = "已禁用"
        elif st in (NodeStatus.OK, NodeStatus.NG, NodeStatus.ERROR):
            right = f"{node.last_time_ms:.1f} ms"
        elif st == NodeStatus.SKIPPED and node.last_error:
            right = node.last_error
        else:
            right = ""
        fm = ui_font(11)
        metrics = QFontMetricsF(fm)
        right = metrics.elidedText(right, Qt.ElideRight, NODE_W * 0.45)
        rw = metrics.horizontalAdvance(right) + 10
        painter.setPen(QPen(QColor(C["ng"] if (st == NodeStatus.ERROR and enabled) else C["faint"])))
        painter.drawText(QRectF(NODE_W - 8 - rw, foot_y, rw, FOOT_H), Qt.AlignVCenter | Qt.AlignRight, right)
        summary = metrics.elidedText(self._summary(), Qt.ElideRight, NODE_W - SPINE_W - 18 - rw)
        painter.setPen(QPen(QColor(C["muted"] if enabled else C["faint"])))
        painter.drawText(QRectF(SPINE_W + 8, foot_y, NODE_W - SPINE_W - 16 - rw, FOOT_H),
                         Qt.AlignVCenter | Qt.AlignLeft, summary)

        # 边框：选中 > 失败 > 不合格 > 运行中 > 普通
        if selected:
            painter.setPen(QPen(QColor(C["accent"]), 1.6))
        elif not enabled:
            pen = QPen(QColor(C["border2"]), 1)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
        elif st in (NodeStatus.ERROR, NodeStatus.NG, NodeStatus.RUNNING):
            painter.setPen(QPen(status_col, 1.4))
        else:
            painter.setPen(QPen(QColor(C["border2"]), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        if selected:
            self._draw_corners(painter, body)

    def _pill_width(self, st: NodeStatus) -> float:
        text = f"{STATUS_GLYPH.get(st, '')} {STATUS_TEXT.get(st, st.value)}"
        return QFontMetricsF(ui_font(11, True)).horizontalAdvance(text) + 14

    def _draw_pill(self, painter, rect: QRectF, st: NodeStatus, col: QColor, enabled: bool) -> None:
        """状态胶囊：字形 + 文字 + 颜色。成功 / 不合格 / 失败彼此不会混淆。"""
        pill = QPainterPath()
        pill.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        if not enabled:
            fill, fg = QColor(C["panel2"]), QColor(C["faint"])
        elif st in _PILL_STATES:
            fill, fg = QColor(col), QColor("#FFFFFF")
        else:
            fill, fg = QColor(C["panel2"]), QColor(C["muted"])
        painter.fillPath(pill, fill)
        painter.setPen(QPen(fg))
        painter.setFont(ui_font(11, True))
        painter.drawText(rect, Qt.AlignCenter, f"{STATUS_GLYPH.get(st, '')} {STATUS_TEXT.get(st, st.value)}")

    def _draw_corners(self, painter, body: QRectF) -> None:
        """四角定位标记：与图像上的 ROI 标记同一套语言，只用在选中的节点上。"""
        pen = QPen(QColor(C["accent"]), 1.6)
        pen.setCapStyle(Qt.FlatCap)
        painter.setPen(pen)
        o = 3.0
        for (x, y, dx, dy) in ((body.left() - o, body.top() - o, 1, 1), (body.right() + o, body.top() - o, -1, 1),
                               (body.left() - o, body.bottom() + o, 1, -1), (body.right() + o, body.bottom() + o, -1, -1)):
            painter.drawLine(QPointF(x, y), QPointF(x + dx * CORNER, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + dy * CORNER))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node.position = [float(self.pos().x()), float(self.pos().y())]
            sc = self.scene()
            if sc is not None:
                sc.update_links_for(self.node.id)
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event) -> None:
        self.scene().node_double_clicked.emit(self.node.id)
        super().mouseDoubleClickEvent(event)


class LinkItem(QGraphicsPathItem):
    def __init__(self, link: Link, src: PortItem, dst: PortItem) -> None:
        super().__init__()
        self.link = link
        self.src = src
        self.dst = dst
        self.setZValue(0)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        color = QColor(DTYPE_COLORS.get(src.port.dtype, C["faint"]))
        color.setAlpha(170)                       # 连线克制：不与节点抢视觉重量
        self._pen = QPen(color, 1.6)
        self._pen.setCapStyle(Qt.RoundCap)
        self._pen_sel = QPen(QColor(C["accent"]), 2.4)
        self._pen_sel.setCapStyle(Qt.RoundCap)
        self.setPen(self._pen)
        self.update_path()

    def update_path(self) -> None:
        a, b = self.src.scene_center(), self.dst.scene_center()
        self.setPath(bezier(a, b))

    def shape(self):
        from PySide6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(10)
        return stroker.createStroke(self.path())

    def paint(self, painter, option, widget=None) -> None:
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        painter.setPen(self._pen_sel if self.isSelected() else self._pen)
        painter.drawPath(self.path())


def bezier(a: QPointF, b: QPointF):
    path = QPainterPath(a)
    dx = max(50.0, abs(b.x() - a.x()) * 0.5)
    path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
    return path

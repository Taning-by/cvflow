"""节点、端口、连线的图形项。"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem

from ...core.graph import Link
from ...core.node import Node, NodeStatus, Port
from ..i18n import tr
from ..theme import C, DTYPE_COLORS, STATUS_COLORS

NODE_W = 184
HEADER_H = 30
ROW_H = 20
PORT_R = 5.5
FOOT_H = 22
RADIUS = 9

_STATUS_TEXT = {NodeStatus.OK: "OK", NodeStatus.NG: "NG", NodeStatus.ERROR: "错误", NodeStatus.SKIPPED: "跳过",
                NodeStatus.RUNNING: "运行中", NodeStatus.IDLE: "待运行"}


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", port: Port, is_output: bool) -> None:
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, node_item)
        self.node_item = node_item
        self.port = port
        self.is_output = is_output
        self.color = QColor(DTYPE_COLORS.get(port.dtype, "#d7dbe2"))
        self.setBrush(QBrush(self.color))
        self.setPen(QPen(QColor(C["canvas"]), 2))
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.setToolTip(f"{port.name}: {port.dtype.value}" + ("（可选）" if port.optional else "")
                        + (f"\n{tr(port.description)}" if port.description else ""))

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))

    def hoverEnterEvent(self, event) -> None:
        self.setScale(1.45)
        self.setPen(QPen(QColor("#ffffff"), 1.5))

    def hoverLeaveEvent(self, event) -> None:
        self.setScale(1.0)
        self.setPen(QPen(QColor(C["canvas"]), 2))

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
        self._body_top = HEADER_H + 6
        self._h = self._body_top + rows * ROW_H + 6 + FOOT_H
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
        return QRectF(-PORT_R - 2, -4, NODE_W + 2 * PORT_R + 4, self._h + 10)

    def paint(self, painter, option, widget=None) -> None:
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        body = QRectF(0, 0, NODE_W, self._h)
        enabled = self.node.enabled
        # 阴影
        shadow = QPainterPath()
        shadow.addRoundedRect(body.translated(0, 3), RADIUS, RADIUS)
        painter.fillPath(shadow, QColor(0, 0, 0, 90))
        # 主体
        path = QPainterPath()
        path.addRoundedRect(body, RADIUS, RADIUS)
        painter.fillPath(path, QColor("#2a2f3a" if enabled else "#22262e"))
        # 标题栏（分类色渐变）
        hc = QColor(self.node.color)
        if not enabled:
            hc = hc.darker(220)
        grad = QLinearGradient(0, 0, NODE_W, 0)
        grad.setColorAt(0, hc)
        grad.setColorAt(1, hc.darker(135))
        header = QPainterPath()
        header.addRoundedRect(QRectF(0, 0, NODE_W, HEADER_H), RADIUS, RADIUS)
        header.addRect(QRectF(0, HEADER_H / 2, NODE_W, HEADER_H / 2))
        painter.fillPath(header.simplified(), QBrush(grad))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.setFont(QFont(painter.font().family(), 9, QFont.DemiBold))
        name = self.node.name if len(self.node.name) <= 18 else self.node.name[:17] + "…"
        painter.drawText(QRectF(10, 0, NODE_W - 20, HEADER_H), Qt.AlignVCenter | Qt.AlignLeft, name)
        if not enabled:
            painter.setFont(QFont(painter.font().family(), 7))
            painter.drawText(QRectF(10, 0, NODE_W - 20, HEADER_H), Qt.AlignVCenter | Qt.AlignRight, "已禁用")
        # 端口标签
        painter.setFont(QFont(painter.font().family(), 8))
        painter.setPen(QPen(QColor("#cfd4dc")))
        for i, p in enumerate(self.node.inputs):
            y = self._body_top + i * ROW_H
            painter.drawText(QRectF(12, y, NODE_W / 2 - 10, ROW_H), Qt.AlignVCenter | Qt.AlignLeft, p.label)
        for i, p in enumerate(self.node.outputs):
            y = self._body_top + i * ROW_H
            painter.drawText(QRectF(NODE_W / 2, y, NODE_W / 2 - 12, ROW_H), Qt.AlignVCenter | Qt.AlignRight, p.label)
        # 底部状态胶囊 + 耗时
        st = self.node.status
        foot_y = self._h - FOOT_H
        painter.setPen(QPen(QColor(C["border"])))
        painter.drawLine(QPointF(10, foot_y), QPointF(NODE_W - 10, foot_y))
        color = QColor(STATUS_COLORS.get(st, "#6b7280"))
        pill = QRectF(10, foot_y + 5, 46, 13)
        pill_path = QPainterPath()
        pill_path.addRoundedRect(pill, 6.5, 6.5)
        painter.fillPath(pill_path, color if st in (NodeStatus.OK, NodeStatus.NG, NodeStatus.ERROR, NodeStatus.RUNNING) else QColor("#394050"))
        painter.setPen(QPen(QColor("#111") if st in (NodeStatus.OK, NodeStatus.ERROR) else QColor("#f5f5f5")))
        painter.setFont(QFont(painter.font().family(), 7, QFont.Bold))
        painter.drawText(pill, Qt.AlignCenter, _STATUS_TEXT.get(st, st.value))
        painter.setPen(QPen(QColor(C["muted"])))
        painter.setFont(QFont(painter.font().family(), 7))
        if st in (NodeStatus.OK, NodeStatus.NG, NodeStatus.ERROR):
            foot = f"{self.node.last_time_ms:.1f} ms"
        elif st == NodeStatus.SKIPPED and self.node.last_error:
            foot = self.node.last_error if len(self.node.last_error) <= 16 else self.node.last_error[:15] + "…"
        else:
            foot = ""
        painter.drawText(QRectF(60, foot_y + 3, NODE_W - 70, FOOT_H - 4), Qt.AlignVCenter | Qt.AlignRight, foot)
        # 边框 / 选中
        if self.isSelected():
            painter.setPen(QPen(QColor(C["sel"]), 2))
        elif st == NodeStatus.ERROR:
            painter.setPen(QPen(QColor(C["warn"]), 1.2))
        elif st == NodeStatus.NG:
            painter.setPen(QPen(QColor(C["ng"]), 1.2))
        else:
            painter.setPen(QPen(QColor("#3a4150"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

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
        color = QColor(DTYPE_COLORS.get(src.port.dtype, "#d7dbe2"))
        self._pen = QPen(color, 2.2)
        self._pen.setCapStyle(Qt.RoundCap)
        self._pen_sel = QPen(QColor(C["sel"]), 3.2)
        self._glow = QPen(QColor(color.red(), color.green(), color.blue(), 60), 7)
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
        painter.setPen(self._glow)
        painter.drawPath(self.path())
        painter.setPen(self._pen_sel if self.isSelected() else self._pen)
        painter.drawPath(self.path())


def bezier(a: QPointF, b: QPointF):
    path = QPainterPath(a)
    dx = max(50.0, abs(b.x() - a.x()) * 0.5)
    path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
    return path

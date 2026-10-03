"""QGraphicsItems for nodes, ports and links."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem

from ...core.graph import Link
from ...core.node import Node, NodeStatus, Port
from ..theme import DTYPE_COLORS, STATUS_COLORS

NODE_W = 170
HEADER_H = 26
ROW_H = 18
PORT_R = 5


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", port: Port, is_output: bool) -> None:
        super().__init__(-PORT_R, -PORT_R, 2 * PORT_R, 2 * PORT_R, node_item)
        self.node_item = node_item
        self.port = port
        self.is_output = is_output
        color = QColor(DTYPE_COLORS.get(port.dtype, "#e0e0e0"))
        self.setBrush(QBrush(color))
        self.setPen(QPen(color.darker(150), 1))
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.setToolTip(f"{port.name}: {port.dtype.value}" + (" (optional)" if port.optional else "")
                        + (f"\n{port.description}" if port.description else ""))

    def scene_center(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))

    def hoverEnterEvent(self, event) -> None:
        self.setScale(1.4)

    def hoverLeaveEvent(self, event) -> None:
        self.setScale(1.0)

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
        self._h = HEADER_H + 8 + rows * ROW_H + 14
        for i, p in enumerate(node.inputs):
            it = PortItem(self, p, False)
            it.setPos(0, HEADER_H + 8 + i * ROW_H + ROW_H / 2)
            self.inputs[p.name] = it
        for i, p in enumerate(node.outputs):
            it = PortItem(self, p, True)
            it.setPos(NODE_W, HEADER_H + 8 + i * ROW_H + ROW_H / 2)
            self.outputs[p.name] = it
        self.setPos(node.position[0], node.position[1])
        self.setToolTip(f"{node.label} [{node.type_id}]\n{node.description}")

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_R, -2, NODE_W + 2 * PORT_R, self._h + 4)

    def paint(self, painter, option, widget=None) -> None:
        body = QRectF(0, 0, NODE_W, self._h)
        path = QPainterPath()
        path.addRoundedRect(body, 6, 6)
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        painter.fillPath(path, QBrush(QColor("#3a3a3a" if self.node.enabled else "#2d2d2d")))
        # header
        header = QPainterPath()
        header.addRoundedRect(QRectF(0, 0, NODE_W, HEADER_H), 6, 6)
        header.addRect(QRectF(0, HEADER_H / 2, NODE_W, HEADER_H / 2))
        hc = QColor(self.node.color)
        if not self.node.enabled:
            hc = hc.darker(200)
        painter.fillPath(header.simplified(), QBrush(hc))
        painter.setPen(QPen(QColor("#f0f0f0")))
        painter.setFont(QFont("Sans", 9, QFont.Bold))
        name = self.node.name if len(self.node.name) <= 20 else self.node.name[:19] + "…"
        painter.drawText(QRectF(8, 0, NODE_W - 40, HEADER_H), Qt.AlignVCenter | Qt.AlignLeft, name)
        # status dot + time
        st = self.node.status
        painter.setBrush(QBrush(QColor(STATUS_COLORS.get(st, "#777"))))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(NODE_W - 14, HEADER_H / 2), 5, 5)
        painter.setPen(QPen(QColor("#bbbbbb")))
        painter.setFont(QFont("Sans", 7))
        foot = f"{self.node.last_time_ms:.1f} ms" if st in (NodeStatus.OK, NodeStatus.NG, NodeStatus.ERROR) else st.value
        if st in (NodeStatus.ERROR, NodeStatus.SKIPPED) and self.node.last_error:
            foot = (self.node.last_error[:26] + "…") if len(self.node.last_error) > 27 else self.node.last_error
        painter.drawText(QRectF(6, self._h - 14, NODE_W - 12, 13), Qt.AlignVCenter | Qt.AlignRight, foot)
        # port labels
        painter.setFont(QFont("Sans", 8))
        painter.setPen(QPen(QColor("#dddddd")))
        for i, p in enumerate(self.node.inputs):
            y = HEADER_H + 8 + i * ROW_H
            painter.drawText(QRectF(10, y, NODE_W / 2 - 10, ROW_H), Qt.AlignVCenter | Qt.AlignLeft, p.label)
        for i, p in enumerate(self.node.outputs):
            y = HEADER_H + 8 + i * ROW_H
            painter.drawText(QRectF(NODE_W / 2, y, NODE_W / 2 - 10, ROW_H), Qt.AlignVCenter | Qt.AlignRight, p.label)
        # selection / border
        pen = QPen(QColor("#ffd54f") if self.isSelected() else QColor("#222222"), 2 if self.isSelected() else 1)
        painter.setPen(pen)
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
        color = QColor(DTYPE_COLORS.get(src.port.dtype, "#e0e0e0"))
        self._pen = QPen(color, 2)
        self._pen_sel = QPen(QColor("#ffd54f"), 3)
        self.setPen(self._pen)
        self.update_path()

    def update_path(self) -> None:
        a, b = self.src.scene_center(), self.dst.scene_center()
        self.setPath(bezier(a, b))

    def paint(self, painter, option, widget=None) -> None:
        self.setPen(self._pen_sel if self.isSelected() else self._pen)
        option.state &= ~option.state.__class__.State_Selected  # no default dashed selection box
        super().paint(painter, option, widget)


def bezier(a: QPointF, b: QPointF) -> QPainterPath:
    path = QPainterPath(a)
    dx = max(40.0, abs(b.x() - a.x()) * 0.5)
    path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
    return path

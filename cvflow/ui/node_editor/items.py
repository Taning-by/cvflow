"""节点、端口、连线的图形项。

流程图形语言（产品辨识度的主要来源）：
* 节点＝浅色卡片，左侧 4px 分类色书脊，标题右侧是状态胶囊（字形＋文字＋颜色，三重编码）。
  只有 NG／执行失败／运行中用实心胶囊，成功与待运行用浅底，一屏里跳出来的永远是要处理的那几个。
* 端口统一为圆点，位置与原来一致；必填端口实心，可选端口空心，不靠颜色区分。
* 底栏左边是参数摘要（扫一眼就知道这个节点在做什么），右边是耗时或错误。
* 选中＝细品牌色边框 + 浅底 + 四角定位标记，呼应图像上的 ROI。
* 可单独触发的源节点（相机、图像文件、图像文件夹）在标题栏里多一个圆形触发图标：
  空闲时是"暂停"两道竖杠，正在触发时是实心三角。点它就只触发这一路。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetricsF, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem

from ...core.graph import Link
from ...core.node import Node, NodeStatus, Port
from ..i18n import tr
from ..theme import C, DTYPE_COLORS, STATUS_COLORS, STATUS_GLYPH, STATUS_TEXT, tabular, ui_font

NODE_W = 184          # 与既有方案文件里的节点间距保持一致，不改变旧流程的疏密
HEADER_H = 28
ROW_H = 22
PORT_R = 5.0
FOOT_H = 22
RADIUS = 6
SPINE_W = 4.0
CORNER = 7.0          # 选中时四角定位标记的长度
TRIG_D = 16.0         # 触发图标的直径（标题栏里，状态胶囊左边）

# 实心胶囊只留给需要被一眼找到的状态：不合格、执行失败、正在运行。
# 成功/跳过/待运行用浅底，这样一屏节点里跳出来的永远是要处理的那几个。
_SOLID_STATES = (NodeStatus.NG, NodeStatus.ERROR, NodeStatus.RUNNING)
_SOFT_BG = {NodeStatus.OK: C["ok_bg"], NodeStatus.SKIPPED: C["panel2"], NodeStatus.IDLE: C["panel2"]}


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
        self._tip = f"{tr(node.label)} [{node.type_id}]\n{tr(node.description)}"
        self.setToolTip(self._tip)
        if node.triggerable:
            self.setAcceptHoverEvents(True)       # 触发图标要有悬停反馈
        self._trig_hover = False

    def boundingRect(self) -> QRectF:
        pad = PORT_R + CORNER
        return QRectF(-pad, -pad, NODE_W + 2 * pad, self._h + 2 * pad)

    def shape(self) -> QPainterPath:
        """命中区域只有卡片本身：外扩的四角标记不参与点击，免得盖住邻近节点的端口。"""
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, NODE_W, self._h), RADIUS, RADIUS)
        return path

    # ---- 参数摘要：底栏里一行能扫到的关键设置 ----
    def _summary(self) -> tuple[str, Qt.TextElideMode]:
        """返回 (摘要文字, 省略方式)。只有一个文件/目录时从中间省略，保住名字两头和扩展名。"""
        bits: list[str] = []
        kinds: list[str] = []
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
                bits.append(str(v).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
            else:
                bits.append(f"{tr(p.label)} {v}")
            kinds.append(p.kind)
            if len(bits) == 2:
                break
        mode = Qt.ElideMiddle if (len(bits) == 1 and kinds[0] in ("file", "dir")) else Qt.ElideRight
        return " · ".join(bits), mode

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

        # 左侧分类色书脊：用卡片轮廓裁剪后填一条直边，
        # 不能用两个圆角矩形求交——交集会在上下圆角处留下细碎尖角。
        cat = QColor(node.color)
        if not enabled:
            cat.setAlpha(90)
        painter.save()
        painter.setClipPath(path)
        painter.fillRect(QRectF(0, 0, SPINE_W, self._h), cat)
        painter.restore()
        # 标题与底栏的分隔线：与书脊对齐，给卡片一个清楚的三段结构
        painter.setPen(QPen(QColor(C["border"])))
        painter.drawLine(QPointF(SPINE_W, HEADER_H), QPointF(NODE_W, HEADER_H))

        # 标题文字
        painter.setPen(QPen(QColor(C["text"] if enabled else C["faint"])))
        f = ui_font(13, True)
        painter.setFont(f)
        pill_w = self._pill_width(st)
        trig = self.trigger_rect()
        reserved = 18 + pill_w + (TRIG_D + 6 if trig is not None else 0)
        name_rect = QRectF(SPINE_W + 8, 0, NODE_W - SPINE_W - reserved, HEADER_H)
        name = QFontMetricsF(f).elidedText(node.name, Qt.ElideMiddle, name_rect.width())
        painter.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, name)
        self._draw_pill(painter, QRectF(NODE_W - 8 - pill_w, (HEADER_H - 15) / 2, pill_w, 15), st, status_col, enabled)
        if trig is not None:
            self._draw_trigger(painter, trig, enabled)

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
        painter.drawLine(QPointF(SPINE_W, foot_y), QPointF(NODE_W, foot_y))
        painter.setFont(tabular(ui_font(11)))     # 等宽数字：连续运行时耗时不会左右跳动
        if not enabled:
            right = "已禁用"
        elif st in (NodeStatus.OK, NodeStatus.NG, NodeStatus.ERROR):
            right = f"{node.last_time_ms:.1f} ms"
        elif st == NodeStatus.SKIPPED and node.last_error:
            right = node.last_error
        else:
            right = ""
        metrics = QFontMetricsF(tabular(ui_font(11)))
        right = metrics.elidedText(right, Qt.ElideRight, NODE_W * 0.45)
        rw = metrics.horizontalAdvance(right) + 10
        painter.setPen(QPen(QColor(C["ng"] if (st == NodeStatus.ERROR and enabled) else
                                   (C["muted"] if enabled else C["faint"]))))
        painter.drawText(QRectF(NODE_W - 8 - rw, foot_y, rw, FOOT_H), Qt.AlignVCenter | Qt.AlignRight, right)
        painter.setFont(ui_font(11))
        metrics = QFontMetricsF(ui_font(11))
        summary_text, summary_mode = self._summary()
        summary = metrics.elidedText(summary_text, summary_mode, NODE_W - SPINE_W - 18 - rw)
        painter.setPen(QPen(QColor(C["faint"])))
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

    # ---- 触发图标：只在可单独触发的源节点上 ----
    def trigger_rect(self) -> QRectF | None:
        """触发图标在节点坐标系里的圆形区域；不可触发的节点返回 None。"""
        if not self.node.triggerable:
            return None
        pill_w = self._pill_width(self.node.status)
        x = NODE_W - 8 - pill_w - 6 - TRIG_D
        return QRectF(x, (HEADER_H - TRIG_D) / 2, TRIG_D, TRIG_D)

    def is_triggering(self) -> bool:
        """这一路是不是正有一次触发没跑完（问场景，场景问 FlowRunner）。"""
        sc = self.scene()
        probe = getattr(sc, "branch_busy", None)
        try:
            return bool(probe(self.node.id)) if probe is not None else False
        except Exception:                          # pragma: no cover - 界面不该因为探测失败而崩
            return False

    def _draw_trigger(self, painter, rect: QRectF, enabled: bool) -> None:
        """画触发图标：空闲＝暂停两道竖杠，正在触发＝实心三角。

        两个状态用**形状**区分而不只是颜色，和节点状态胶囊一样的思路：一屏里扫一眼就知道
        哪一路正在跑。悬停时底色加深，提示它可以点。
        """
        busy = enabled and self.is_triggering()
        if not enabled:
            fill, fg = QColor(C["panel2"]), QColor(C["faint"])
        elif busy:
            fill, fg = QColor(STATUS_COLORS.get(NodeStatus.RUNNING, C["accent"])), QColor("#FFFFFF")
        elif self._trig_hover:
            fill, fg = QColor(C["accent_bg"]), QColor(C["accent"])
        else:
            fill, fg = QColor(C["panel2"]), QColor(C["muted"])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(rect)
        if enabled and not busy:
            painter.setPen(QPen(QColor(C["border2"]), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(fg))
        c = rect.center()
        if busy:                                   # ▶ 正在触发
            tri = QPainterPath()
            tri.moveTo(c.x() - 2.6, c.y() - 4.0)
            tri.lineTo(c.x() + 4.0, c.y())
            tri.lineTo(c.x() - 2.6, c.y() + 4.0)
            tri.closeSubpath()
            painter.drawPath(tri)
        else:                                      # ⏸ 空闲（暂停）
            painter.drawRoundedRect(QRectF(c.x() - 3.4, c.y() - 4.0, 2.4, 8.0), 1.0, 1.0)
            painter.drawRoundedRect(QRectF(c.x() + 1.0, c.y() - 4.0, 2.4, 8.0), 1.0, 1.0)

    def _pill_width(self, st: NodeStatus) -> float:
        text = f"{STATUS_GLYPH.get(st, '')} {STATUS_TEXT.get(st, st.value)}"
        return QFontMetricsF(ui_font(11, True)).horizontalAdvance(text) + 14

    def _draw_pill(self, painter, rect: QRectF, st: NodeStatus, col: QColor, enabled: bool) -> None:
        """状态胶囊：字形 + 文字 + 颜色三重编码，成功 / 不合格 / 失败不会混淆。"""
        pill = QPainterPath()
        pill.addRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        if not enabled:
            fill, fg, border = QColor(C["panel2"]), QColor(C["faint"]), None
        elif st in _SOLID_STATES:
            fill, fg, border = QColor(col), QColor("#FFFFFF"), None
        else:
            fill, fg, border = QColor(_SOFT_BG.get(st, C["panel2"])), QColor(col), QColor(col)
        painter.fillPath(pill, fill)
        if border is not None:
            painter.setPen(QPen(border, 1))
            painter.drawPath(pill)
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

    def mousePressEvent(self, event) -> None:
        """点在触发图标上就只触发这一路，并且不要顺手把节点拖走。"""
        rect = self.trigger_rect()
        if rect is not None and event.button() == Qt.LeftButton and rect.contains(event.pos()):
            if self.node.enabled:
                self.scene().trigger_requested.emit(self.node.id)
                self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        rect = self.trigger_rect()
        hover = rect is not None and rect.contains(event.pos())
        if hover != self._trig_hover:
            self._trig_hover = hover
            self.setCursor(Qt.PointingHandCursor if hover else Qt.ArrowCursor)
            self.setToolTip(tr("Trigger this branch only: grab one frame and run down to the "
                               "deep-learning node, where it waits for the other branches.") if hover else self._tip)
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._trig_hover:
            self._trig_hover = False
            self.setCursor(Qt.ArrowCursor)
            self.setToolTip(self._tip)
            self.update()
        super().hoverLeaveEvent(event)

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

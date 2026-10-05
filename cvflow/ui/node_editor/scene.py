"""Graphics scene bound to a Graph: keeps items and the model in sync."""
from __future__ import annotations

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from ..theme import C
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsScene

from ...core.graph import Graph, GraphError, Link
from ...core.node import Node
from ...core.registry import registry
from ..i18n import tr
from .items import LinkItem, NodeItem, PortItem, bezier


class NodeScene(QGraphicsScene):
    node_selected = Signal(object)        # node id or None
    node_enabled_changed = Signal(str, bool)
    node_double_clicked = Signal(str)
    graph_changed = Signal()
    message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.graph: Graph = Graph()
        self.node_items: dict[str, NodeItem] = {}
        self.link_items: dict[str, LinkItem] = {}
        self._temp: QGraphicsPathItem | None = None
        self._temp_src: PortItem | None = None
        self._locked = False
        self.setSceneRect(-2000, -2000, 6000, 6000)
        self.setBackgroundBrush(QColor(C["canvas"]))
        self.selectionChanged.connect(self._on_selection)

    def drawBackground(self, painter, rect) -> None:
        super().drawBackground(painter, rect)
        painter.setRenderHint(painter.RenderHint.Antialiasing, False)
        small, big = 24, 120
        left, top = int(rect.left()) - int(rect.left()) % small, int(rect.top()) - int(rect.top()) % small
        pen_s = QPen(QColor("#E9EDF1"), 1)       # 细格：提供对位参考
        pen_b = QPen(QColor("#DCE3E8"), 1)       # 粗格：每 120 单位一条
        x = left
        while x < rect.right():
            painter.setPen(pen_b if x % big == 0 else pen_s)
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            x += small
        y = top
        while y < rect.bottom():
            painter.setPen(pen_b if y % big == 0 else pen_s)
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            y += small

    # ---- model binding ----
    def set_graph(self, graph: Graph) -> None:
        self.blockSignals(True)
        self.clear()
        self.node_items.clear()
        self.link_items.clear()
        self._temp = None
        self.graph = graph
        for node in graph.nodes.values():
            self._add_item(node)
        for link in graph.links:
            self._add_link_item(link)
        self.blockSignals(False)
        self.node_selected.emit(None)

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        for it in self.node_items.values():
            it.setFlag(NodeItem.ItemIsMovable, not locked)

    def _add_item(self, node: Node) -> NodeItem:
        it = NodeItem(node)
        it.setFlag(NodeItem.ItemIsMovable, not self._locked)
        self.addItem(it)
        self.node_items[node.id] = it
        return it

    def _add_link_item(self, link: Link) -> None:
        s = self.node_items[link.src_node].outputs[link.src_port]
        d = self.node_items[link.dst_node].inputs[link.dst_port]
        li = LinkItem(link, s, d)
        self.addItem(li)
        self.link_items[link.id] = li

    # ---- editing ----
    def add_node(self, type_id: str, pos: QPointF) -> Node | None:
        if self._locked:
            self.message.emit(tr("Stop run mode to edit the flow"))
            return None
        node = registry.create(type_id)
        node.name = self.graph.unique_name(tr(node.label))
        node.position = [pos.x(), pos.y()]
        self.graph.add_node(node)
        it = self._add_item(node)
        self.clearSelection()
        it.setSelected(True)
        self.graph_changed.emit()
        return node

    def remove_selected(self) -> None:
        if self._locked:
            self.message.emit(tr("Stop run mode to edit the flow"))
            return
        changed = False
        for it in list(self.selectedItems()):
            if isinstance(it, LinkItem):
                self.graph.remove_link(it.link)
                self._drop_link_item(it.link.id)
                changed = True
        for it in list(self.selectedItems()):
            if isinstance(it, NodeItem):
                nid = it.node.id
                for lid, li in list(self.link_items.items()):
                    if nid in (li.link.src_node, li.link.dst_node):
                        self._drop_link_item(lid)
                self.graph.remove_node(nid)
                self.removeItem(it)
                self.node_items.pop(nid, None)
                changed = True
        if changed:
            self.graph_changed.emit()
            self.node_selected.emit(None)

    def _drop_link_item(self, lid: str) -> None:
        li = self.link_items.pop(lid, None)
        if li is not None:
            self.removeItem(li)

    def set_node_enabled(self, node_id: str, enabled: bool) -> None:
        node = self.graph.nodes[node_id]
        node.set_enabled(enabled)          # 统一入口：禁用的节点会顺带放掉模型权重等资源
        self.node_items[node_id].update()
        self.node_enabled_changed.emit(node_id, bool(enabled))
        self.graph_changed.emit()

    def rename_node(self, node_id: str, name: str) -> None:
        node = self.graph.nodes[node_id]
        if name and name != node.name:
            node.name = self.graph.unique_name(name)
            self.node_items[node_id].update()
            self.graph_changed.emit()

    def rebuild_node(self, node_id: str) -> int:
        """端口数量变化后重建节点图元，断开指向已消失端口的连线，返回断开的条数。"""
        node = self.graph.nodes[node_id]
        valid_in = {p.name for p in node.inputs}
        valid_out = {p.name for p in node.outputs}
        stale = [l for l in self.graph.links
                 if (l.dst_node == node_id and l.dst_port not in valid_in)
                 or (l.src_node == node_id and l.src_port not in valid_out)]
        for l in stale:
            self.graph.remove_link(l)
            self._drop_link_item(l.id)
        it = self.node_items.pop(node_id)
        self.removeItem(it)
        self._add_item(node)
        for lid, li in list(self.link_items.items()):
            if node_id in (li.link.src_node, li.link.dst_node):
                self._drop_link_item(lid)
                self._add_link_item(li.link)
        if stale:
            self.graph_changed.emit()
        return len(stale)

    # ---- links by drag ----
    def start_link(self, port: PortItem, pos: QPointF) -> None:
        if self._locked:
            return
        self._temp_src = port
        self._temp = QGraphicsPathItem(bezier(port.scene_center(), pos))
        pen = QPen(QColor(C["accent"]), 2.0, Qt.DashLine)
        self._temp.setPen(pen)
        self._temp.setZValue(5)
        self.addItem(self._temp)

    def update_temp_link(self, pos: QPointF) -> None:
        if self._temp is not None and self._temp_src is not None:
            self._temp.setPath(bezier(self._temp_src.scene_center(), pos))

    def finish_link(self, pos: QPointF) -> None:
        src = self._temp_src
        if self._temp is not None:
            self.removeItem(self._temp)
        self._temp = None
        self._temp_src = None
        if src is None:
            return
        target = next((it for it in self.items(pos) if isinstance(it, PortItem) and it is not src), None)
        if target is None:
            # dropping an existing input link on empty space removes it
            if not src.is_output:
                existing = [l for l in self.graph.links if l.dst_node == src.node_item.node.id and l.dst_port == src.port.name]
                for l in existing:
                    self.graph.remove_link(l)
                    self._drop_link_item(l.id)
                    self.graph_changed.emit()
            return
        if src.is_output == target.is_output:
            self.message.emit(tr("Connect an output to an input"))
            return
        out, inp = (src, target) if src.is_output else (target, src)
        try:
            # replace any existing link into this input
            for l in [l for l in self.graph.links if l.dst_node == inp.node_item.node.id and l.dst_port == inp.port.name]:
                self._drop_link_item(l.id)
            link = self.graph.add_link(out.node_item.node.id, out.port.name, inp.node_item.node.id, inp.port.name)
            self._add_link_item(link)
            self.graph_changed.emit()
        except GraphError as e:
            # restore dropped item(s) if add_link failed
            for l in self.graph.links:
                if l.id not in self.link_items:
                    self._add_link_item(l)
            self.message.emit(str(e))

    def update_links_for(self, node_id: str) -> None:
        for li in self.link_items.values():
            if node_id in (li.link.src_node, li.link.dst_node):
                li.update_path()

    # ---- status ----
    def refresh_status(self) -> None:
        for it in self.node_items.values():
            it.update()

    def selected_node_ids(self) -> list[str]:
        return [it.node.id for it in self.selectedItems() if isinstance(it, NodeItem)]

    def _on_selection(self) -> None:
        ids = self.selected_node_ids()
        self.node_selected.emit(ids[0] if ids else None)

    def select_node(self, node_id: str) -> None:
        self.clearSelection()
        it = self.node_items.get(node_id)
        if it is not None:
            it.setSelected(True)

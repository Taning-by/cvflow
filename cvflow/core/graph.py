"""Flow graph: nodes + typed links, with cycle detection and JSON persistence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .node import Node
from .types import types_compatible

if TYPE_CHECKING:  # pragma: no cover
    from .registry import NodeRegistry


class GraphError(Exception):
    pass


@dataclass(frozen=True)
class Link:
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    @property
    def id(self) -> str:
        return f"{self.src_node}.{self.src_port}->{self.dst_node}.{self.dst_port}"

    def to_dict(self) -> dict:
        return {"src": [self.src_node, self.src_port], "dst": [self.dst_node, self.dst_port]}

    @classmethod
    def from_dict(cls, d: dict) -> "Link":
        return cls(d["src"][0], d["src"][1], d["dst"][0], d["dst"][1])


class Graph:
    def __init__(self, name: str = "main") -> None:
        self.name = name
        self.description = ""
        self.nodes: dict[str, Node] = {}
        self.links: list[Link] = []

    # ---- nodes ----
    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise GraphError(f"节点 id 重复：{node.id}")
        self.nodes[node.id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.links = [l for l in self.links if node_id not in (l.src_node, l.dst_node)]

    def get_node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise GraphError(f"流程 {self.name!r} 中没有节点 {node_id!r}") from None

    def find_by_name(self, name: str) -> Node | None:
        return next((n for n in self.nodes.values() if n.name == name), None)

    def unique_name(self, base: str) -> str:
        names = {n.name for n in self.nodes.values()}
        if base not in names:
            return base
        i = 2
        while f"{base} {i}" in names:
            i += 1
        return f"{base} {i}"

    # ---- links ----
    def add_link(self, src_node: str, src_port: str, dst_node: str, dst_port: str) -> Link:
        if src_node == dst_node:
            raise GraphError("节点不能连接到自身")
        src = self.get_node(src_node)
        dst = self.get_node(dst_node)
        sp = src.get_output_port(src_port)
        dp = dst.get_input_port(dst_port)
        if sp is None:
            raise GraphError(f"{src.name!r} 没有输出端口 {src_port!r}")
        if dp is None:
            raise GraphError(f"{dst.name!r} 没有输入端口 {dst_port!r}")
        if not types_compatible(sp.dtype, dp.dtype):
            raise GraphError(f"类型不匹配：{src.name}.{src_port} ({sp.dtype.value}) -> "
                             f"{dst.name}.{dst_port} ({dp.dtype.value})")
        link = Link(src_node, src_port, dst_node, dst_port)
        # an input accepts a single link: replace any existing one
        old = [l for l in self.links if l.dst_node == dst_node and l.dst_port == dst_port]
        trial = [l for l in self.links if l not in old] + [link]
        if self._has_cycle(trial):
            raise GraphError("该连线会形成环路")
        self.links = trial
        return link

    def remove_link(self, link: Link) -> None:
        self.links = [l for l in self.links if l != link]

    def remove_link_to(self, dst_node: str, dst_port: str) -> None:
        self.links = [l for l in self.links if not (l.dst_node == dst_node and l.dst_port == dst_port)]

    def input_links(self, node_id: str) -> dict[str, Link]:
        return {l.dst_port: l for l in self.links if l.dst_node == node_id}

    def output_links(self, node_id: str, port: str | None = None) -> list[Link]:
        return [l for l in self.links if l.src_node == node_id and (port is None or l.src_port == port)]

    # ---- ordering ----
    def _has_cycle(self, links: list[Link]) -> bool:
        try:
            self._topo(links)
            return False
        except GraphError:
            return True

    def _topo(self, links: list[Link]) -> list[str]:
        indeg = {nid: 0 for nid in self.nodes}
        succ: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for l in links:
            if l.src_node in indeg and l.dst_node in indeg:
                indeg[l.dst_node] += 1
                succ[l.src_node].append(l.dst_node)
        order: list[str] = []
        ready = [nid for nid in self.nodes if indeg[nid] == 0]  # insertion order = deterministic
        while ready:
            # prefer non-output nodes so Save/Render/Send nodes see the final judgement and all overlays
            idx = next((i for i, n in enumerate(ready) if self.nodes[n].category != "Output"), 0)
            nid = ready.pop(idx)
            order.append(nid)
            for s in succ[nid]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
        if len(order) != len(self.nodes):
            raise GraphError("流程中存在环路")
        return order

    def topological_order(self) -> list[str]:
        return self._topo(self.links)

    def validate(self) -> list[str]:
        """Return human-readable warnings (unconnected required inputs, etc.)."""
        warnings = []
        for node in self.nodes.values():
            if not node.enabled:
                continue
            linked = self.input_links(node.id)
            for p in node.inputs:
                if not p.optional and p.name not in linked:
                    warnings.append(f"{node.name}：输入 '{p.name}' 未连接")
        try:
            self.topological_order()
        except GraphError as e:
            warnings.append(str(e))
        return warnings

    def clear(self) -> None:
        self.nodes.clear()
        self.links.clear()

    # ---- persistence ----
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "links": [l.to_dict() for l in self.links],
        }

    @classmethod
    def from_dict(cls, d: dict, registry: "NodeRegistry") -> "Graph":
        g = cls(d.get("name", "main"))
        g.description = d.get("description", "")
        for nd in d.get("nodes", []):
            g.add_node(registry.node_from_dict(nd))
        for ld in d.get("links", []):
            link = Link.from_dict(ld)
            g.add_link(link.src_node, link.src_port, link.dst_node, link.dst_port)
        return g

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, registry: "NodeRegistry") -> "Graph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), registry)

"""Synchronous DAG executor.

One ``Engine.run`` call executes a flow once for one trigger: nodes are visited
in topological order, each node's outputs are cached in the ``RunContext`` and
handed to its successors. Failures are isolated: a node that raises is marked
ERROR, its dependants are SKIPPED, and every other branch keeps running.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from . import events
from .events import EventBus
from .graph import Graph, GraphError
from .node import Node, NodeStatus
from .types import Overlay, to_jsonable
from .variables import GlobalVariables

log = logging.getLogger("cvflow.engine")


@dataclass
class NodeResult:
    node_id: str
    node_name: str
    status: NodeStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    time_ms: float = 0.0
    error: str = ""
    overlays: list[Overlay] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: int
    flow: str
    trigger: Any = None
    started: float = 0.0
    duration_ms: float = 0.0
    ok: bool = True                     # False if any node raised
    judgement: bool | None = None       # AND of all ctx.judge() calls; None if none made
    node_results: dict[str, NodeResult] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)   # values published via ctx.publish
    variables: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def status_text(self) -> str:
        if not self.ok:
            return "ERROR"
        if self.judgement is False:
            return "NG"
        return "OK"

    @property
    def passed(self) -> bool:
        return self.ok and self.judgement is not False

    def node(self, name_or_id: str) -> NodeResult | None:
        r = self.node_results.get(name_or_id)
        if r is not None:
            return r
        return next((r for r in self.node_results.values() if r.node_name == name_or_id), None)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "flow": self.flow, "status": self.status_text,
            "ok": self.ok, "judgement": self.judgement, "duration_ms": round(self.duration_ms, 3),
            "error": self.error, "outputs": to_jsonable(self.outputs),
            "nodes": {r.node_name: {"status": r.status.value, "time_ms": round(r.time_ms, 3),
                                    "error": r.error, "outputs": to_jsonable(r.outputs)}
                      for r in self.node_results.values()},
        }


class RunContext:
    """Per-run scratch space handed to every node's ``process``."""

    def __init__(self, run_id: int, graph: Graph, variables: GlobalVariables,
                 trigger: Any, bus: EventBus | None) -> None:
        self.run_id = run_id
        self.graph = graph
        self.variables = variables
        self.trigger = trigger
        self.bus = bus
        self.results: dict[str, NodeResult] = {}
        self.outputs: dict[str, Any] = {}
        self.judgement: bool | None = None
        self.judge_reasons: list[str] = []
        self._current: Node | None = None
        self._overlays: list[Overlay] = []
        self._overlay_group = ""
        self._cancel = threading.Event()

    # ---- used by nodes ----
    @property
    def node(self) -> Node | None:
        return self._current

    @property
    def payload(self) -> dict:
        return getattr(self.trigger, "payload", None) or {}

    def judge(self, ok: bool, reason: str = "") -> None:
        """Combine a pass/fail decision into the run's judgement (AND)."""
        self.judgement = bool(ok) if self.judgement is None else (self.judgement and bool(ok))
        if not ok and reason:
            self.judge_reasons.append(reason)

    def publish(self, name: str, value: Any) -> None:
        """Expose a named result for communication templates and the result panel."""
        self.outputs[name] = value

    def add_overlay(self, overlay: Overlay) -> None:
        if self._overlay_group and not overlay.group:
            overlay.group = self._overlay_group
        self._overlays.append(overlay)

    @contextlib.contextmanager
    def overlay_group(self, name: str):
        """在这个作用域里产生的叠加层都会标记为属于 ``name`` 这一路输入。"""
        prev = self._overlay_group
        self._overlay_group = name
        try:
            yield
        finally:
            self._overlay_group = prev

    def get_output(self, node_id_or_name: str, port: str) -> Any:
        r = self.results.get(node_id_or_name)
        if r is None:
            r = next((x for x in self.results.values() if x.node_name == node_id_or_name), None)
        return None if r is None else r.outputs.get(port)

    def log(self, message: str, level: str = "info") -> None:
        name = self._current.name if self._current else "flow"
        getattr(log, level, log.info)("[%s] %s", name, message)
        if self.bus is not None:
            self.bus.emit(events.LOG, level=level, message=f"[{name}] {message}")

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ---- used by the engine ----
    def _begin(self, node: Node) -> None:
        self._current = node
        self._overlays = []
        self._overlay_group = ""

    def _end(self) -> list[Overlay]:
        ov, self._overlays, self._current = self._overlays, [], None
        return ov


class Engine:
    def __init__(self, graph: Graph, variables: GlobalVariables | None = None,
                 bus: EventBus | None = None) -> None:
        self.graph = graph
        self.variables = variables if variables is not None else GlobalVariables(bus)
        self.bus = bus
        self.last_result: RunResult | None = None
        # called with the finished RunResult before RUN_FINISHED is emitted, so that
        # bookkeeping (run statistics) is consistent by the time subscribers react
        self.on_result: Callable[[RunResult], None] | None = None
        self._run_counter = 0
        self._lock = threading.Lock()

    # ---- lifecycle ----
    def setup_nodes(self) -> list[str]:
        """Call ``setup`` on nodes not yet set up. Returns error messages (never raises)."""
        errors = []
        for node in self.graph.nodes.values():
            if node._is_setup:
                continue
            try:
                node.setup()
                node._is_setup = True
            except Exception as e:
                msg = f"{node.name}：初始化失败：{type(e).__name__}: {e}"
                node.last_error = msg
                node.status = NodeStatus.ERROR
                errors.append(msg)
                log.error(msg)
        return errors

    def teardown_nodes(self) -> None:
        for node in self.graph.nodes.values():
            if node._is_setup:
                try:
                    node.teardown()
                except Exception:
                    log.exception("%s: teardown failed", node.name)
                node._is_setup = False

    # ---- execution ----
    def run(self, trigger: Any = None, run_id: int | None = None) -> RunResult:
        with self._lock:  # one run at a time per flow; nodes keep per-instance state
            return self._run(trigger, run_id)

    def _run(self, trigger: Any, run_id: int | None) -> RunResult:
        self._run_counter += 1
        rid = run_id if run_id is not None else self._run_counter
        started = time.time()
        t_run = time.perf_counter()
        result = RunResult(run_id=rid, flow=self.graph.name, trigger=trigger, started=started)
        ctx = RunContext(rid, self.graph, self.variables, trigger, self.bus)
        self._emit(events.RUN_STARTED, run_id=rid, flow=self.graph.name)

        try:
            order = self.graph.topological_order()
        except GraphError as e:
            result.ok = False
            result.error = str(e)
            result.duration_ms = (time.perf_counter() - t_run) * 1000
            self._finish(result)
            return result

        self.setup_nodes()

        for nid in order:
            node = self.graph.nodes[nid]
            nres = NodeResult(node_id=nid, node_name=node.name, status=NodeStatus.SKIPPED)

            if not node.enabled:
                nres.error = "已禁用"
            else:
                inputs, skip_reason = self._gather_inputs(node, ctx)
                if skip_reason:
                    nres.error = skip_reason
                else:
                    node.status = NodeStatus.RUNNING
                    ctx._begin(node)
                    t0 = time.perf_counter()
                    try:
                        out = node.process(ctx, inputs)
                        if out is None:
                            out = {}
                        if not isinstance(out, dict):
                            raise TypeError(f"process() 必须返回 dict，实际返回 {type(out).__name__}")
                        if "__skip__" in out:          # node asked to skip its dependants (e.g. Gate)
                            nres.status = NodeStatus.SKIPPED
                            nres.error = str(out.get("__skip__") or "已跳过")
                            out = {}
                        else:
                            nres.status = NodeStatus.OK
                            for port in node.outputs:          # 未返回的声明输出补 None，保证下游语义稳定
                                out.setdefault(port.name, None)
                        nres.outputs = out
                        if nres.status == NodeStatus.OK and node.is_judge and "ok" in out:
                            ok = bool(out["ok"])
                            ctx.judge(ok, out.get("reason", "") or f"{node.name} NG")
                            if not ok:
                                nres.status = NodeStatus.NG
                    except Exception as e:
                        nres.status = NodeStatus.ERROR
                        nres.error = f"{type(e).__name__}: {e}"
                        result.ok = False
                        log.error("[%s] %s", node.name, nres.error)
                        log.debug("%s", traceback.format_exc())
                    nres.time_ms = (time.perf_counter() - t0) * 1000
                    nres.overlays = ctx._end()

            node.status = nres.status
            node.last_error = nres.error if nres.status in (NodeStatus.ERROR, NodeStatus.SKIPPED) else ""
            node.last_time_ms = nres.time_ms
            ctx.results[nid] = nres
            self._emit(events.NODE_FINISHED, run_id=rid, flow=self.graph.name, node_id=nid,
                       status=nres.status, time_ms=nres.time_ms, error=nres.error)
            if ctx.cancelled:
                result.error = "已取消"
                break

        result.node_results = ctx.results
        result.outputs = ctx.outputs
        result.judgement = ctx.judgement
        if ctx.judge_reasons and not result.error:
            result.error = "; ".join(ctx.judge_reasons)
        result.variables = self.variables.snapshot()
        result.duration_ms = (time.perf_counter() - t_run) * 1000
        self._finish(result)
        return result

    def _gather_inputs(self, node: Node, ctx: RunContext) -> tuple[dict[str, Any], str]:
        inputs: dict[str, Any] = {}
        links = self.graph.input_links(node.id)
        for port in node.inputs:
            link = links.get(port.name)
            if link is None:
                if not port.optional:
                    return {}, f"输入 '{port.name}' 未连接"
                inputs[port.name] = None
                continue
            up = ctx.results.get(link.src_node)
            if up is None or up.status in (NodeStatus.ERROR, NodeStatus.SKIPPED):
                # a *linked* upstream that did not produce outputs always skips this node,
                # whether or not the port is optional (optional only means "may stay unlinked")
                up_name = self.graph.nodes[link.src_node].name if link.src_node in self.graph.nodes else link.src_node
                state = {"error": "出错", "skipped": "已跳过"}.get(up.status.value, up.status.value) if up else "无结果"
                return {}, f"上游 '{up_name}' {state}"
            inputs[port.name] = up.outputs.get(link.src_port)
        return inputs, ""

    def _finish(self, result: RunResult) -> None:
        self.last_result = result
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:
                log.exception("on_result 回调失败")
        self._emit(events.RUN_FINISHED, result=result)

    def _emit(self, event: str, **payload: Any) -> None:
        if self.bus is not None:
            self.bus.emit(event, **payload)

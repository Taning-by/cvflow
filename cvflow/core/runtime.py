"""Runtime: triggers, the per-flow worker (FlowRunner) and the Solution container.

Edit mode   -> ``FlowRunner.run_once`` executes synchronously in the caller's thread.
Run mode    -> ``FlowRunner.start`` launches a worker; ``trigger`` enqueues work coming
               from the UI, a PLC message, a timer or a camera callback.
"""
from __future__ import annotations

import enum
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import events, paths
from .engine import Engine, RunResult
from .events import EventBus
from .graph import Graph
from .registry import NodeRegistry, registry as default_registry
from .variables import GlobalVariables

log = logging.getLogger("cvflow.runtime")


class TriggerSource(str, enum.Enum):
    MANUAL = "manual"
    COMM = "comm"
    TIMER = "timer"
    CAMERA = "camera"
    CLI = "cli"
    API = "api"


@dataclass
class Trigger:
    source: TriggerSource = TriggerSource.MANUAL
    payload: dict[str, Any] = field(default_factory=dict)
    device: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"source": self.source.value, "device": self.device, "message": self.message,
                "payload": self.payload, "timestamp": self.timestamp}


@dataclass
class RunStats:
    count: int = 0
    ok: int = 0
    ng: int = 0
    error: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    dropped: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def add(self, r: RunResult) -> None:
        self.count += 1
        self.total_ms += r.duration_ms
        self.last_ms = r.duration_ms
        if not r.ok:
            self.error += 1
        elif r.judgement is False:
            self.ng += 1
        else:
            self.ok += 1

    def reset(self) -> None:
        self.__init__()

    def to_dict(self) -> dict:
        return {"count": self.count, "ok": self.ok, "ng": self.ng, "error": self.error,
                "avg_ms": round(self.avg_ms, 2), "last_ms": round(self.last_ms, 2), "dropped": self.dropped}


class _Job:
    __slots__ = ("trigger", "done", "result")

    def __init__(self, trigger: Trigger) -> None:
        self.trigger = trigger
        self.done = threading.Event()
        self.result: RunResult | None = None


class FlowRunner:
    """Owns one flow and executes it, either inline or on its worker thread."""

    def __init__(self, graph: Graph, variables: GlobalVariables | None = None,
                 bus: EventBus | None = None, max_queue: int = 64) -> None:
        self.graph = graph
        self.bus = bus if bus is not None else EventBus()
        self.variables = variables if variables is not None else GlobalVariables(self.bus)
        self.engine = Engine(graph, self.variables, self.bus)
        self.stats = RunStats()
        self.engine.on_result = self.stats.add   # counted before RUN_FINISHED reaches comm/UI
        self.max_queue = max_queue
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._timer_thread: threading.Thread | None = None
        self._timer_interval: float | None = None
        self._timer_stop = threading.Event()
        # 事件驱动：每一路一个队列 + 一个线程（见 trigger_source 的说明）
        self._branches: dict[str, queue.Queue[_Job]] = {}
        self._branch_threads: dict[str, threading.Thread] = {}
        self._branch_lock = threading.Lock()
        self._branch_stop = threading.Event()
        self._branch_busy: dict[str, int] = {}     # 每一路还有几次触发没跑完（界面上的触发图标据此显示）

    @property
    def name(self) -> str:
        return self.graph.name

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_result(self) -> RunResult | None:
        return self.engine.last_result

    # ---- edit-mode execution ----
    def run_once(self, trigger: Trigger | None = None) -> RunResult:
        """Execute synchronously (edit mode, CLI, tests)."""
        trig = trigger or Trigger(TriggerSource.MANUAL)
        return self.engine.run(trig)

    # ---- run mode ----
    def start(self) -> list[str]:
        if self.running:
            return []
        errors = self.engine.setup_nodes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name=f"flow-{self.name}", daemon=True)
        self._thread.start()
        self.bus.emit(events.RUNNER_STATE, flow=self.name, running=True)
        if self._timer_interval:
            self._start_timer()
        return errors

    def stop(self, timeout: float = 5.0, teardown: bool = True) -> None:
        self._stop_timer()
        if self._thread is not None:
            self._stop.set()
        if self._branch_threads:
            self._branch_stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        with self._branch_lock:
            threads, self._branch_threads, self._branches = self._branch_threads, {}, {}
        for t in threads.values():
            t.join(timeout)
        # fail any jobs still queued
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            job.done.set()
        if teardown:
            self.engine.teardown_nodes()
        self.bus.emit(events.RUNNER_STATE, flow=self.name, running=False)

    def trigger(self, trigger: Trigger | None = None, wait: bool = False,
                timeout: float | None = None) -> RunResult | None:
        """Enqueue a run (run mode). Returns the result only when ``wait`` is True."""
        trig = trigger or Trigger(TriggerSource.MANUAL)
        if not self.running:
            log.warning("流程 %s：未启动，忽略触发", self.name)
            return None
        if self._queue.qsize() >= self.max_queue:
            self.stats.dropped += 1
            log.warning("流程 %s：队列已满 (%d)，丢弃触发", self.name, self.max_queue)
            return None
        job = _Job(trig)
        self._queue.put(job)
        if wait:
            job.done.wait(timeout)
            return job.result
        return None

    # ---- 事件驱动：单路触发 ----
    def async_gates(self) -> list[str]:
        """流程里所有"异步汇合"的节点 id（深度学习节点把 arrival 设成 async 时）。"""
        return [nid for nid, n in self.graph.nodes.items() if getattr(n, "is_async", False)]

    def branch_only(self, node_id: str) -> set[str] | None:
        """node_id 被单独触发时，这一次应该跑哪些节点。

        跑的是：本路的上游链 + 汇合点 + 汇合点的下游 + 其它与支路无关的节点（变量之类）；
        不跑的是**别的路**的上游链——那些路这一轮没被触发，不该去抓图。
        没有汇合点或该节点不属于任何一路时返回 None，表示整条流程照常跑。
        """
        for gate in self.async_gates():
            branches = self.graph.input_branches(gate)
            mine = next((p for p, nodes in branches.items() if node_id in nodes), None)
            if mine is None:
                continue
            others: set[str] = set()
            for port, nodes in branches.items():
                if port != mine:
                    others |= nodes
            return set(self.graph.nodes) - others
        return None

    def trigger_source(self, node_id: str, trigger: Trigger | None = None,
                       wait: bool = False, timeout: float | None = None) -> RunResult | None:
        """只触发某一路：从 node_id 这一路跑到汇合点，在那里等别的路凑批。

        每一路有自己的工作线程，原因有两个：
          * 汇合点会**阻塞**等待其它路，如果所有路挤在一个队列里，先到的那条占着唯一的线程、
            后到的那条永远进不来，窗口必然空等到期——死锁；
          * 一路相机的两次触发天然应该排队（不会出现同一路的两帧挤进同一批）。
        """
        trig = trigger or Trigger(TriggerSource.MANUAL)
        only = self.branch_only(node_id)
        if only is None:
            # 没有异步汇合点：退回"整条流程跑一次"。运行模式进队列，编辑模式直接同步跑
            if self.running:
                return self.trigger(trig, wait=wait, timeout=timeout)
            return self.run_once(trig)
        job = _Job(trig)
        q = self._branch_queue(node_id)
        with self._branch_lock:
            self._branch_busy[node_id] = self._branch_busy.get(node_id, 0) + 1
        q.put(job)
        if wait:
            job.done.wait(timeout)
            return job.result
        return None

    def branch_busy(self, node_id: str) -> bool:
        """这一路是不是正有一次触发没跑完。界面上的触发图标用它决定画"正在触发"。"""
        with self._branch_lock:
            return self._branch_busy.get(node_id, 0) > 0

    def _branch_queue(self, node_id: str) -> "queue.Queue[_Job]":
        """取（或建）某一路的工作队列与线程。"""
        with self._branch_lock:
            q = self._branches.get(node_id)
            if q is None:
                self._branch_stop.clear()      # 之前 stop() 过也能再用（编辑模式下反复点触发）
                q = self._branches[node_id] = queue.Queue()
                name = self.graph.nodes[node_id].name if node_id in self.graph.nodes else node_id
                t = threading.Thread(target=self._branch_worker, args=(node_id, q),
                                     name=f"branch-{self.name}-{name}", daemon=True)
                self._branch_threads[node_id] = t
                t.start()
            return q

    def _branch_worker(self, node_id: str, q: "queue.Queue[_Job]") -> None:
        while not self._branch_stop.is_set():
            try:
                job = q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                only = self.branch_only(node_id)
                job.result = self.engine.run(job.trigger, only=only)
            except Exception:                  # 引擎本身已经隔离了节点异常，这里是兜底
                log.exception("流程 %s：支路 %s 运行异常", self.name, node_id)
            finally:
                with self._branch_lock:
                    self._branch_busy[node_id] = max(0, self._branch_busy.get(node_id, 1) - 1)
                job.done.set()

    def set_continuous(self, interval_s: float | None) -> None:
        """Fire TIMER triggers every ``interval_s`` seconds while running (None/0 disables)."""
        self._timer_interval = interval_s if interval_s and interval_s > 0 else None
        self._stop_timer()
        if self._timer_interval and self.running:
            self._start_timer()

    @property
    def continuous(self) -> bool:
        return self._timer_thread is not None and self._timer_thread.is_alive()

    def pending(self) -> int:
        return self._queue.qsize()

    # ---- internals ----
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                job.result = self.engine.run(job.trigger)
            except Exception:  # the engine already isolates node errors; this is a safety net
                log.exception("流程 %s：引擎异常", self.name)
            finally:
                job.done.set()

    def _start_timer(self) -> None:
        self._timer_stop.clear()
        self._timer_thread = threading.Thread(target=self._timer_loop, name=f"timer-{self.name}", daemon=True)
        self._timer_thread.start()

    def _stop_timer(self) -> None:
        if self._timer_thread is not None:
            self._timer_stop.set()
            self._timer_thread.join(2.0)
            self._timer_thread = None

    def _timer_loop(self) -> None:
        interval = self._timer_interval or 1.0
        while not self._timer_stop.is_set() and not self._stop.is_set():
            if self._queue.empty():  # never let timer triggers pile up
                self.trigger(Trigger(TriggerSource.TIMER))
            self._timer_stop.wait(interval)


class Solution:
    """Everything a deployment needs: flows, variables, communication config, plugins."""

    FORMAT_VERSION = 1

    def __init__(self, name: str = "solution", bus: EventBus | None = None) -> None:
        self.name = name
        self.bus = bus if bus is not None else EventBus()
        self.variables = GlobalVariables(self.bus)
        self.flows: dict[str, Graph] = {}
        self.comm_config: dict[str, Any] = {"devices": [], "receive_rules": [], "send_rules": []}
        self.plugin_dirs: list[str] = []
        self.path: Path | None = None

    def add_flow(self, graph: Graph | None = None, name: str | None = None) -> Graph:
        g = graph or Graph(name or "main")
        if name:
            g.name = name
        base, i = g.name, 2
        while g.name in self.flows:
            g.name = f"{base}_{i}"
            i += 1
        self.flows[g.name] = g
        return g

    def remove_flow(self, name: str) -> None:
        self.flows.pop(name, None)

    def rename_flow(self, old: str, new: str) -> None:
        if new in self.flows or old not in self.flows:
            raise ValueError(f"无法把流程 {old!r} 重命名为 {new!r}")
        g = self.flows.pop(old)
        g.name = new
        self.flows[new] = g

    def to_dict(self) -> dict:
        return {
            "format": self.FORMAT_VERSION,
            "name": self.name,
            "plugin_dirs": list(self.plugin_dirs),
            "variables": self.variables.to_dict(),
            "comm": self.comm_config,
            "flows": [g.to_dict() for g in self.flows.values()],
        }

    @classmethod
    def from_dict(cls, d: dict, reg: NodeRegistry | None = None, bus: EventBus | None = None,
                  base_dir: Path | None = None) -> "Solution":
        reg = reg or default_registry
        reg.load_builtins()
        sol = cls(d.get("name", "solution"), bus)
        sol.plugin_dirs = list(d.get("plugin_dirs", []))
        for pdir in sol.plugin_dirs:
            p = Path(pdir)
            if not p.is_absolute() and base_dir is not None:
                p = base_dir / p
            reg.load_plugin_dir(p)
        sol.variables.from_dict(d.get("variables", {}))
        sol.comm_config = d.get("comm") or {"devices": [], "receive_rules": [], "send_rules": []}
        for gd in d.get("flows", []):
            sol.add_flow(Graph.from_dict(gd, reg))
        return sol

    def rebase(self, new_dir: Path) -> None:
        """Rewrite relative file/dir parameters so they stay valid when the solution moves to ``new_dir``."""
        old = paths.base_dir()
        new_dir = Path(new_dir).resolve()
        if old == new_dir:
            return
        for g in self.flows.values():
            for node in g.nodes.values():
                for p in node.params:
                    v = node.values.get(p.name)
                    if p.kind in ("file", "dir") and v and not Path(str(v)).is_absolute():
                        node.values[p.name] = paths.make_relative((old / str(v)).resolve(), new_dir)
        self.plugin_dirs = [paths.make_relative((old / d).resolve(), new_dir) if not Path(d).is_absolute() else d
                            for d in self.plugin_dirs]

    def save(self, path: str | Path) -> None:
        path = Path(path).resolve()
        self.rebase(path.parent)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self.path = path
        paths.set_base_dir(path.parent)

    @classmethod
    def load(cls, path: str | Path, reg: NodeRegistry | None = None, bus: EventBus | None = None) -> "Solution":
        path = Path(path).resolve()
        paths.set_base_dir(path.parent)
        sol = cls.from_dict(json.loads(path.read_text(encoding="utf-8")), reg, bus, base_dir=path.parent)
        sol.path = path
        return sol

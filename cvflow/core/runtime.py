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

from . import events
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
        self.max_queue = max_queue
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._timer_thread: threading.Thread | None = None
        self._timer_interval: float | None = None
        self._timer_stop = threading.Event()

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
        result = self.engine.run(trig)
        self.stats.add(result)
        return result

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
            self._thread.join(timeout)
            self._thread = None
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
            log.warning("flow %s: trigger ignored, runner not started", self.name)
            return None
        if self._queue.qsize() >= self.max_queue:
            self.stats.dropped += 1
            log.warning("flow %s: trigger dropped, queue full (%d)", self.name, self.max_queue)
            return None
        job = _Job(trig)
        self._queue.put(job)
        if wait:
            job.done.wait(timeout)
            return job.result
        return None

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
                self.stats.add(job.result)
            except Exception:  # the engine already isolates node errors; this is a safety net
                log.exception("flow %s: unexpected engine failure", self.name)
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
            raise ValueError(f"cannot rename flow {old!r} to {new!r}")
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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        self.path = path

    @classmethod
    def load(cls, path: str | Path, reg: NodeRegistry | None = None, bus: EventBus | None = None) -> "Solution":
        path = Path(path)
        sol = cls.from_dict(json.loads(path.read_text(encoding="utf-8")), reg, bus, base_dir=path.parent)
        sol.path = path
        return sol

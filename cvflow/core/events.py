"""Minimal thread-safe publish/subscribe bus.

Callbacks run synchronously in the emitting thread, so the UI must bridge
events onto the Qt thread (see ``cvflow.ui.bridge``). Event names are plain
strings; the common ones are listed as constants below.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

log = logging.getLogger("cvflow.events")

RUN_STARTED = "run_started"          # run_id, flow
NODE_FINISHED = "node_finished"      # run_id, flow, node_id, status, time_ms, error
RUN_FINISHED = "run_finished"        # result: RunResult
RUNNER_STATE = "runner_state"        # flow, running: bool
COMM_CONNECTED = "comm_connected"    # device
COMM_DISCONNECTED = "comm_disconnected"
COMM_RECEIVED = "comm_received"      # device, data (bytes|dict), text
COMM_SENT = "comm_sent"              # device, data, text
COMM_ERROR = "comm_error"            # device, error
VAR_CHANGED = "var_changed"          # name, value
LOG = "log"                          # level, message

Callback = Callable[..., None]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callback]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, event: str, callback: Callback) -> Callback:
        """Subscribe to ``event`` ("*" receives everything, with ``event`` kwarg prepended)."""
        with self._lock:
            self._subs[event].append(callback)
        return callback

    def unsubscribe(self, event: str, callback: Callback) -> None:
        with self._lock:
            try:
                self._subs[event].remove(callback)
            except ValueError:
                pass

    def emit(self, event: str, **payload: Any) -> None:
        with self._lock:
            targets = list(self._subs.get(event, ())) + [("*", cb) for cb in self._subs.get("*", ())]
        for t in targets:
            try:
                if isinstance(t, tuple):
                    t[1](event, **payload)
                else:
                    t(**payload)
            except Exception:  # a broken subscriber must not break the producer
                log.exception("event handler for %s failed", event)

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()

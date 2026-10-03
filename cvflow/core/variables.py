"""Global variables shared by all flows, communication rules and the UI."""
from __future__ import annotations

import threading
from typing import Any

from . import events
from .events import EventBus


class GlobalVariables:
    def __init__(self, bus: EventBus | None = None) -> None:
        self._vars: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.bus = bus

    def define(self, name: str, value: Any = None, dtype: str = "any", description: str = "") -> None:
        with self._lock:
            self._vars[name] = {"value": value, "dtype": dtype, "description": description}

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._vars

    def get(self, name: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._vars.get(name)
            return default if entry is None else entry["value"]

    def set(self, name: str, value: Any) -> None:
        with self._lock:
            if name in self._vars:
                self._vars[name]["value"] = value
            else:
                self._vars[name] = {"value": value, "dtype": "any", "description": ""}
        if self.bus is not None:
            self.bus.emit(events.VAR_CHANGED, name=name, value=value)

    def remove(self, name: str) -> None:
        with self._lock:
            self._vars.pop(name, None)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._vars)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {k: v["value"] for k, v in self._vars.items()}

    def entries(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._vars.items()}

    def to_dict(self) -> dict:
        return self.entries()

    def from_dict(self, d: dict) -> None:
        with self._lock:
            self._vars = {k: {"value": v.get("value"), "dtype": v.get("dtype", "any"),
                              "description": v.get("description", "")} for k, v in (d or {}).items()}

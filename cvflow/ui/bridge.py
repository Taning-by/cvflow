"""Forward EventBus events (emitted from worker/comm threads) onto the Qt GUI thread."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..core.events import EventBus


class EventBridge(QObject):
    event = Signal(str, object)  # (event name, payload dict) - queued across threads by Qt

    def __init__(self, bus: EventBus, parent=None) -> None:
        super().__init__(parent)
        self._bus = bus
        bus.subscribe("*", self._on_event)

    def _on_event(self, event: str, **payload) -> None:
        self.event.emit(event, payload)

    def detach(self) -> None:
        self._bus.unsubscribe("*", self._on_event)

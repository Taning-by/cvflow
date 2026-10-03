"""Camera abstraction. Nodes never talk to a vendor SDK directly; they ask a
``Camera`` for frames so the same flow runs against a folder of test images, a
USB webcam or a GigE Vision camera."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.types import Image


class CameraError(Exception):
    pass


class Camera(ABC):
    kind: str = ""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config: dict[str, Any] = dict(config or {})
        self.is_open = False
        self._frame_id = 0
        self._lock = threading.RLock()

    # ---- lifecycle ----
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def _grab_raw(self, timeout_s: float) -> np.ndarray | None:
        """Return one frame (HxW or HxWx3 uint8, BGR) or None on timeout."""

    def grab(self, timeout_s: float = 2.0) -> Image | None:
        with self._lock:
            if not self.is_open:
                self.open()
            data = self._grab_raw(timeout_s)
            if data is None:
                return None
            self._frame_id += 1
            if self.config.get("grayscale") and data.ndim == 3:
                import cv2
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            return Image(data=data, frame_id=self._frame_id, timestamp=time.time(), source=self.name)

    # ---- optional feature access (exposure, gain, trigger mode ...) ----
    def set_feature(self, name: str, value: Any) -> None:  # noqa: B027
        pass

    def get_feature(self, name: str) -> Any:
        return None

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "open": self.is_open, "frames": self._frame_id}

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

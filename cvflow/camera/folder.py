"""Simulated camera that replays image files from a directory."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base import Camera, CameraError

DEFAULT_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


class FolderCamera(Camera):
    kind = "folder"

    def __init__(self, name: str, config=None) -> None:
        super().__init__(name, config)
        self._files: list[Path] = []
        self._index = 0

    def open(self) -> None:
        d = Path(str(self.config.get("source", "")))
        if not d.is_dir():
            raise CameraError(f"folder camera {self.name!r}: {d} is not a directory")
        patterns = self.config.get("patterns") or DEFAULT_PATTERNS
        files: list[Path] = []
        for p in patterns:
            files.extend(d.glob(p))
        self._files = sorted(set(files))
        if not self._files:
            raise CameraError(f"folder camera {self.name!r}: no images in {d}")
        self._index = 0
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def _grab_raw(self, timeout_s: float) -> np.ndarray | None:
        if self._index >= len(self._files):
            if not self.config.get("loop", True):
                return None
            self._index = 0
        path = self._files[self._index]
        self._index += 1
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise CameraError(f"cannot read {path}")
        return img

    @property
    def current_file(self) -> str:
        i = max(0, min(self._index - 1, len(self._files) - 1))
        return str(self._files[i]) if self._files else ""

    def info(self):
        d = super().info()
        d.update({"files": len(self._files), "index": self._index, "current": self.current_file})
        return d

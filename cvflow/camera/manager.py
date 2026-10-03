"""Named camera instances shared between nodes and flows."""
from __future__ import annotations

import logging
import threading
from typing import Any

from .base import Camera, CameraError
from .folder import FolderCamera
from .genicam_cam import GenICamCamera
from .opencv_cam import OpenCVCamera

log = logging.getLogger("cvflow.camera")

CAMERA_KINDS: dict[str, type[Camera]] = {
    FolderCamera.kind: FolderCamera,
    OpenCVCamera.kind: OpenCVCamera,
    GenICamCamera.kind: GenICamCamera,
}


def create_camera(kind: str, name: str, config: dict[str, Any] | None = None) -> Camera:
    try:
        cls = CAMERA_KINDS[kind]
    except KeyError:
        raise CameraError(f"未知的相机类型 {kind!r}，可用：{sorted(CAMERA_KINDS)}") from None
    return cls(name, config)


class CameraManager:
    def __init__(self) -> None:
        self._cams: dict[str, Camera] = {}
        self._lock = threading.RLock()

    def get_or_create(self, name: str, kind: str, config: dict[str, Any] | None = None) -> Camera:
        with self._lock:
            cam = self._cams.get(name)
            if cam is not None and (cam.kind != kind or cam.config != dict(config or {})):
                cam.close()
                cam = None
            if cam is None:
                cam = create_camera(kind, name, config)
                self._cams[name] = cam
            return cam

    def get(self, name: str) -> Camera | None:
        return self._cams.get(name)

    def names(self) -> list[str]:
        return list(self._cams)

    def close(self, name: str) -> None:
        with self._lock:
            cam = self._cams.pop(name, None)
            if cam is not None:
                cam.close()

    def close_all(self) -> None:
        with self._lock:
            for cam in self._cams.values():
                try:
                    cam.close()
                except Exception:
                    log.exception("关闭相机 %s 失败", cam.name)
            self._cams.clear()


camera_manager = CameraManager()

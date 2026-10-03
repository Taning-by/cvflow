"""Named camera instances shared between nodes and flows."""
from __future__ import annotations

import logging
import threading
from typing import Any

from .base import Camera, CameraError
from .folder import FolderCamera
from .genicam_cam import GenICamCamera
from .hik_cam import HikCamera
from .opencv_cam import OpenCVCamera

log = logging.getLogger("cvflow.camera")

CAMERA_KINDS: dict[str, type[Camera]] = {
    FolderCamera.kind: FolderCamera,
    OpenCVCamera.kind: OpenCVCamera,
    GenICamCamera.kind: GenICamCamera,
    HikCamera.kind: HikCamera,
}


def enumerate_cameras(timeout: float = 1.0, cti: str | None = None, targets: list[str] | None = None) -> list:
    """像 VisionMaster 一样搜索相机：GVCP 广播发现 + 海康 MVS 枚举 + GenICam 枚举，按序列号/MAC 合并。"""
    from . import discovery
    found: dict[str, Any] = {}

    def add(dev, prefer: bool) -> None:
        key = dev.serial or dev.mac or dev.ip or f"{dev.source}:{dev.extra.get('index')}"
        if key in found and not prefer:
            old = found[key]
            for f in ("ip", "mac", "subnet", "gateway", "manufacturer", "model", "version", "user_name"):
                if not getattr(old, f) and getattr(dev, f):
                    setattr(old, f, getattr(dev, f))
            old.extra.setdefault("sources", []).append(dev.source)
            return
        dev.extra.setdefault("sources", [dev.source])
        found[key] = dev

    try:
        for d in discovery.discover(timeout, targets=targets):
            add(d, prefer=False)
    except Exception as e:
        log.warning("GVCP 搜索失败：%s", e)
    try:
        from .hik_cam import enumerate_hik, sdk_available
        if sdk_available():
            for d in enumerate_hik():
                add(d, prefer=True)
    except Exception as e:
        log.warning("MVS 枚举失败：%s", e)
    if cti:
        try:
            from .genicam_cam import enumerate_genicam
            for d in enumerate_genicam(cti):
                add(d, prefer=False)
        except Exception as e:
            log.warning("GenICam 枚举失败：%s", e)
    return sorted(found.values(), key=lambda d: (d.ip or "~", d.serial))


def test_camera(kind: str, source: str, config: dict[str, Any] | None = None) -> tuple[bool, str]:
    """连接测试。GigE 相机先走 GVCP 读寄存器，再按 kind 尝试用 SDK 打开。"""
    from . import discovery
    msgs = []
    if kind in ("hik", "genicam") and source and source.count(".") == 3:
        ok, msg = discovery.test_connection(source)
        msgs.append(msg)
        if not ok:
            return False, msg
    try:
        cam = create_camera(kind, "_test", {"source": source, **(config or {})})
    except CameraError as e:
        return False, str(e)
    ok, msg = cam.test_connection()
    msgs.append(msg)
    return ok, "；".join(msgs)


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

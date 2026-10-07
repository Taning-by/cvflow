"""Webcam / video file / RTSP stream through OpenCV VideoCapture."""
from __future__ import annotations

import cv2
import numpy as np

from .base import Camera, CameraError

#: 采集后端。Windows 上默认走 MSMF，它只认 UVC 设备；厂商注册的 DirectShow 滤镜
#: （比如迈德威视的 MindVision.ax）只有 dshow 后端看得见，所以要能手动指定。
BACKENDS = {
    "auto": 0,                                   # cv2.CAP_ANY
    "dshow": getattr(cv2, "CAP_DSHOW", 700),     # Windows DirectShow
    "msmf": getattr(cv2, "CAP_MSMF", 1400),      # Windows Media Foundation
    "v4l2": getattr(cv2, "CAP_V4L2", 200),       # Linux
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", 1800),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", 1900),  # 视频文件 / RTSP
}


def probe_indices(backend: str = "auto", count: int = 6) -> list[int]:
    """挨个试设备号，返回能打开的那些。

    DirectShow 这类后端没法列出设备名，只能一个个试——"搜索相机"对话框和排查时都用它。
    """
    api = BACKENDS.get(backend, 0)
    found = []
    # 打不开的设备号 OpenCV 会往 stderr 刷一大段警告，探测时这些全是噪音，先静音
    quiet = getattr(getattr(cv2, "utils", None), "logging", None)
    prev = None
    if quiet is not None:
        try:
            prev = quiet.getLogLevel()
            quiet.setLogLevel(quiet.LOG_LEVEL_SILENT)
        except Exception:                                # pragma: no cover - 取决于 OpenCV 版本
            prev = None
    try:
        for i in range(count):
            cap = cv2.VideoCapture(i, api) if api else cv2.VideoCapture(i)
            try:
                if cap.isOpened():
                    found.append(i)
            finally:
                cap.release()
    finally:
        if prev is not None:
            try:
                quiet.setLogLevel(prev)
            except Exception:                            # pragma: no cover
                pass
    return found


class OpenCVCamera(Camera):
    kind = "opencv"

    def __init__(self, name: str, config=None) -> None:
        super().__init__(name, config)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        src = self.config.get("source", 0)
        try:
            src = int(src)
        except (TypeError, ValueError):
            pass
        backend = str(self.config.get("backend", "auto") or "auto").lower()
        if backend not in BACKENDS:
            raise CameraError(f"未知的采集后端 {backend!r}，可用：{sorted(BACKENDS)}")
        api = BACKENDS[backend]
        cap = cv2.VideoCapture(src, api) if api else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            hint = ""
            if backend == "auto" and isinstance(src, int):
                hint = ("。设备号可能不对，或者这是厂商注册的 DirectShow 设备——"
                        "默认后端（Windows 上是 MSMF）只认 UVC，试试把采集后端设成 dshow")
            raise CameraError(f"OpenCV 相机 {self.name!r}：无法打开 {src!r}（后端 {backend}）{hint}")
        if self.config.get("width"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config["width"]))
        if self.config.get("height"):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config["height"]))
        self._cap = cap
        self.is_open = True
        self.apply_settings(self.config)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.is_open = False

    def _grab_raw(self, timeout_s: float) -> np.ndarray | None:
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            if self.config.get("loop") and self._cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        return frame

    def set_feature(self, name: str, value) -> None:
        if self._cap is None:
            return
        prop = {"exposure": cv2.CAP_PROP_EXPOSURE, "ExposureTime": cv2.CAP_PROP_EXPOSURE,
                "gain": cv2.CAP_PROP_GAIN, "Gain": cv2.CAP_PROP_GAIN,
                "fps": cv2.CAP_PROP_FPS, "brightness": cv2.CAP_PROP_BRIGHTNESS}.get(name)
        if prop is not None:
            self._cap.set(prop, float(value))

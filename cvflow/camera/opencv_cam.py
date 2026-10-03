"""Webcam / video file / RTSP stream through OpenCV VideoCapture."""
from __future__ import annotations

import cv2
import numpy as np

from .base import Camera, CameraError


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
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise CameraError(f"OpenCV 相机 {self.name!r}：无法打开 {src!r}")
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

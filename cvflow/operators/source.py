"""Image sources: files, folders, cameras, constants and trigger data."""
from __future__ import annotations

import os
from pathlib import Path

import cv2

from ..camera import camera_manager, CAMERA_KINDS
from ..core import paths
from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType, Image
from ._util import odd  # noqa: F401  (re-export convenience)

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@register
class ImageFile(Node):
    type_id = "source.image_file"
    category = "Source"
    label = "Image File"
    description = "Load a single image from disk (cached until the file changes)."
    color = "#2e7d32"
    outputs = [Port("image", DataType.IMAGE), Port("path", DataType.STRING)]
    params = [Param("path", "", "file", label="Path", filter=_IMG_FILTER),
              Param("grayscale", False, "bool")]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._cache: tuple[str, float, bool, Image] | None = None

    def process(self, ctx, inputs):
        path = paths.resolve(self.get("path"))
        if not path:
            raise NodeError("no image path set")
        if not os.path.isfile(path):
            raise NodeError(f"file not found: {path}")
        mtime = os.path.getmtime(path)
        gray = bool(self.get("grayscale"))
        if self._cache and self._cache[:3] == (path, mtime, gray):
            img = self._cache[3]
        else:
            data = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
            if data is None:
                raise NodeError(f"cannot decode {path}")
            img = Image(data=data, source=path)
            self._cache = (path, mtime, gray, img)
        return {"image": Image(img.data, frame_id=ctx.run_id, source=path), "path": path}


@register
class ImageFolder(Node):
    type_id = "source.image_folder"
    category = "Source"
    label = "Image Folder"
    description = "Iterate over the images of a folder, one per run (simulated production)."
    color = "#2e7d32"
    outputs = [Port("image", DataType.IMAGE), Port("path", DataType.STRING), Port("index", DataType.INT)]
    params = [Param("directory", "", "dir"),
              Param("mode", "next", "enum", choices=["next", "fixed", "random"]),
              Param("index", 0, "int", min=0, description="Image index for 'fixed' mode"),
              Param("loop", True, "bool"),
              Param("grayscale", False, "bool")]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._files: list[str] = []
        self._dir_key = None

    def _scan(self):
        d = paths.resolve(self.get("directory"))
        if not d or not os.path.isdir(d):
            raise NodeError(f"directory not found: {d!r}")
        key = (d, os.path.getmtime(d))
        if key != self._dir_key:
            self._files = sorted(str(p) for p in Path(d).iterdir() if p.suffix.lower() in _EXTS)
            self._dir_key = key
        if not self._files:
            raise NodeError(f"no images in {d}")

    def process(self, ctx, inputs):
        self._scan()
        n = len(self._files)
        mode = self.get("mode")
        if mode == "fixed":
            i = min(int(self.get("index")), n - 1)
        elif mode == "random":
            import random
            i = random.randrange(n)
        else:
            i = int(self.state.get("next", 0))
            if i >= n:
                if not self.get("loop"):
                    raise NodeError("end of folder reached")
                i = 0
            self.state["next"] = i + 1
        path = self._files[i]
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.get("grayscale") else cv2.IMREAD_COLOR)
        if data is None:
            raise NodeError(f"cannot decode {path}")
        ctx.log(f"{os.path.basename(path)} ({i + 1}/{n})", "debug")
        return {"image": Image(data, frame_id=ctx.run_id, source=path), "path": path, "index": i}


@register
class CameraSource(Node):
    type_id = "source.camera"
    category = "Source"
    label = "Camera"
    description = "Grab a frame from a named camera (folder simulator, OpenCV device/stream, GenICam)."
    color = "#2e7d32"
    outputs = [Port("image", DataType.IMAGE), Port("frame_id", DataType.INT)]
    params = [Param("camera", "cam0", "string", description="Shared camera name"),
              Param("kind", "folder", "enum", choices=sorted(CAMERA_KINDS)),
              Param("source", "", "string", description="folder path | device index/URL | GenICam index/serial"),
              Param("cti", "", "file", label="GenTL producer (.cti)", advanced=True),
              Param("timeout_s", 2.0, "float", min=0.01, max=60),
              Param("grayscale", False, "bool"),
              Param("features", {}, "json", label="Camera features (JSON)", advanced=True)]

    def _config(self):
        src = self.get("source")
        if self.get("kind") == "folder":
            src = paths.resolve(src)
        cfg = {"source": src, "grayscale": self.get("grayscale"), "loop": True}
        if self.get("cti"):
            cfg["cti"] = paths.resolve(self.get("cti"))
        if self.get("features"):
            cfg["features"] = self.get("features")
        return cfg

    def setup(self, ctx=None):
        camera_manager.get_or_create(self.get("camera"), self.get("kind"), self._config()).open()

    def teardown(self):
        camera_manager.close(self.get("camera"))

    def process(self, ctx, inputs):
        cam = camera_manager.get_or_create(self.get("camera"), self.get("kind"), self._config())
        img = cam.grab(float(self.get("timeout_s")))
        if img is None:
            raise NodeError(f"camera {cam.name!r}: grab timeout")
        return {"image": img, "frame_id": img.frame_id}


@register
class Constant(Node):
    type_id = "source.constant"
    category = "Source"
    label = "Constant"
    description = "Emit a constant value."
    color = "#2e7d32"
    outputs = [Port("value", DataType.ANY)]
    params = [Param("kind", "float", "enum", choices=["float", "int", "string", "bool"]),
              Param("value", "0", "string")]

    def process(self, ctx, inputs):
        k, v = self.get("kind"), self.get("value")
        if k == "float":
            return {"value": float(v)}
        if k == "int":
            return {"value": int(float(v))}
        if k == "bool":
            return {"value": str(v).strip().lower() in ("1", "true", "yes", "on")}
        return {"value": str(v)}


@register
class TriggerData(Node):
    type_id = "source.trigger"
    category = "Source"
    label = "Trigger Data"
    description = "Expose the trigger that started this run (e.g. the PLC message) to the flow."
    color = "#2e7d32"
    outputs = [Port("message", DataType.STRING), Port("source", DataType.STRING),
               Port("device", DataType.STRING), Port("payload", DataType.DICT)]

    def process(self, ctx, inputs):
        t = ctx.trigger
        return {"message": getattr(t, "message", "") or "",
                "source": getattr(getattr(t, "source", None), "value", "manual"),
                "device": getattr(t, "device", "") or "",
                "payload": dict(getattr(t, "payload", None) or {})}

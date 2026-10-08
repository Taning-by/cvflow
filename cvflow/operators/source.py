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
    triggerable = True        # 画布上带触发图标：可以只触发这一路
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
            raise NodeError("未设置图像路径")
        if not os.path.isfile(path):
            raise NodeError(f"文件不存在：{path}")
        mtime = os.path.getmtime(path)
        gray = bool(self.get("grayscale"))
        if self._cache and self._cache[:3] == (path, mtime, gray):
            img = self._cache[3]
        else:
            data = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
            if data is None:
                raise NodeError(f"无法解码 {path}")
            img = Image(data=data, source=path)
            self._cache = (path, mtime, gray, img)
        return {"image": Image(img.data, frame_id=ctx.run_id, source=path), "path": path}


@register
class ImageFolder(Node):
    type_id = "source.image_folder"
    category = "Source"
    label = "Image Folder"
    triggerable = True        # 画布上带触发图标：可以只触发这一路
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
            raise NodeError(f"目录不存在：{d!r}")
        key = (d, os.path.getmtime(d))
        if key != self._dir_key:
            self._files = sorted(str(p) for p in Path(d).iterdir() if p.suffix.lower() in _EXTS)
            self._dir_key = key
        if not self._files:
            raise NodeError(f"{d} 中没有图像")

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
                    raise NodeError("已到文件夹末尾")
                i = 0
            self.state["next"] = i + 1
        path = self._files[i]
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.get("grayscale") else cv2.IMREAD_COLOR)
        if data is None:
            raise NodeError(f"无法解码 {path}")
        ctx.log(f"{os.path.basename(path)} ({i + 1}/{n})", "debug")
        return {"image": Image(data, frame_id=ctx.run_id, source=path), "path": path, "index": i}


@register
class CameraSource(Node):
    type_id = "source.camera"
    category = "Source"
    label = "Camera"
    triggerable = True        # 画布上带触发图标：可以只触发这一路
    description = "Grab a frame from a named camera (folder simulator, OpenCV device/stream, GenICam)."
    color = "#2e7d32"
    outputs = [Port("image", DataType.IMAGE), Port("frame_id", DataType.INT)]
    params = [Param("camera", "cam0", "string", description="Shared camera name"),
              Param("kind", "folder", "enum", choices=sorted(CAMERA_KINDS),
                    description="folder=文件夹模拟 opencv=USB/视频流 hik=海康 MVS "
                                "mindvision=迈德威视（USB3/GigE，走厂商 SDK）genicam=GenTL 驱动"),
              Param("source", "", "string", description="folder path | device index/URL | GenICam index/serial"),
              Param("backend", "auto", "enum", choices=["auto", "dshow", "msmf", "v4l2", "gstreamer", "ffmpeg"],
                    advanced=True,
                    description="只对 opencv 取图方式有效。Windows 默认走 MSMF，它只认 UVC 设备；"
                                "厂商注册的 DirectShow 滤镜（如迈德威视的 MindVision.ax）要选 dshow"),
              Param("trigger_mode", "keep", "enum", choices=["keep", "off", "software", "hardware"],
                    description="keep=沿用相机当前设置 off=自由采集 software=软触发 hardware=外部触发"),
              Param("trigger_source", "Line0", "string", description="硬件触发信号线", advanced=True),
              Param("exposure_us", 0.0, "float", min=0, max=10_000_000, description="0 表示沿用相机当前曝光"),
              Param("gain", -1.0, "float", min=-1, max=100, description="-1 表示沿用相机当前增益"),
              Param("timeout_s", 2.0, "float", min=0.01, max=60),
              Param("grayscale", False, "bool"),
              Param("cti", "", "file", label="GenTL producer (.cti)", advanced=True),
              Param("sdk_path", "", "dir", label="MindVision SDK path", advanced=True,
                    description="迈德威视 SDK 里 mvsdk.py 所在的目录（一般是 Demo\\Python）。"
                                "留空则按环境变量 MVSDK_PATH 和常见安装路径找"),
              Param("features", {}, "json", label="Camera features (JSON)", advanced=True)]

    def _config(self):
        src = self.get("source")
        if self.get("kind") == "folder":
            src = paths.resolve(src)
        cfg = {"source": src, "grayscale": self.get("grayscale"), "loop": True, "backend": self.get("backend"),
               "trigger_mode": self.get("trigger_mode"), "trigger_source": self.get("trigger_source"),
               "exposure_us": self.get("exposure_us"), "gain": self.get("gain")}
        if self.get("cti"):
            cfg["cti"] = paths.resolve(self.get("cti"))
        if self.get("sdk_path"):
            cfg["sdk_path"] = paths.resolve(self.get("sdk_path"))
        if self.get("features"):
            cfg["features"] = self.get("features")
        return cfg

    def setup(self, ctx=None):
        camera_manager.get_or_create(self.get("camera"), self.get("kind"), self._config()).open()

    def teardown(self):
        camera_manager.close(self.get("camera"))

    def process(self, ctx, inputs):
        cam = camera_manager.get_or_create(self.get("camera"), self.get("kind"), self._config())
        if self.get("trigger_mode") == "software":
            if not cam.is_open:
                cam.open()
            cam.software_trigger()
        img = cam.grab(float(self.get("timeout_s")))
        if img is None:
            raise NodeError(f"相机 {cam.name!r}：取图超时")
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
    description = ("Expose the trigger that started this run: the PLC message, the request id and "
                   "the parameters frozen when the request was accepted.")
    color = "#2e7d32"
    outputs = [Port("message", DataType.STRING), Port("source", DataType.STRING),
               Port("device", DataType.STRING), Port("payload", DataType.DICT),
               Port("request_id", DataType.INT), Port("fields", DataType.DICT),
               Port("value", DataType.ANY), Port("peer", DataType.STRING)]
    params = [Param("field", "", "string", description="要单独输出到 value 端口的字段名"),
              Param("required", False, "bool", description="字段不存在时报错")]

    def process(self, ctx, inputs):
        """``fields`` 是**接受请求时冻结**的那一份参数，所以同一条流程并发排队也不会串。"""
        t = ctx.trigger
        payload = dict(getattr(t, "payload", None) or {})
        fields = dict(payload.get("fields") or {})
        name = str(self.get("field") or "")
        if name and self.get("required") and name not in fields:
            raise NodeError(f"本次触发没有字段 {name!r}（有：{'、'.join(fields) or '无'}）")
        return {"message": getattr(t, "message", "") or "",
                "source": getattr(getattr(t, "source", None), "value", "manual"),
                "device": getattr(t, "device", "") or "",
                "payload": payload,
                "request_id": int(payload.get("request_id") or 0),
                "fields": fields,
                "value": fields.get(name) if name else None,
                "peer": str(payload.get("peer") or "")}

"""采集类节点：图像文件、文件夹、相机、常量，以及路径入口「触发」节点。

「触发」节点（``source.trigger``）是**一条路的入口**，只管触发：它的 ``trigger`` 输出接到
采集节点上，这一路就可以被单独触发（通信面板的触发规则指定节点，或者在画布上点它的触发图标），
别的路不动。本次请求的参数去「解析报文」节点取（来源选「触发」）。详见 ``TriggerNode``。
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2

from ..camera import camera_manager, CAMERA_KINDS
from ..core import paths
from ..core.node import TRIGGER_PORT, Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType, Image
from ._roi import apply_roi, roi_auto_enable, roi_params
from ._util import odd  # noqa: F401  (re-export convenience)

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def trigger_input() -> Port:
    """采集节点的"触发"输入：接「触发」节点，声明这一路由它触发。

    端口是**可选**的：不接就跟着整条流程跑（手动运行、定时运行、整流程触发都照旧）。
    接上之后这一路就有了明确的入口，PLC 的一条报文只会跑这一路。
    """
    return Port(TRIGGER_PORT, DataType.ANY, optional=True,
                description="接「触发」节点：这一路由它触发；不接则跟着整条流程跑")


@register
class ImageFile(Node):
    type_id = "source.image_file"
    category = "Source"
    label = "Image File"
    triggerable = True        # 画布上带触发图标：可以只触发这一路
    description = "Load a single image from disk (cached until the file changes)."
    color = "#2e7d32"
    inputs = [trigger_input()]
    outputs = [Port("image", DataType.IMAGE), Port("path", DataType.STRING),
               Port("roi", DataType.RECT, description="输出的图对应原图的哪一块；没开裁剪时就是整幅")]
    params = [Param("path", "", "file", label="Path", filter=_IMG_FILTER),
              Param("grayscale", False, "bool")] + roi_params()

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._cache: tuple[str, float, bool, Image] | None = None
        self._full: Image | None = None          # 裁剪前的整帧，界面上拖 ROI 框时用

    def roi_reference_image(self):
        return self._full

    def on_param_changed(self, name, value):
        roi_auto_enable(self, name, value)

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
        self._full = Image(img.data, frame_id=ctx.run_id, source=path)
        out, rect = apply_roi(self, self._full, ctx)
        return {"image": out, "path": path, "roi": rect}


@register
class ImageFolder(Node):
    type_id = "source.image_folder"
    category = "Source"
    label = "Image Folder"
    triggerable = True        # 画布上带触发图标：可以只触发这一路
    description = "Iterate over the images of a folder, one per run (simulated production)."
    color = "#2e7d32"
    inputs = [trigger_input()]
    outputs = [Port("image", DataType.IMAGE), Port("path", DataType.STRING), Port("index", DataType.INT),
               Port("roi", DataType.RECT, description="输出的图对应原图的哪一块；没开裁剪时就是整幅")]
    params = [Param("directory", "", "dir"),
              Param("mode", "next", "enum", choices=["next", "fixed", "random"]),
              Param("index", 0, "int", min=0, description="Image index for 'fixed' mode"),
              Param("loop", True, "bool"),
              Param("grayscale", False, "bool")] + roi_params()

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._files: list[str] = []
        self._dir_key = None
        self._full: Image | None = None          # 裁剪前的整帧，界面上拖 ROI 框时用

    def roi_reference_image(self):
        return self._full

    def on_param_changed(self, name, value):
        roi_auto_enable(self, name, value)

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
        self._full = Image(data, frame_id=ctx.run_id, source=path)
        out, rect = apply_roi(self, self._full, ctx)
        return {"image": out, "path": path, "index": i, "roi": rect}


@register
class CameraSource(Node):
    type_id = "source.camera"
    category = "Source"
    label = "Camera"
    triggerable = True        # 画布上带触发图标：可以只触发这一路
    description = "Grab a frame from a named camera (folder simulator, OpenCV device/stream, GenICam)."
    color = "#2e7d32"
    inputs = [trigger_input()]
    outputs = [Port("image", DataType.IMAGE), Port("frame_id", DataType.INT),
               Port("roi", DataType.RECT, description="输出的图对应原图的哪一块；没开裁剪时就是整幅")]
    params = [Param("camera", "cam0", "string", description="Shared camera name"),
              Param("kind", "folder", "enum", choices=sorted(CAMERA_KINDS),
                    description="folder=文件夹模拟 opencv=USB/视频流 hik=海康 MVS "
                                "mindvision=迈德威视（USB3/GigE，走厂商 SDK）genicam=GenTL 驱动"),
              Param("source", "", "string", description="folder path | device index/URL | GenICam index/serial"),
              Param("backend", "auto", "enum", choices=["auto", "dshow", "msmf", "v4l2", "gstreamer", "ffmpeg"],
                    advanced=True,
                    description="只对 opencv 取图方式有效。Windows 默认走 MSMF，它只认 UVC 设备；"
                                "厂商注册的 DirectShow 滤镜（如迈德威视的 MindVision.ax）要选 dshow"),
              Param("user_set", "keep", "enum",
                    choices=["keep", "Default", "UserSet1", "UserSet2", "UserSet3"],
                    description="打开相机时从相机 Flash 恢复一组参数（GenTL / 海康）。"
                                "keep=不加载，沿用相机当前状态。加载在曝光/增益/触发之前，"
                                "所以下面几项仍然生效"),
              Param("trigger_mode", "keep", "enum", choices=["keep", "off", "software", "hardware"],
                    description="keep=沿用相机当前设置 off=自由采集 software=软触发 hardware=外部触发"),
              Param("trigger_source", "Line0", "string", description="硬件触发信号线", advanced=True),
              Param("exposure_us", 0.0, "float", min=0, max=10_000_000, description="0 表示沿用相机当前曝光"),
              Param("gain", -1.0, "float", min=-1, max=100, description="-1 表示沿用相机当前增益"),
              Param("timeout_s", 2.0, "float", min=0.01, max=60),
              Param("grayscale", False, "bool")] + roi_params() + [
              Param("cti", "", "file", label="GenTL producer (.cti)", advanced=True),
              Param("sdk_path", "", "dir", label="Vendor SDK path", advanced=True,
                    description="厂商 SDK 的目录，装在非默认路径时才要填。"
                                "迈德威视填 mvsdk.py 所在目录（一般是 Demo\\Python）；"
                                "海康填 MvImport 目录（一般是 MVS\\Development\\Samples\\Python\\MvImport），"
                                "或者填运行库目录（里面有 MvCameraControl.dll，"
                                "一般是 Common Files\\MVS\\Runtime\\Win64_x64）。"
                                "留空则按环境变量和常见安装路径找"),
              Param("features", {}, "json", label="Camera features (JSON)", advanced=True)]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._full: Image | None = None          # 裁剪前的整帧，界面上拖 ROI 框时用

    def roi_reference_image(self):
        return self._full

    def on_param_changed(self, name, value):
        roi_auto_enable(self, name, value)

    def _config(self):
        src = self.get("source")
        if self.get("kind") == "folder":
            src = paths.resolve(src)
        cfg = {"source": src, "grayscale": self.get("grayscale"), "loop": True, "backend": self.get("backend"),
               "user_set": self.get("user_set"),
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
            mode = str(self.get("trigger_mode"))
            why = ("相机多半停在触发模式等信号——先把触发模式改成 off 自由采集试一次"
                   if mode == "keep" else
                   f"触发模式是 {mode}，确认触发信号真的到了相机")
            raise NodeError(f"相机 {cam.name!r}：{self.get('timeout_s')} 秒内没有取到图像。{why}；"
                            f"也可能是防火墙拦了图像流（GigE 的 GVSP 走入站 UDP）、"
                            f"曝光时间超过了超时、或多台相机挤在同一条链路上带宽不够")
        self._full = img
        out, rect = apply_roi(self, img, ctx)
        return {"image": out, "frame_id": img.frame_id, "roi": rect}


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
class TriggerNode(Node):
    """路径入口：一条"路"从哪里开始、由谁触发。**只管触发，不带数据。**

    把 ``trigger`` 输出接到采集节点（相机 / 图像文件 / 图像文件夹）的 ``trigger`` 输入上，
    这条路就被声明出来了：

    * 通信面板「触发」页里把动作选成「触发节点」、再指到这个节点，PLC 的一条报文
      （或一个寄存器位）就**只跑这一路**——它下游的采集、算法、输出都跑，别的路一动不动
      （那些相机不会被白取一帧）。
    * 画布上它自带触发图标，点一下就在编辑模式下手动跑一次这一路，接线对不对当场看得出来。

    本次请求的参数（请求编号、解析出的字段、来源对端）**不在这里取**：
    用通信分类的「解析报文」节点，来源选「触发」——那里拿到的是接受请求时冻结的那一份，
    同一条流程并发排队也不会串；回传报文直接在模板里写 ``{request_id}`` / ``{field.名字}``。
    """

    type_id = "source.trigger"
    category = "Source"
    label = "Trigger"
    triggerable = True        # 画布上带触发图标：点一下只跑这一路
    is_trigger_entry = True   # trigger 输出接了下游时，"它和它的下游"就是一路
    description = ("Entry point of one path: wire 'trigger' into an acquisition node, then a PLC rule "
                   "(action: trigger node) runs just that path. Request parameters come from Parse Message.")
    color = "#2e7d32"
    outputs = [Port(TRIGGER_PORT, DataType.ANY,
                    description="接到「采集」类节点的 trigger 输入上，表示那一路由本节点触发"
                                "（值恒为 True，只用来声明路径和执行顺序）")]

    def process(self, ctx, inputs):
        return {TRIGGER_PORT: True}

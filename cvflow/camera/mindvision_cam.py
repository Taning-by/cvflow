"""迈德威视（MindVision）工业相机，基于厂商自带的 Python 接口 ``mvsdk``。

这类相机（USB3.0 / GigE）走的是厂商自己的协议，不是 USB3 Vision / GigE Vision 标准，
所以用 OpenCV 或 GenICam 那两条路都接不上，必须用它的 SDK。

需要先装 MindVision 的相机驱动与 SDK（装完 ``MVCAMSDK_X64.dll`` 会进系统目录），
SDK 里的 Python 封装是一个 ``mvsdk.py`` 文件，一般在 Demo\\Python 之类的目录下。
找不到时可以用环境变量 ``MVSDK_PATH`` 指向它所在的目录，或在相机参数里填 ``sdk_path``。

和海康那个后端一样，这里只做三件事：按序列号/名字/索引选中相机、把曝光增益触发映射过去、
把一帧转成 BGR 或灰度的 numpy 数组。
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Any

import numpy as np

from .base import Camera, CameraError
from .discovery import GigEDevice

#: mvsdk.py 可能在的地方。装驱动时路径可选，所以多列几个常见的。
_SDK_DIRS = [
    os.environ.get("MVSDK_PATH", ""),
    r"C:\Program Files (x86)\MindVision\Demo\Python",
    r"C:\Program Files\MindVision\Demo\Python",
    r"C:\MindVision\Demo\Python",
    "/opt/MVSDK/Python",
    "/usr/local/MVSDK/Python",
]

_sdk = None


def load_sdk(extra_dir: str = ""):
    """导入 mvsdk；找不到时抛出带安装提示的 CameraError。"""
    global _sdk
    if _sdk is not None:
        return _sdk
    dirs = ([extra_dir] if extra_dir else []) + _SDK_DIRS
    try:
        import mvsdk  # type: ignore
    except ImportError:
        mvsdk = None
        for d in dirs:
            if d and os.path.isdir(d) and d not in sys.path:
                sys.path.append(d)
                try:
                    import mvsdk  # type: ignore
                    break
                except ImportError:
                    continue
        if mvsdk is None:
            raise CameraError(
                "未找到迈德威视 SDK 的 Python 模块 mvsdk。请先安装 MindVision 的相机驱动/SDK，"
                "再把 SDK 里 mvsdk.py 所在的目录（一般是 Demo\\Python）填进相机参数的 sdk_path，"
                "或设成环境变量 MVSDK_PATH。")
    _sdk = mvsdk
    return mvsdk


def sdk_available() -> bool:
    try:
        load_sdk()
        return True
    except CameraError:
        return False


def _text(dev, getter: str, attr: str) -> str:
    """mvsdk 的设备信息里字符串是字节数组，新旧版本取法不一样，两种都试。"""
    fn = getattr(dev, getter, None)
    if callable(fn):
        try:
            v = fn()
            return v.decode("utf-8", "replace").strip() if isinstance(v, bytes) else str(v).strip()
        except Exception:                                  # pragma: no cover - 取决于 SDK 版本
            pass
    raw = getattr(dev, attr, b"")
    try:
        return bytes(raw).split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    except Exception:                                      # pragma: no cover
        return ""


def enumerate_mindvision() -> list[GigEDevice]:
    """枚举迈德威视相机，结果并进"搜索相机"的列表里。"""
    try:
        mvsdk = load_sdk()
    except CameraError:
        return []
    try:
        devs = mvsdk.CameraEnumerateDevice()
    except Exception:                                      # pragma: no cover - 取决于本机安装
        return []
    out = []
    for i, d in enumerate(devs or []):
        serial = _text(d, "GetSn", "acSn")
        out.append(GigEDevice(
            ip="", mac="", manufacturer="MindVision",
            model=_text(d, "GetProductModel", "acProductName"),
            serial=serial,
            user_name=_text(d, "GetFriendlyName", "acFriendlyName"),
            source="mindvision",
            extra={"index": i, "port": _text(d, "GetPortType", "acPortType")},
        ))
    return out


class MindVisionCamera(Camera):
    """``source`` 可以是序列号、友好名、型号或枚举索引；留空取第一台。

    额外参数：``sdk_path`` —— mvsdk.py 所在目录（环境变量 MVSDK_PATH 之外的另一条路）。
    """

    kind = "mindvision"

    def __init__(self, name: str, config=None) -> None:
        super().__init__(name, config)
        self._sdk = None
        self._h = None                    # 相机句柄
        self._buf = None                  # 对齐分配的输出缓冲，open 时一次分配
        self._mono = False

    # ---- 选相机 ----
    def _select(self, mvsdk, devs):
        want = str(self.config.get("source", "")).strip()
        for i, d in enumerate(devs):
            keys = {_text(d, "GetSn", "acSn"), _text(d, "GetFriendlyName", "acFriendlyName"),
                    _text(d, "GetProductModel", "acProductName")}
            if not want or want in keys or want == str(i):
                return d
        names = "、".join(f"{i}:{_text(d, 'GetFriendlyName', 'acFriendlyName')}" for i, d in enumerate(devs))
        raise CameraError(f"迈德威视：未找到相机 {want!r}（可用：{names}；也可以填序列号或索引）")

    # ---- 生命周期 ----
    def open(self) -> None:
        mvsdk = load_sdk(str(self.config.get("sdk_path", "")))
        devs = mvsdk.CameraEnumerateDevice()
        if not devs:
            raise CameraError("迈德威视：未发现任何相机（驱动装了吗？相机是否被其它软件占用？）")
        info = self._select(mvsdk, devs)
        try:
            h = mvsdk.CameraInit(info, -1, -1)
        except Exception as e:
            raise CameraError(f"迈德威视：打开相机失败（{e}）。相机可能正被厂商的演示软件占用") from e
        self._sdk, self._h = mvsdk, h
        try:
            cap = mvsdk.CameraGetCapability(h)
            self._mono = bool(cap.sIspCapacity.bMonoSensor)
            mvsdk.CameraSetIspOutFormat(
                h, mvsdk.CAMERA_MEDIA_TYPE_MONO8 if self._mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            # 输出缓冲按最大分辨率一次分配，之后每帧复用
            size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if self._mono else 3)
            self._buf = mvsdk.CameraAlignMalloc(size, 16)
            self.apply_settings(self.config)
            mvsdk.CameraPlay(h)
        except Exception:
            self.close()
            raise
        self.is_open = True

    def close(self) -> None:
        mvsdk, h, buf = self._sdk, self._h, self._buf
        try:
            if h is not None:
                mvsdk.CameraUnInit(h)
        except Exception:                                  # pragma: no cover - 关相机失败不该再抛
            pass
        finally:
            if buf is not None and mvsdk is not None:
                try:
                    mvsdk.CameraAlignFree(buf)
                except Exception:                          # pragma: no cover
                    pass
            self._h = self._buf = None
            self.is_open = False

    # ---- 取图 ----
    def _grab_raw(self, timeout_s: float) -> np.ndarray | None:
        mvsdk, h = self._sdk, self._h
        if h is None:
            return None
        try:
            raw, head = mvsdk.CameraGetImageBuffer(h, int(timeout_s * 1000))
        except Exception as e:
            code = getattr(e, "error_code", None)
            if code is not None and code == getattr(mvsdk, "CAMERA_STATUS_TIME_OUT", -12):
                return None                                # 超时按"这一轮没图"处理，交给上层
            raise CameraError(f"迈德威视：取图失败（{e}）") from e
        try:
            mvsdk.CameraImageProcess(h, raw, self._buf, head)
        finally:
            mvsdk.CameraReleaseImageBuffer(h, raw)
        # Windows 下 SDK 输出的是自下而上的位图，不翻转的话整幅图是倒的
        if platform.system() == "Windows":
            mvsdk.CameraFlipFrameBuffer(self._buf, head, 1)
        n = int(head.uBytes)
        data = np.frombuffer((mvsdk.c_ubyte * n).from_address(self._buf), dtype=np.uint8)
        h_, w_ = int(head.iHeight), int(head.iWidth)
        ch = 1 if self._mono else 3
        if n < h_ * w_ * ch:                               # pragma: no cover - SDK 不该给出这种帧
            raise CameraError(f"迈德威视：帧长度不对（{n} < {h_}×{w_}×{ch}）")
        img = data[: h_ * w_ * ch].reshape(h_, w_, ch)
        return img[:, :, 0].copy() if ch == 1 else img.copy()

    # ---- 参数 ----
    def apply_settings(self, cfg: dict[str, Any]) -> None:
        """把通用设置映射到 mvsdk 的接口。

        基类那套是 GenICam 的特性名（ExposureTime / TriggerMode…），迈德威视不吃那一套，
        它有自己的函数，所以这里整个覆盖掉。
        """
        mvsdk, h = self._sdk, self._h
        if h is None:
            return
        mode = cfg.get("trigger_mode", "keep")
        if mode != "keep":
            # 0=连续采集 1=软触发 2=硬触发
            mvsdk.CameraSetTriggerMode(h, {"off": 0, "software": 1, "hardware": 2}[mode])
        exp = cfg.get("exposure_us")
        if exp not in (None, "", 0, 0.0):
            mvsdk.CameraSetAeState(h, 0)                   # 先关自动曝光，否则设了也会被改回去
            mvsdk.CameraSetExposureTime(h, float(exp))
        gain = cfg.get("gain")
        if gain not in (None, "") and float(gain) >= 0:
            mvsdk.CameraSetAeState(h, 0)
            setter = getattr(mvsdk, "CameraSetAnalogGainX", None)
            if callable(setter):
                setter(h, float(gain))                     # 新版本按倍数
            else:                                          # pragma: no cover - 老版本按档位
                mvsdk.CameraSetAnalogGain(h, int(gain))

    def software_trigger(self) -> None:
        mvsdk, h = self._sdk, self._h
        if h is not None:
            mvsdk.CameraSoftTrigger(h)

    def test_connection(self) -> tuple[bool, str]:
        try:
            was_open = self.is_open
            if not was_open:
                self.open()
            kind = "黑白" if self._mono else "彩色"
            if not was_open:
                self.close()
            return True, f"迈德威视：相机 {self.config.get('source') or ''} 可以打开（{kind}）"
        except CameraError as e:
            return False, str(e)
        except Exception as e:                             # pragma: no cover - SDK 可能抛自己的异常
            return False, f"{type(e).__name__}: {e}"

"""迈德威视（MindVision）工业相机，基于厂商自带的 Python 接口 ``mvsdk``。

这类相机（USB3.0 / GigE）走的是厂商自己的协议，不是 USB3 Vision / GigE Vision 标准，
所以用 OpenCV 或 GenICam 那两条路都接不上，必须用它的 SDK。

需要先装 MindVision 的相机驱动与 SDK（装完 ``MVCAMSDK_X64.dll`` 会进系统目录），
SDK 里的 Python 封装是一个 ``mvsdk.py`` 文件，一般在 Demo\\Python 之类的目录下。
找不到时可以用环境变量 ``MVSDK_PATH`` 指向它所在的目录，或在相机参数里填 ``sdk_path``。

和海康那个后端一样，这里只做四件事：按序列号/名字/索引选中相机、加载厂商软件里存好的
那套成像参数（白平衡这类不加载就会整幅偏色）、把曝光增益触发映射过去、
把一帧转成 BGR 或灰度的 numpy 数组。
"""
from __future__ import annotations

import os
import platform
import sys
from typing import Any

import numpy as np

from .base import Camera, CameraError, parse_wb_gain
from .discovery import GigEDevice

#: mvsdk.py 可能在的地方。装驱动时路径可选，各版本 Demo 的层级也不一致
#: （实测有装在 Demo\Python\Advanced 下的），所以按「根目录 × 子路径」组合着找。
_SDK_ROOTS = [
    os.environ.get("MVSDK_PATH", ""),
    r"C:\Program Files (x86)\MindVision",
    r"C:\Program Files\MindVision",
    r"C:\MindVision",
    "/opt/MVSDK",
    "/usr/local/MVSDK",
]
_SDK_SUBDIRS = ["", "Demo/Python/Advanced", "Demo/Python/Basic", "Demo/Python", "Python"]

#: 「相机配置参数存档」的分组，对应厂商软件相机设置页底部那排 A/B/C/D 单选。
_PARAM_TEAMS = {"A": 0, "B": 1, "C": 2, "D": 3}


def _call(mvsdk, name: str, *args) -> bool:
    """调 SDK 里不一定存在的函数（各版本接口有出入），没有就当没调过。"""
    fn = getattr(mvsdk, name, None)
    if not callable(fn):
        return False
    fn(*args)
    return True



def _sdk_dirs(extra: str = "") -> list[str]:
    """存在的候选目录，按优先级排好；extra 来自相机参数 sdk_path。"""
    out: list[str] = []
    for root in ([extra] if extra else []) + _SDK_ROOTS:
        if not root:
            continue
        for sub in _SDK_SUBDIRS:
            d = os.path.join(root, *sub.split("/")) if sub else root
            if os.path.isdir(d) and d not in out:
                out.append(d)
    return out

_sdk = None


def load_sdk(extra_dir: str = ""):
    """导入 mvsdk；找不到时抛出带安装提示的 CameraError。"""
    global _sdk
    if _sdk is not None:
        return _sdk
    try:
        import mvsdk  # type: ignore
    except ImportError:
        mvsdk, tried = None, _sdk_dirs(extra_dir)
        for d in tried:
            if d not in sys.path:
                sys.path.append(d)
            try:
                import mvsdk  # type: ignore
                break
            except ImportError:
                continue
        if mvsdk is None:
            looked = "\n  ".join(tried) if tried else "（没有一个候选目录存在）"
            raise CameraError(
                "未找到迈德威视 SDK 的 Python 模块 mvsdk。请先安装 MindVision 的相机驱动/SDK，"
                "再把 mvsdk.py 所在的目录填进相机参数的 sdk_path，或设成环境变量 MVSDK_PATH；"
                "典型位置是 <MindVision>\\Demo\\Python\\Advanced。"
                f"\n找过这些目录：\n  {looked}")
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
            self._load_saved_params()
            # 存档里也带输出格式，所以要放在加载之后设，免得被改回去
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

    def _load_saved_params(self) -> None:
        """把厂商软件里调好的那套参数搬进来：白平衡、颜色矩阵、Gamma、饱和度、LUT…

        ``CameraInit(info, -1, -1)`` 只会按 SDK 自己的存档目录和上次的加载方式去找，
        换个进程、换个用户、换台机器都可能对不上，于是相机就停在出厂的白平衡上——
        现象就是整幅图偏色。所以这里显式再加载一次：
        先按分组（对应厂商软件底部那排 A/B/C/D），再按文件（更具体，放后面好覆盖）。
        """
        mvsdk, h = self._sdk, self._h
        team = str(self.config.get("param_group", "keep") or "keep").strip().upper()
        if team in _PARAM_TEAMS:
            try:
                mvsdk.CameraLoadParameter(h, _PARAM_TEAMS[team])
            except Exception as e:
                raise CameraError(
                    f"迈德威视：加载参数组 {team} 失败（{e}）。这组可能还没存过——"
                    f"在厂商软件里选中该组、点「保存」，或者改用参数文件") from e
        path = str(self.config.get("param_file", "") or "").strip()
        if not path:
            return
        if not os.path.isfile(path):
            raise CameraError(f"迈德威视：找不到参数文件 {path}（在厂商软件里用"
                              f"「保存当前配置参数到指定文件」导出）")
        try:
            mvsdk.CameraReadParameterFromFile(h, path)
        except Exception as e:
            raise CameraError(f"迈德威视：读取参数文件 {path} 失败（{e}）") from e

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
        self.apply_white_balance(cfg)

    def apply_white_balance(self, cfg: dict[str, Any]) -> None:
        """白平衡。基类那套 GenICam 特性名迈德威视不认，换成它自己的函数。

        黑白相机没有白平衡这回事，直接跳过。
        """
        mvsdk, h = self._sdk, self._h
        mode = str(cfg.get("wb_mode") or "keep").strip().lower()
        if mode == "keep" or self._mono:
            return
        if mode == "auto":
            _call(mvsdk, "CameraSetWbMode", h, True)
            return
        _call(mvsdk, "CameraSetWbMode", h, False)          # 不关自动，设完下一帧就被算回去
        if mode == "once":
            # 画面里得有白面或灰面，不然算出来的增益是歪的
            _call(mvsdk, "CameraSetOnceWB", h)
            return
        gains = parse_wb_gain(cfg.get("wb_gain"))
        if gains is None:
            raise CameraError("迈德威视：白平衡选了手动，但没填增益（形如 1.0,1.0,1.6）")
        scaled = tuple(round(v * 100) for v in gains)      # SDK 收的是 ×100 的整数刻度
        if not _call(mvsdk, "CameraSetGain", h, *scaled):  # pragma: no cover - 老版本没有
            raise CameraError("迈德威视：当前 SDK 版本没有 CameraSetGain，手动白平衡用不了")

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

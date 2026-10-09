"""海康机器人（HIKROBOT）工业相机，基于 MVS SDK 的 Python 接口 ``MvCameraControl_class``。

这是 VisionMaster 使用的同一套 SDK。需要先安装 MVS（Windows/Linux 都有，且要带上 Python 示例），
Python 模块在 ``<MVS>/Development/Samples/Python/MvImport`` 下。MVS 可以装在任意路径，所以这里
按环境变量（``MVCAM_SDK_PATH`` / ``MVCAM_COMMON_RUNENV``）与常见安装位置组合着找；都找不到时
可以在相机参数的 ``sdk_path`` 里直接指定。

装不到 Python 示例时改用 GenICam 方式（``kind=genicam``）：cti 在
``<Common Files>/MVS/Runtime/Win64_x64/`` 下，GigE 相机用 ``MvProducerGEV.cti``，USB3 用 ``MvProducerU3V.cti``。
"""
from __future__ import annotations

import logging
import os
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memmove

import numpy as np

from .base import Camera, CameraError
from .discovery import GigEDevice

log = logging.getLogger("cvflow.camera")

#: MvCameraControl_class.py 可能在的地方。MVS 可以装在任意盘符，两个环境变量指向的层级
#: 也不一致（有的到 Development，有的到 MvImport），所以按「根目录 × 子路径」组合着找。
_SDK_ROOTS = [
    os.environ.get("MVCAM_SDK_PATH", ""),
    os.environ.get("MVCAM_COMMON_RUNENV", ""),
    r"C:\Program Files (x86)\MVS\Development",
    r"C:\Program Files\MVS\Development",
    "/opt/MVS",
]
_SDK_SUBDIRS = ["", "Samples/Python/MvImport", "Samples/64/Python/MvImport",
                "Samples/aarch64/Python/MvImport", "Samples/arm64/Python/MvImport",
                "Development/Samples/Python/MvImport"]


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


_PIX_MONO8 = 0x01080001
_PIX_MONO10 = 0x01100003
_PIX_MONO12 = 0x01100005
_PIX_BAYER = {0x01080008: "GR", 0x01080009: "RG", 0x0108000A: "GB", 0x0108000B: "BG"}
_PIX_RGB8 = 0x02180014
_PIX_BGR8 = 0x02180015

_mv = None


def load_sdk(extra_dir: str = ""):
    """导入 MvCameraControl_class；找不到时抛出带安装提示的 CameraError。"""
    global _mv
    if _mv is not None:
        return _mv
    try:
        import MvCameraControl_class as mv  # type: ignore
    except ImportError:
        mv, tried = None, _sdk_dirs(extra_dir)
        for d in tried:
            if d not in sys.path:
                sys.path.append(d)
            try:
                import MvCameraControl_class as mv  # type: ignore
                break
            except ImportError:
                continue
        if mv is None:
            looked = "\n  ".join(tried) if tried else "（没有一个候选目录存在）"
            raise CameraError(
                "未找到海康 MVS SDK 的 Python 模块 MvCameraControl_class。请先安装 MVS（含 Python 示例），"
                "再把 MvImport 目录填进相机参数的 sdk_path，或设成环境变量 MVCAM_SDK_PATH；"
                "典型位置是 <MVS 安装目录>\\Development\\Samples\\Python\\MvImport。"
                f"\n找过这些目录：\n  {looked}")
    _mv = mv
    return mv


def sdk_available() -> bool:
    try:
        load_sdk()
        return True
    except CameraError:
        return False


def _ip_str(n: int) -> str:
    return f"{(n >> 24) & 0xFF}.{(n >> 16) & 0xFF}.{(n >> 8) & 0xFF}.{n & 0xFF}"


def _cstr(arr) -> str:
    return bytes(arr).split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def _enum_list(mv):
    lst = mv.MV_CC_DEVICE_INFO_LIST()
    ret = mv.MvCamera.MV_CC_EnumDevices(mv.MV_GIGE_DEVICE | mv.MV_USB_DEVICE, lst)
    if ret != 0:
        raise CameraError(f"MVS 枚举设备失败：0x{ret & 0xFFFFFFFF:08X}")
    infos = []
    for i in range(lst.nDeviceNum):
        infos.append(cast(lst.pDeviceInfo[i], POINTER(mv.MV_CC_DEVICE_INFO)).contents)
    return lst, infos


def _mac(g) -> str:
    """MV_GIGE_DEVICE_INFO 里 MAC 的字段名各版本不一致，取不到就留空——去重本来就优先用序列号。"""
    hi, lo = getattr(g, "nMacAddrHigh", 0), getattr(g, "nMacAddrLow", 0)
    if not (hi or lo):
        return ""
    return ":".join(f"{b:02X}" for b in ((hi >> 8) & 0xFF, hi & 0xFF, (lo >> 24) & 0xFF,
                                         (lo >> 16) & 0xFF, (lo >> 8) & 0xFF, lo & 0xFF))


def enumerate_hik() -> list[GigEDevice]:
    """通过 MVS SDK 枚举 GigE 与 USB3 相机。"""
    mv = load_sdk()
    _, infos = _enum_list(mv)
    out: list[GigEDevice] = []
    for i, info in enumerate(infos):
        try:
            if info.nTLayerType == mv.MV_GIGE_DEVICE:
                g = info.SpecialInfo.stGigEInfo
                out.append(GigEDevice(ip=_ip_str(g.nCurrentIp), mac=_mac(g), subnet=_ip_str(g.nCurrentSubNetMask),
                                      gateway=_ip_str(g.nDefultGateWay), manufacturer=_cstr(g.chManufacturerName),
                                      model=_cstr(g.chModelName), version=_cstr(g.chDeviceVersion),
                                      serial=_cstr(g.chSerialNumber), user_name=_cstr(g.chUserDefinedName),
                                      interface_ip=_ip_str(g.nNetExport), source="hik",
                                      extra={"index": i, "layer": "gige"}))
            elif info.nTLayerType == mv.MV_USB_DEVICE:
                u = info.SpecialInfo.stUsb3VInfo
                out.append(GigEDevice(ip="", mac="", manufacturer=_cstr(u.chManufacturerName),
                                      model=_cstr(u.chModelName), version=_cstr(u.chDeviceVersion),
                                      serial=_cstr(u.chSerialNumber), user_name=_cstr(u.chUserDefinedName),
                                      source="hik", extra={"index": i, "layer": "usb"}))
        except Exception as e:      # 一台读不出来不该让其余几台也跟着消失
            log.warning("MVS：第 %d 台相机的信息解析失败，已跳过：%s", i, e)
    return out


class HikCamera(Camera):
    """``source`` 可以是序列号、IP、用户自定义名或枚举索引。"""

    kind = "hik"

    def __init__(self, name: str, config=None) -> None:
        super().__init__(name, config)
        self._cam = None
        self._mv = None

    def _select(self, mv, infos):
        want = str(self.config.get("source", "")).strip()
        for i, info in enumerate(infos):
            if info.nTLayerType == mv.MV_GIGE_DEVICE:
                g = info.SpecialInfo.stGigEInfo
                keys = {_ip_str(g.nCurrentIp), _cstr(g.chSerialNumber), _cstr(g.chUserDefinedName)}
            else:
                u = info.SpecialInfo.stUsb3VInfo
                keys = {_cstr(u.chSerialNumber), _cstr(u.chUserDefinedName)}
            if not want or want in keys or want == str(i):
                return info
        raise CameraError(f"MVS：未找到相机 {want!r}（可用：序列号 / IP / 用户名 / 索引）")

    def open(self) -> None:
        mv = load_sdk(str(self.config.get("sdk_path", "")))
        _, infos = _enum_list(mv)
        if not infos:
            raise CameraError("MVS：未发现任何相机")
        info = self._select(mv, infos)
        cam = mv.MvCamera()
        if cam.MV_CC_CreateHandle(info) != 0:
            raise CameraError("MVS：创建句柄失败")
        ret = cam.MV_CC_OpenDevice(mv.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise CameraError(f"MVS：打开相机失败 0x{ret & 0xFFFFFFFF:08X}（是否被其它软件占用？）")
        self._cam, self._mv = cam, mv
        if info.nTLayerType == mv.MV_GIGE_DEVICE:
            size = cam.MV_CC_GetOptimalPacketSize()
            if size > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", size)
        self.set_feature("AcquisitionMode", "Continuous")
        self.apply_settings(self.config)
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.close()
            raise CameraError(f"MVS：开始采集失败 0x{ret & 0xFFFFFFFF:08X}")
        self.is_open = True

    def close(self) -> None:
        cam = self._cam
        if cam is not None:
            try:
                cam.MV_CC_StopGrabbing()
                cam.MV_CC_CloseDevice()
            finally:
                cam.MV_CC_DestroyHandle()
        self._cam = None
        self.is_open = False

    def _grab_raw(self, timeout_s: float) -> np.ndarray | None:
        mv, cam = self._mv, self._cam
        assert cam is not None
        frame = mv.MV_FRAME_OUT()
        ret = cam.MV_CC_GetImageBuffer(frame, int(timeout_s * 1000))
        if ret != 0:
            return None
        try:
            fi = frame.stFrameInfo
            w, h, pt, n = fi.nWidth, fi.nHeight, fi.enPixelType, fi.nFrameLen
            buf = (c_ubyte * n)()
            memmove(byref(buf), frame.pBufAddr, n)
            data = np.frombuffer(buf, dtype=np.uint8)
            return _convert(data, w, h, pt)
        finally:
            cam.MV_CC_FreeImageBuffer(frame)

    # ---- 特性 ----
    def set_feature(self, name: str, value) -> None:
        cam = self._cam
        if cam is None:
            return
        if value is None:
            ret = cam.MV_CC_SetCommandValue(name)
        elif isinstance(value, bool):
            ret = cam.MV_CC_SetBoolValue(name, value)
        elif isinstance(value, int):
            ret = cam.MV_CC_SetIntValue(name, value)
        elif isinstance(value, float):
            ret = cam.MV_CC_SetFloatValue(name, value)
        else:
            ret = cam.MV_CC_SetEnumValueByString(name, str(value))
        if ret != 0:
            raise CameraError(f"MVS：设置 {name}={value!r} 失败 0x{ret & 0xFFFFFFFF:08X}")

    def get_feature(self, name: str):
        cam, mv = self._cam, self._mv
        if cam is None:
            return None
        v = mv.MVCC_FLOATVALUE()
        if cam.MV_CC_GetFloatValue(name, v) == 0:
            return float(v.fCurValue)
        iv = mv.MVCC_INTVALUE()
        if cam.MV_CC_GetIntValue(name, iv) == 0:
            return int(iv.nCurValue)
        return None

    def software_trigger(self) -> None:
        self.set_feature("TriggerSoftware", None)

    def test_connection(self) -> tuple[bool, str]:
        try:
            was_open = self.is_open
            if not was_open:
                self.open()
            model = self.get_feature("Width")
            if not was_open:
                self.close()
            return True, f"MVS：相机 {self.config.get('source') or ''} 可以打开，宽度 {model}"
        except CameraError as e:
            return False, str(e)


def _convert(data: np.ndarray, w: int, h: int, pt: int) -> np.ndarray:
    if pt == _PIX_MONO8:
        return data[: w * h].reshape(h, w).copy()
    if pt in _PIX_BAYER:
        import cv2
        code = getattr(cv2, f"COLOR_Bayer{_PIX_BAYER[pt]}2BGR")
        return cv2.cvtColor(data[: w * h].reshape(h, w), code)
    if pt == _PIX_RGB8:
        import cv2
        return cv2.cvtColor(data[: w * h * 3].reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if pt == _PIX_BGR8:
        return data[: w * h * 3].reshape(h, w, 3).copy()
    if pt in (_PIX_MONO10, _PIX_MONO12):
        bits = 10 if pt == _PIX_MONO10 else 12
        raw = data[: w * h * 2].view("<u2").reshape(h, w)
        return (raw >> (bits - 8)).astype(np.uint8)
    raise CameraError(f"MVS：暂不支持的像素格式 0x{pt:08X}，请在相机上改为 Mono8/Bayer8/RGB8")

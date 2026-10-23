"""海康机器人（HIKROBOT）工业相机，基于 MVS SDK 的 Python 接口 ``MvCameraControl_class``。

这是 VisionMaster 使用的同一套 SDK。需要先安装 MVS（Windows/Linux 都有，且要带上 Python 示例）。
SDK 分两半，**得分别找到**，少哪一半都报错：

* Python 封装 ``MvCameraControl_class.py``：在 ``<MVS>/Development/Samples/Python/MvImport`` 下，
  按 ``MVCAM_SDK_PATH`` 和常见安装位置找（见 ``_sdk_dirs``）。
* 运行库 ``MvCameraControl.dll``（Linux 上 ``libMvCameraControl.so``）：在
  ``<Common Files>/MVS/Runtime/Win64_x64`` 下，按 ``MVCAM_COMMON_RUNENV`` 和常见位置找，
  再挂进 DLL 搜索路径（见 ``_open_runtime_search_path``——MVS 只把它加进 PATH，而
  Windows 上 Python 3.8 起加载 DLL 不认 PATH，所以必须自己挂）。

MVS 可以装在任意路径；两半都能在相机参数的 ``sdk_path`` 里直接指定（填哪一半都认）。

装不到 Python 示例时改用 GenICam 方式（``kind=genicam``）：cti 在
``<Common Files>/MVS/Runtime/Win64_x64/`` 下，GigE 相机用 ``MvProducerGEV.cti``，USB3 用 ``MvProducerU3V.cti``。
"""
from __future__ import annotations

import ctypes
import logging
import os
import platform
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


#: 运行库（MvCameraControl.dll / libMvCameraControl.so）所在目录的候选根。MVS 把运行库装在
#: Common Files 下，和上面的 MvImport 不在同一棵树上（环境变量 MVCAM_COMMON_RUNENV 指到
#: Runtime 这一级），所以要单独找一遍。
_RUNTIME_ROOTS = [
    os.environ.get("MVCAM_COMMON_RUNENV", ""),
    *(os.path.join(v, "MVS", "Runtime") for v in
      (os.environ.get("CommonProgramFiles(x86)", ""), os.environ.get("CommonProgramW6432", ""),
       os.environ.get("CommonProgramFiles", "")) if v),
    r"C:\Program Files (x86)\Common Files\MVS\Runtime",
    r"C:\Program Files\Common Files\MVS\Runtime",
    "/opt/MVS/lib",
]

_LIB_NAME = "MvCameraControl.dll" if os.name == "nt" else "libMvCameraControl.so"


def _runtime_subdirs() -> list[str]:
    """运行库按平台分的子目录，当前解释器的架构排在前面。

    架构不对不是「找不到」而是加载失败（Windows 上是 WinError 193），所以候选宁可多排几个，
    由 _runtime_dirs 逐个确认文件在不在、_open_runtime_search_path 逐个试着加载。
    """
    if os.name == "nt":
        return ["Win64_x64", "Win32_i86"] if sys.maxsize > 2 ** 32 else ["Win32_i86", "Win64_x64"]
    m = platform.machine().lower()
    if m in ("aarch64", "arm64"):
        return ["aarch64", "64"]
    if m.startswith("arm"):
        return ["arm", "32"]
    return ["64", "32"] if sys.maxsize > 2 ** 32 else ["32", "64"]


def _runtime_dirs(extra: str = "") -> list[str]:
    """确实放着运行库的目录；extra 来自相机参数 sdk_path（有人把 dll 和 MvImport 放一起）。"""
    out: list[str] = []
    for root in ([extra] if extra else []) + _RUNTIME_ROOTS:
        if not root:
            continue
        for sub in _runtime_subdirs() + [""]:
            d = os.path.join(root, sub) if sub else root
            if d not in out and os.path.isfile(os.path.join(d, _LIB_NAME)):
                out.append(d)
    return out


_runtime_tried: set[str] = set()


def _open_runtime_search_path(extra: str = "") -> None:
    r"""把 MVS 运行库目录挂进 DLL 搜索路径，再按绝对路径预加载一次。

    MvImport 里是**按名字**加载运行库的（``WinDLL("MvCameraControl.dll")``），而 MVS 装完
    只把运行库目录加进了 PATH。Windows 上 Python 3.8 起 LoadLibraryEx 带的是
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS，**不再搜 PATH**，于是：

        FileNotFoundError: Could not find module 'MvCameraControl.dll'

    打包成 exe 之后同一句话被 PyInstaller 的 ctypes 钩子包了一层（它只去 _internal 里找，
    找不到就原样再试一次），报出来是 PyInstallerImportError——是同一件事，和「装没装 MVS」无关。

    add_dll_directory 管带标志位的 LoadLibraryEx，PATH 管朴素的 LoadLibraryA；之后按绝对路径
    加载一次，MvImport 那句按名字的加载就直接命中进程里已有的模块，厂商的 .py 一个字都不用改。
    """
    for d in _runtime_dirs(extra):
        if d in _runtime_tried:
            continue
        _runtime_tried.add(d)
        if os.name == "nt":
            try:
                os.add_dll_directory(d)           # 句柄故意不释放：整个进程期间都要有效
            except (AttributeError, OSError):     # pragma: no cover - 取决于系统
                pass
            path = os.environ.get("PATH", "")
            if d.lower() not in path.lower():
                os.environ["PATH"] = d + os.pathsep + path
        lib = os.path.join(d, _LIB_NAME)
        try:
            if os.name == "nt":
                ctypes.WinDLL(lib)
            else:
                ctypes.CDLL(lib)
        except OSError as e:                      # pragma: no cover - 取决于本机安装
            log.debug("预加载 %s 失败（位数/架构对不上？）：%s", lib, e)
            continue
        log.info("MVS 运行库：%s", lib)
        return


_PIX_MONO8 = 0x01080001
_PIX_MONO10 = 0x01100003
_PIX_MONO12 = 0x01100005
_PIX_BAYER = {0x01080008: "GR", 0x01080009: "RG", 0x0108000A: "GB", 0x0108000B: "BG"}
_PIX_RGB8 = 0x02180014
_PIX_BGR8 = 0x02180015

_mv = None


def _import_sdk() -> tuple[object | None, str]:
    """试着 ``import MvCameraControl_class``，返回（模块，失败原因）。

    导入失败不只有「模块没找到」一种：模块找到了而运行库没加载上时抛的是 OSError
    （打包版是 PyInstaller 包装过的 PyInstallerImportError，同样是 OSError 的子类），
    版本对不上的 .py 还能抛出别的。这些都不该越过 sdk_available() 把界面带崩，
    所以一律收敛成一句原因，由 load_sdk 拼成 CameraError。
    """
    try:
        import MvCameraControl_class as mv  # type: ignore
        return mv, ""
    except ModuleNotFoundError as e:
        return None, "" if e.name == "MvCameraControl_class" else str(e)
    except Exception as e:                         # pragma: no cover - 取决于本机安装
        return None, f"{type(e).__name__}: {e}"


def _not_found_hint(tried: list[str], why: str, extra: str = "") -> str:
    """装不上时给一句能照着做的提示：分「模块没找到」和「运行库没加载上」两种。"""
    if not why:
        looked = "\n  ".join(tried) if tried else "（没有一个候选目录存在）"
        return ("未找到海康 MVS SDK 的 Python 模块 MvCameraControl_class。请先安装 MVS（含 Python 示例），"
                "再把 MvImport 目录填进相机参数的 sdk_path，或设成环境变量 MVCAM_SDK_PATH；"
                "典型位置是 <MVS 安装目录>\\Development\\Samples\\Python\\MvImport。"
                f"\n找过这些目录：\n  {looked}")
    rt = _runtime_dirs(extra)
    where = "\n  ".join(rt) if rt else f"（候选目录里没有一个有 {_LIB_NAME}）"
    return (f"海康 MVS SDK 的 Python 模块导入失败：{why}\n"
            f"多半是运行库 {_LIB_NAME} 没加载上：请确认 MVS 和本软件位数一致（都是 64 位），"
            "并把运行库目录设进环境变量 MVCAM_COMMON_RUNENV（装 MVS 时本该自动设好），"
            "典型位置是 C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64；"
            "也可以把这个目录直接填进相机参数的 sdk_path。"
            f"\n找到的运行库目录：\n  {where}")


def load_sdk(extra_dir: str = ""):
    """导入 MvCameraControl_class；找不到时抛出带安装提示的 CameraError。"""
    global _mv
    if _mv is not None:
        return _mv
    _open_runtime_search_path(extra_dir)       # 先挂运行库：模块一导入就会去加载它
    tried = _sdk_dirs(extra_dir)
    mv, why = _import_sdk()
    if mv is None:
        for d in tried:
            if d not in sys.path:
                sys.path.append(d)
            mv, err = _import_sdk()
            if mv is not None:
                break
            why = err or why
    if mv is None:
        raise CameraError(_not_found_hint(tried, why, extra_dir))
    _mv = mv
    return mv


def sdk_status() -> tuple[bool, str]:
    """(能不能用, 用不了的原因)。界面和 ``cvflow cameras`` 都用它，免得各写一份提示——
    「没装 MVS」和「装了 MVS 但运行库没加载上」要让用户当场分得清。"""
    try:
        load_sdk()
        return True, ""
    except CameraError as e:
        return False, str(e)


def sdk_available() -> bool:
    return sdk_status()[0]


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

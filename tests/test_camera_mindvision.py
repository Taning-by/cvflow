"""迈德威视（MindVision）相机后端的黑盒测试。

手上没有这台相机，但它的 SDK 接口是固定的，所以这里**造一个假的 mvsdk 模块**顶上去，
把整条路跑一遍：选相机、设曝光/增益/触发、取图、翻转、转成 BGR/灰度、关相机。
真机上还可能有别的意外，但"代码这边对不对"能在这里验完。
"""
from __future__ import annotations

import ctypes
import sys
import types

import numpy as np
import pytest

from cvflow.camera.base import CameraError


class FakeDev:
    def __init__(self, sn, name, model):
        self._sn, self._name, self._model = sn, name, model

    def GetSn(self):
        return self._sn.encode()

    def GetFriendlyName(self):
        return self._name.encode()

    def GetProductModel(self):
        return self._model.encode()

    def GetPortType(self):
        return b"USB3.0"


class FakeHead:
    def __init__(self, w, h, ch):
        self.iWidth, self.iHeight = w, h
        self.uBytes = w * h * ch


def make_fake_sdk(mono=False, w=8, h=4, fail_timeout=False):
    """造一个最小可用的 mvsdk 替身，记录调用供断言。"""
    sdk = types.ModuleType("mvsdk")
    calls: list = []
    ch = 1 if mono else 3
    # 每个像素写上行号，翻转与否一眼能看出来
    frame = np.zeros((h, w, ch), dtype=np.uint8)
    for y in range(h):
        frame[y, :, :] = y + 1
    store = {"buf": None, "data": frame.tobytes()}

    class CameraException(Exception):
        def __init__(self, code, msg="fake"):
            super().__init__(msg)
            self.error_code = code

    sdk.CameraException = CameraException
    sdk.CAMERA_STATUS_TIME_OUT = -12
    sdk.CAMERA_MEDIA_TYPE_MONO8 = 0x01080001
    sdk.CAMERA_MEDIA_TYPE_BGR8 = 0x02180015
    sdk.c_ubyte = ctypes.c_ubyte

    sdk.CameraEnumerateDevice = lambda: [FakeDev("SN001", "相机甲", "MV-A"), FakeDev("SN002", "相机乙", "MV-B")]

    def CameraInit(dev, a, b):
        calls.append(("init", dev.GetSn().decode()))
        return 42

    sdk.CameraInit = CameraInit
    cap = types.SimpleNamespace(
        sIspCapacity=types.SimpleNamespace(bMonoSensor=1 if mono else 0),
        sResolutionRange=types.SimpleNamespace(iWidthMax=w, iHeightMax=h))
    sdk.CameraGetCapability = lambda hc: cap
    sdk.CameraSetIspOutFormat = lambda hc, fmt: calls.append(("fmt", fmt))

    def CameraAlignMalloc(size, align):
        store["buf"] = ctypes.create_string_buffer(store["data"], size)
        return ctypes.addressof(store["buf"])

    sdk.CameraAlignMalloc = CameraAlignMalloc
    sdk.CameraAlignFree = lambda p: calls.append(("free", p))
    sdk.CameraPlay = lambda hc: calls.append(("play",))
    sdk.CameraUnInit = lambda hc: calls.append(("uninit",))
    sdk.CameraSetTriggerMode = lambda hc, m: calls.append(("trigger_mode", m))
    sdk.CameraSetAeState = lambda hc, v: calls.append(("ae", v))
    sdk.CameraSetExposureTime = lambda hc, v: calls.append(("exposure", v))
    sdk.CameraSetAnalogGainX = lambda hc, v: calls.append(("gain", v))
    sdk.CameraSoftTrigger = lambda hc: calls.append(("soft_trigger",))

    def CameraGetImageBuffer(hc, ms):
        if fail_timeout:
            raise CameraException(sdk.CAMERA_STATUS_TIME_OUT, "timeout")
        calls.append(("get", ms))
        return 1234, FakeHead(w, h, ch)

    sdk.CameraGetImageBuffer = CameraGetImageBuffer
    sdk.CameraImageProcess = lambda hc, raw, buf, head: calls.append(("process",))
    sdk.CameraReleaseImageBuffer = lambda hc, raw: calls.append(("release",))
    sdk.CameraFlipFrameBuffer = lambda buf, head, mode: calls.append(("flip", mode))
    sdk.calls = calls
    return sdk


@pytest.fixture
def fake_sdk(monkeypatch):
    """把假 mvsdk 塞进 sys.modules，并清掉模块里缓存的句柄。"""
    def install(**kw):
        sdk = make_fake_sdk(**kw)
        monkeypatch.setitem(sys.modules, "mvsdk", sdk)
        from cvflow.camera import mindvision_cam
        monkeypatch.setattr(mindvision_cam, "_sdk", None)
        return sdk
    return install


def make_cam(**cfg):
    from cvflow.camera.mindvision_cam import MindVisionCamera
    return MindVisionCamera("cam0", cfg)


def test_opens_grabs_and_closes(fake_sdk):
    """一条龙：打开 → 取一帧彩色图 → 关闭。"""
    sdk = fake_sdk()
    cam = make_cam()
    cam.open()
    assert cam.is_open
    img = cam.grab(1.0)
    assert img is not None and img.data.shape == (4, 8, 3)     # BGR
    assert img.data[0, 0, 0] == 1 and img.data[3, 0, 0] == 4   # 行号对得上，没有错位
    cam.close()
    assert not cam.is_open
    kinds = [c[0] for c in sdk.calls]
    assert kinds.count("init") == 1 and "play" in kinds and "uninit" in kinds
    assert "release" in kinds                                  # 每帧都要还回去，否则缓冲会耗尽
    assert "free" in kinds                                     # 对齐分配的缓冲也要释放


def test_mono_camera_gives_single_channel(fake_sdk):
    fake_sdk(mono=True)
    cam = make_cam()
    cam.open()
    img = cam.grab(1.0)
    assert img is not None and img.data.ndim == 2 and img.data.shape == (4, 8)
    cam.close()


def test_selects_camera_by_serial_name_model_or_index(fake_sdk):
    """source 支持序列号 / 友好名 / 型号 / 索引，和海康那个后端保持一致。"""
    for want, expect in (("SN002", "SN002"), ("相机乙", "SN002"), ("MV-B", "SN002"),
                         ("1", "SN002"), ("", "SN001"), ("0", "SN001")):
        sdk = fake_sdk()
        cam = make_cam(source=want)
        cam.open()
        assert ("init", expect) in sdk.calls, f"source={want!r} 选错了相机"
        cam.close()


def test_unknown_camera_says_what_is_available(fake_sdk):
    fake_sdk()
    cam = make_cam(source="不存在的相机")
    with pytest.raises(CameraError) as e:
        cam.open()
    assert "相机甲" in str(e.value) and "相机乙" in str(e.value)   # 报错里列出可用的


def test_settings_map_to_the_vendor_api(fake_sdk):
    """曝光/增益/触发要映射到 mvsdk 自己的函数，而不是 GenICam 那套特性名。"""
    sdk = fake_sdk()
    cam = make_cam(exposure_us=5000, gain=2.5, trigger_mode="software")
    cam.open()
    assert ("trigger_mode", 1) in sdk.calls                     # 0=连续 1=软触发 2=硬触发
    assert ("exposure", 5000.0) in sdk.calls
    assert ("gain", 2.5) in sdk.calls
    assert ("ae", 0) in sdk.calls                               # 设曝光前必须关自动曝光
    cam.software_trigger()
    assert ("soft_trigger",) in sdk.calls
    cam.close()


def test_hardware_and_free_run_trigger_modes(fake_sdk):
    for mode, want in (("hardware", 2), ("off", 0)):
        sdk = fake_sdk()
        cam = make_cam(trigger_mode=mode)
        cam.open()
        assert ("trigger_mode", want) in sdk.calls
        cam.close()


def test_keep_does_not_touch_the_trigger_mode(fake_sdk):
    """keep＝沿用相机当前设置，不该去改它。"""
    sdk = fake_sdk()
    cam = make_cam(trigger_mode="keep")
    cam.open()
    assert not any(c[0] == "trigger_mode" for c in sdk.calls)
    cam.close()


def test_timeout_returns_none_instead_of_raising(fake_sdk):
    """取图超时是产线常态（等硬触发），要返回 None 让上层处理，不能抛异常。"""
    fake_sdk(fail_timeout=True)
    cam = make_cam()
    cam.open()
    assert cam.grab(0.1) is None
    cam.close()


def test_image_is_flipped_only_on_windows(fake_sdk, monkeypatch):
    """SDK 在 Windows 上输出自下而上的位图，不翻转整幅图是倒的；Linux 上不该翻。"""
    import platform
    for system, flipped in (("Windows", True), ("Linux", False)):
        sdk = fake_sdk()
        monkeypatch.setattr(platform, "system", lambda s=system: s)
        cam = make_cam()
        cam.open()
        cam.grab(1.0)
        assert any(c[0] == "flip" for c in sdk.calls) is flipped, system
        cam.close()


def test_missing_sdk_explains_how_to_install(monkeypatch):
    """没装 SDK 时报错要说清楚去哪儿找 mvsdk.py。"""
    from cvflow.camera import mindvision_cam
    monkeypatch.setattr(mindvision_cam, "_sdk", None)
    monkeypatch.setitem(sys.modules, "mvsdk", None)           # 让 import mvsdk 失败
    monkeypatch.setattr(mindvision_cam, "_SDK_DIRS", ["/不存在的目录"])
    with pytest.raises(CameraError) as e:
        mindvision_cam.load_sdk()
    msg = str(e.value)
    assert "mvsdk" in msg and "MVSDK_PATH" in msg and "sdk_path" in msg


def test_enumerate_lists_cameras_for_the_search_dialog(fake_sdk):
    """"搜索相机"要能列出迈德威视的设备。"""
    fake_sdk()
    from cvflow.camera.mindvision_cam import enumerate_mindvision
    devs = enumerate_mindvision()
    assert [d.serial for d in devs] == ["SN001", "SN002"]
    assert devs[0].user_name == "相机甲" and devs[0].source == "mindvision"
    assert devs[0].extra["port"] == "USB3.0"


def test_registered_as_a_camera_kind():
    from cvflow.camera.manager import CAMERA_KINDS
    assert "mindvision" in CAMERA_KINDS

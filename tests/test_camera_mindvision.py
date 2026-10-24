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
    sdk.CameraLoadParameter = lambda hc, team: calls.append(("load_team", team))
    sdk.CameraReadParameterFromFile = lambda hc, path: calls.append(("load_file", path))
    sdk.CameraSetWbMode = lambda hc, auto: calls.append(("wb_mode", bool(auto)))
    sdk.CameraSetOnceWB = lambda hc: calls.append(("wb_once",))
    sdk.CameraSetGain = lambda hc, r, g, b: calls.append(("wb_gain", (r, g, b)))

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
    monkeypatch.setattr(mindvision_cam, "_SDK_ROOTS", ["/不存在的目录"])
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


# ==================================================== OpenCV 后端选择与诊断命令
def test_opencv_backend_names_map_to_opencv_constants():
    """采集后端的名字要对上 OpenCV 的常量：Windows 默认走 MSMF，只认 UVC；
    厂商注册的 DirectShow 滤镜（迈德威视的 MindVision.ax）只有 dshow 看得见。"""
    import cv2
    from cvflow.camera.opencv_cam import BACKENDS
    assert BACKENDS["dshow"] == getattr(cv2, "CAP_DSHOW", 700)
    assert BACKENDS["msmf"] == getattr(cv2, "CAP_MSMF", 1400)
    assert BACKENDS["auto"] == 0


def test_opencv_rejects_unknown_backend():
    from cvflow.camera.manager import create_camera
    cam = create_camera("opencv", "c", {"source": 0, "backend": "没这个后端"})
    with pytest.raises(CameraError) as e:
        cam.open()
    assert "未知的采集后端" in str(e.value)


def test_opencv_failure_hints_at_dshow(monkeypatch):
    """用默认后端按设备号打不开时，要提示可能得换 dshow——这正是迈德威视相机的情形。"""
    import cv2
    from cvflow.camera.manager import create_camera

    class Dead:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: Dead())
    cam = create_camera("opencv", "c", {"source": 0, "backend": "auto"})
    with pytest.raises(CameraError) as e:
        cam.open()
    assert "dshow" in str(e.value)


def test_cameras_command_reports_every_route(capsys, monkeypatch):
    """cvflow cameras 要把每条取图路线分别报一遍，并在缺 mvsdk 时说清它在哪个包里。"""
    from cvflow import cli
    from cvflow.camera import mindvision_cam
    monkeypatch.setattr(mindvision_cam, "sdk_available", lambda: False)
    monkeypatch.setattr("cvflow.camera.opencv_cam.probe_indices", lambda b, n: [] if b != "auto" else [0])
    assert cli.main(["cameras", "--probe", "2"]) == 0
    out = capsys.readouterr().out
    assert "OpenCV" in out and "迈德威视" in out and "海康" in out
    assert "设备号 0" in out                                  # 探到的设备号要列出来
    assert "mvsdk" in out and "Python Demo" in out            # 告诉用户去哪个包里找


# ==================================================== 成像参数：存档与白平衡
def test_loads_the_saved_parameter_group(fake_sdk):
    """不加载厂商软件存好的那组参数，相机就停在出厂白平衡上，整幅图偏色。"""
    for team, idx in (("A", 0), ("B", 1), ("C", 2), ("D", 3)):
        sdk = fake_sdk()
        cam = make_cam(param_group=team)
        cam.open()
        assert ("load_team", idx) in sdk.calls, team
        cam.close()


def test_parameter_group_keep_loads_nothing(fake_sdk):
    sdk = fake_sdk()
    cam = make_cam(param_group="keep")
    cam.open()
    assert not any(c[0] == "load_team" for c in sdk.calls)
    cam.close()


def test_loads_the_parameter_file(fake_sdk, tmp_path):
    f = tmp_path / "cam.config"
    f.write_text("fake", encoding="utf-8")
    sdk = fake_sdk()
    cam = make_cam(param_file=str(f))
    cam.open()
    assert ("load_file", str(f)) in sdk.calls
    cam.close()


def test_missing_parameter_file_says_where_to_export_it(fake_sdk, tmp_path):
    fake_sdk()
    cam = make_cam(param_file=str(tmp_path / "没有这个文件.config"))
    with pytest.raises(CameraError) as e:
        cam.open()
    assert "保存当前配置参数到指定文件" in str(e.value)


def test_saved_params_load_before_format_and_exposure(fake_sdk, tmp_path):
    """存档里也带输出格式和曝光，所以必须先加载、再设我们自己的，否则会被覆盖回去。"""
    f = tmp_path / "cam.config"
    f.write_text("fake", encoding="utf-8")
    sdk = fake_sdk()
    cam = make_cam(param_group="A", param_file=str(f), exposure_us=3000)
    cam.open()
    order = [c[0] for c in sdk.calls]
    assert order.index("load_team") < order.index("load_file") < order.index("fmt")
    assert order.index("load_file") < order.index("exposure")
    cam.close()


def test_white_balance_modes(fake_sdk):
    sdk = fake_sdk()
    cam = make_cam(wb_mode="auto")
    cam.open()
    assert ("wb_mode", True) in sdk.calls and not any(c[0] == "wb_once" for c in sdk.calls)
    cam.close()

    sdk = fake_sdk()
    cam = make_cam(wb_mode="once")
    cam.open()
    # 一次性白平衡之前要先关连续自动，否则下一帧就被算回去了
    assert sdk.calls.index(("wb_mode", False)) < sdk.calls.index(("wb_once",))
    cam.close()


def test_keep_does_not_touch_white_balance(fake_sdk):
    sdk = fake_sdk()
    cam = make_cam(wb_mode="keep")
    cam.open()
    assert not any(c[0] in ("wb_mode", "wb_once", "wb_gain") for c in sdk.calls)
    cam.close()


def test_manual_white_balance_gain_is_in_multiples(fake_sdk):
    """参数里一律填倍数（和 GenICam 的 BalanceRatio 同单位），到了 SDK 才换成 ×100 的刻度。"""
    for text, want in (("1.0,1.0,1.6", (100, 100, 160)),
                       ("1, 1, 1.6", (100, 100, 160)),
                       ("1.0 1.0 1.6", (100, 100, 160))):
        sdk = fake_sdk()
        cam = make_cam(wb_mode="manual", wb_gain=text)
        cam.open()
        assert ("wb_gain", want) in sdk.calls, text
        assert sdk.calls.index(("wb_mode", False)) < sdk.calls.index(("wb_gain", want))
        cam.close()


def test_gain_in_vendor_scale_is_refused_with_the_conversion(fake_sdk):
    """照抄厂商软件里的 160 会把画面打爆，要当场拒掉并说清怎么换算。"""
    fake_sdk()
    cam = make_cam(wb_mode="manual", wb_gain="100,100,160")
    with pytest.raises(CameraError) as e:
        cam.open()
    assert "除以 100" in str(e.value)


def test_manual_white_balance_rejects_bad_gain(fake_sdk):
    for bad in ("", "1.0,1.6", "红,绿,蓝", "-1,1,1"):
        fake_sdk()
        cam = make_cam(wb_mode="manual", wb_gain=bad)
        with pytest.raises(CameraError):
            cam.open()


def test_mono_camera_skips_white_balance(fake_sdk):
    """黑白相机没有白平衡这回事，设了反而可能报错。"""
    sdk = fake_sdk(mono=True)
    cam = make_cam(wb_mode="once")
    cam.open()
    assert not any(c[0] in ("wb_mode", "wb_once") for c in sdk.calls)
    cam.close()


def test_camera_node_passes_the_imaging_params_through(tmp_path):
    """这些参数得真从节点传到相机配置里，否则界面上填了也没用。"""
    from cvflow.operators.source import CameraSource
    f = tmp_path / "cam.config"
    f.write_text("fake", encoding="utf-8")
    node = CameraSource("n1")
    node.set("kind", "mindvision")
    node.set("param_group", "A")
    node.set("param_file", str(f))
    node.set("wb_mode", "manual")
    node.set("wb_gain", "1.0,1.0,1.6")
    cfg = node._config()
    assert cfg["param_group"] == "A" and cfg["param_file"] == str(f)
    assert cfg["wb_mode"] == "manual" and cfg["wb_gain"] == "1.0,1.0,1.6"


# ==================================================== 白平衡是通用能力，不是哪一家专有
def test_white_balance_on_a_genicam_style_camera():
    """海康 / GenTL 走的是 SFNC 特性名，基类里实现，和迈德威视无关。"""
    from cvflow.camera.base import Camera

    class Fake(Camera):
        kind = "fake"

        def __init__(self):
            super().__init__("c", {})
            self.feats: list = []

        def open(self): ...
        def close(self): ...
        def _grab_raw(self, t): ...
        def set_feature(self, name, value):
            self.feats.append((name, value))

    cam = Fake()
    cam.apply_white_balance({"wb_mode": "auto"})
    assert ("BalanceWhiteAuto", "Continuous") in cam.feats

    cam = Fake()
    cam.apply_white_balance({"wb_mode": "once"})
    assert ("BalanceWhiteAuto", "Once") in cam.feats

    cam = Fake()
    cam.apply_white_balance({"wb_mode": "manual", "wb_gain": "1.0,1.0,1.6"})
    assert cam.feats[0] == ("BalanceWhiteAuto", "Off")      # 不先关自动就会被算回去
    assert ("BalanceRatioSelector", "Blue") in cam.feats and ("BalanceRatio", 1.6) in cam.feats

    cam = Fake()
    cam.apply_white_balance({"wb_mode": "keep"})
    assert cam.feats == []                                  # keep 不碰相机


def test_wb_gain_parser_rejects_the_vendor_scale():
    from cvflow.camera.base import parse_wb_gain
    assert parse_wb_gain("1.0,1.0,1.6") == (1.0, 1.0, 1.6)
    assert parse_wb_gain("") is None
    with pytest.raises(CameraError) as e:
        parse_wb_gain("100,100,160")
    assert "除以 100" in str(e.value)


# ==================================================== 参数按相机类型露面
def test_camera_params_are_filtered_by_camera_kind():
    """选了哪种相机就只留哪种用得上的参数——别让客户在一堆厂商专有项里找。"""
    from cvflow.operators.source import CameraSource
    node = CameraSource("n")
    names = lambda: {p.name for p in node.visible_params()}

    node.set("kind", "folder")
    assert not (names() & {"backend", "cti", "sdk_path", "user_set",
                           "param_group", "param_file", "wb_mode", "exposure_us"})

    node.set("kind", "mindvision")
    assert {"param_group", "param_file", "sdk_path", "wb_mode", "exposure_us"} <= names()
    assert not (names() & {"cti", "backend", "user_set"})   # 这几样迈德威视用不上

    node.set("kind", "hik")
    assert {"user_set", "sdk_path", "wb_mode"} <= names()
    assert not (names() & {"param_group", "param_file", "cti", "backend"})

    node.set("kind", "genicam")
    assert {"cti", "user_set", "features"} <= names()

    node.set("kind", "opencv")
    assert "backend" in names() and not (names() & {"cti", "sdk_path", "param_group"})


def test_hidden_params_still_have_values():
    """只是不显示，取值照常在——节点代码读到的永远是完整的一套。"""
    from cvflow.operators.source import CameraSource
    node = CameraSource("n")
    node.set("kind", "folder")
    assert "param_group" not in {p.name for p in node.visible_params()}
    assert node.get("param_group") == "keep"
    assert node._config()["param_group"] == "keep"


def test_changing_the_camera_kind_triggers_a_panel_rebuild():
    """改了相机类型，面板得知道要整块重建，否则藏起来的参数还挂在那儿。"""
    from cvflow.operators.source import CameraSource
    node = CameraSource("n")
    assert [p.name for p in node.params_watching("kind")]
    assert node.params_watching("exposure_us") == []


def test_enum_choices_are_shown_in_chinese_but_stored_raw():
    """下拉框翻译只管显示：存进方案文件的必须还是原始取值。"""
    from cvflow.ui.i18n import tr_choice
    assert tr_choice("wb_mode", "once") == "打开时做一次"
    assert tr_choice("kind", "mindvision") == "迈德威视"
    # 同名参数在不同节点上含义不同，按参数名分域查，不会串味
    assert tr_choice("method", "linear") == "线性拉伸"
    assert tr_choice("interpolation", "linear") == "双线性"
    assert tr_choice("mode", "list") == "全部（不分层）"
    # 表里没有的原样返回，插件自定义的选项不会出错
    assert tr_choice("没这个参数", "whatever") == "whatever"


def test_every_enum_choice_has_a_chinese_label():
    """新加 enum 选项忘了配翻译，这条会拦住——只放过符号和专有名词。"""
    import re
    import cvflow.operators  # noqa: F401
    from cvflow.core.registry import registry
    from cvflow.ui.i18n import tr_choice
    keep_as_is = {"BGR", "RGB", "HSV", "HLS", "Lab", "NCHW", "NHWC",
                  "PNG", "JPG", "BMP", "CPU", "TensorRT", "FFmpeg", "GStreamer"}
    missing = set()
    for cls in registry.all():
        if not cls.__module__.startswith("cvflow.operators"):
            continue                                        # 第三方插件自己的选项不强求
        for p in getattr(cls, "params", []):
            if p.kind != "enum":
                continue
            for c in p.choices:
                shown = tr_choice(p.name, str(c))
                if shown in keep_as_is or not re.search(r"[A-Za-z]{2}", shown):
                    continue
                if re.search(r"[一-鿿]", shown):     # 带中文说明就算翻过了
                    continue
                missing.add(f"{p.name}.{c}")
    assert not missing, f"这些选项还没有中文：{sorted(missing)}"

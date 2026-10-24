"""Camera abstraction. Nodes never talk to a vendor SDK directly; they ask a
``Camera`` for frames so the same flow runs against a folder of test images, a
USB webcam or a GigE Vision camera."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.types import Image


class CameraError(Exception):
    pass


#: 白平衡增益一律按**倍数**填（1.0 = 不动），GenICam 的 BalanceRatio 就是这个单位。
#: 厂商软件里常显示 ×100 的刻度（160 表示 1.6 倍），照抄过来会把画面打爆，所以超过这个
#: 上限直接拒掉并说清怎么换算——比悄悄猜一个刻度要安全。
WB_GAIN_MAX = 16.0


def parse_wb_gain(text: Any) -> tuple[float, float, float] | None:
    """把 ``"1.0,1.0,1.6"`` 解析成 R/G/B 三个增益倍数；留空返回 None。"""
    s = str(text or "").replace("，", ",").replace(";", ",").replace(" ", ",")
    parts = [p for p in s.split(",") if p]
    if not parts:
        return None
    if len(parts) != 3:
        raise CameraError(f"白平衡增益要填 R,G,B 三个数，收到 {text!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise CameraError(f"白平衡增益不是数字：{text!r}") from None
    if any(v < 0 for v in vals):
        raise CameraError(f"白平衡增益不能为负：{text!r}")
    if any(v > WB_GAIN_MAX for v in vals):
        raise CameraError(
            f"白平衡增益 {text!r} 超出范围。这里填的是倍数（1.0 = 不动）；"
            f"厂商软件里显示的是 ×100 的刻度，照抄要先除以 100（160 → 1.6）")
    return vals[0], vals[1], vals[2]


class Camera(ABC):
    kind: str = ""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.config: dict[str, Any] = dict(config or {})
        self.is_open = False
        self._frame_id = 0
        self._lock = threading.RLock()

    # ---- lifecycle ----
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def _grab_raw(self, timeout_s: float) -> np.ndarray | None:
        """Return one frame (HxW or HxWx3 uint8, BGR) or None on timeout."""

    def grab(self, timeout_s: float = 2.0) -> Image | None:
        with self._lock:
            if not self.is_open:
                self.open()
            data = self._grab_raw(timeout_s)
            if data is None:
                return None
            self._frame_id += 1
            if self.config.get("grayscale") and data.ndim == 3:
                import cv2
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            return Image(data=data, frame_id=self._frame_id, timestamp=time.time(), source=self.name)

    # ---- optional feature access (exposure, gain, trigger mode ...) ----
    def set_feature(self, name: str, value: Any) -> None:  # noqa: B027
        pass

    def get_feature(self, name: str) -> Any:
        return None

    def execute_command(self, name: str) -> None:
        """执行命令型特性（UserSetLoad、TriggerSoftware 这类）；约定 value=None 即命令。"""
        self.set_feature(name, None)

    def load_user_set(self, name: str) -> None:
        """把相机 Flash 里的一组参数恢复到工作寄存器。name 形如 UserSet1 / Default。"""
        try:
            self.set_feature("UserSetSelector", name)
            self.execute_command("UserSetLoad")
        except Exception as e:
            raise CameraError(f"相机 {self.name!r}：加载用户集 {name} 失败"
                              f"（这台相机可能没有这一组，或当前正在采集）：{e}") from e

    def save_user_set(self, name: str) -> None:
        """把当前参数写进相机 Flash。Flash 擦写有寿命，只在调试确认后手动调用，别放进流程。"""
        self.set_feature("UserSetSelector", name)
        self.execute_command("UserSetSave")

    def apply_settings(self, cfg: dict[str, Any]) -> None:
        """把通用设置映射到 GenICam SFNC 特性：用户集、曝光（微秒）、增益、触发模式/源。"""
        user_set = str(cfg.get("user_set") or "keep")
        if user_set != "keep":
            # 先恢复整套，后面的曝光/增益/触发才是在它之上的覆盖
            self.load_user_set(user_set)
        exp = cfg.get("exposure_us")
        if exp not in (None, "", 0, 0.0):
            self.set_feature("ExposureAuto", "Off")
            self.set_feature("ExposureTime", float(exp))
        gain = cfg.get("gain")
        if gain not in (None, "") and float(gain) >= 0:
            self.set_feature("GainAuto", "Off")
            self.set_feature("Gain", float(gain))
        self.apply_white_balance(cfg)
        mode = cfg.get("trigger_mode", "keep")
        if mode == "off":
            self.set_feature("TriggerMode", "Off")
        elif mode == "software":
            self.set_feature("TriggerMode", "On")
            self.set_feature("TriggerSource", "Software")
        elif mode == "hardware":
            self.set_feature("TriggerMode", "On")
            self.set_feature("TriggerSource", str(cfg.get("trigger_source") or "Line0"))

    def apply_white_balance(self, cfg: dict[str, Any]) -> None:
        """白平衡，走 GenICam SFNC 的 BalanceWhiteAuto / BalanceRatio。

        彩色相机不做白平衡就会整幅偏色，这是所有彩色工业相机的共性，不是哪一家的专利，
        所以放在基类里；厂商 SDK 不认这套特性名的（迈德威视）自己覆盖掉。
        """
        mode = str(cfg.get("wb_mode") or "keep")
        if mode == "keep":
            return
        if mode == "auto":
            self.set_feature("BalanceWhiteAuto", "Continuous")
            return
        if mode == "once":
            # 画面里得有白面或灰面，不然算出来的增益是歪的
            self.set_feature("BalanceWhiteAuto", "Once")
            return
        gains = parse_wb_gain(cfg.get("wb_gain"))
        if gains is None:
            raise CameraError(f"相机 {self.name!r}：白平衡选了手动，但没填增益（形如 1.0,1.0,1.6）")
        self.set_feature("BalanceWhiteAuto", "Off")        # 不关自动，设完下一帧就被算回去
        for sel, val in zip(("Red", "Green", "Blue"), gains):
            self.set_feature("BalanceRatioSelector", sel)
            self.set_feature("BalanceRatio", val)

    def software_trigger(self) -> None:  # noqa: B027
        """软触发一帧（触发模式为 software 时由相机节点在取图前调用）。"""

    def test_connection(self) -> tuple[bool, str]:
        """连接测试：默认尝试打开再关闭。"""
        try:
            was_open = self.is_open
            if not was_open:
                self.open()
            if not was_open:
                self.close()
            return True, f"相机 {self.name!r}（{self.kind}）可以打开"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "open": self.is_open, "frames": self._frame_id}

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

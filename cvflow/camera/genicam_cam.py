"""GenICam / GigE Vision / USB3 Vision camera via the `harvesters` package.

Requires a GenTL producer (.cti) from any vendor (MVTec, Matrix Vision, Basler,
Hikrobot MVS ships one too). Install with ``pip install harvesters``.
"""
from __future__ import annotations

import numpy as np

from .base import Camera, CameraError

_BAYER = {
    "BayerRG8": "COLOR_BayerRG2BGR", "BayerGR8": "COLOR_BayerGR2BGR",
    "BayerGB8": "COLOR_BayerGB2BGR", "BayerBG8": "COLOR_BayerBG2BGR",
}


class GenICamCamera(Camera):
    kind = "genicam"

    def __init__(self, name: str, config=None) -> None:
        super().__init__(name, config)
        self._h = None
        self._ia = None

    def open(self) -> None:
        try:
            from harvesters.core import Harvester
        except ImportError as e:  # pragma: no cover - optional dependency
            raise CameraError("GenICam 支持需要 `pip install harvesters` 以及厂商的 GenTL .cti 驱动") from e
        cti = self.config.get("cti")
        if not cti:
            raise CameraError("GenICam 相机：必须配置 'cti'（GenTL 驱动路径）")
        h = Harvester()
        h.add_file(str(cti))
        h.update()
        if not h.device_info_list:
            raise CameraError("GenICam：未发现设备")
        sel = self.config.get("source", 0)
        kwargs = {"serial_number": str(sel)} if isinstance(sel, str) and not sel.isdigit() else {"list_index": int(sel)}
        ia = h.create(**kwargs)
        node_map = ia.remote_device.node_map
        for feat, val in (self.config.get("features") or {}).items():  # e.g. ExposureTime, TriggerMode
            try:
                setattr(node_map, feat, val)
            except Exception as e:
                raise CameraError(f"GenICam：无法设置 {feat}={val!r}：{e}") from e
        self._h, self._ia = h, ia
        self.apply_settings(self.config)
        ia.start()
        self.is_open = True

    def close(self) -> None:
        try:
            if self._ia is not None:
                self._ia.stop()
                self._ia.destroy()
            if self._h is not None:
                self._h.reset()
        finally:
            self._ia = self._h = None
            self.is_open = False

    def _grab_raw(self, timeout_s: float) -> np.ndarray | None:
        assert self._ia is not None
        try:
            with self._ia.fetch(timeout=timeout_s) as buffer:
                comp = buffer.payload.components[0]
                w, h = comp.width, comp.height
                fmt = str(comp.data_format)
                data = np.asarray(comp.data)
                if fmt in ("Mono8",):
                    return data.reshape(h, w).copy()
                if fmt in ("RGB8", "RGB8Packed"):
                    import cv2
                    return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                if fmt in ("BGR8", "BGR8Packed"):
                    return data.reshape(h, w, 3).copy()
                if fmt in _BAYER:
                    import cv2
                    return cv2.cvtColor(data.reshape(h, w), getattr(cv2, _BAYER[fmt]))
                if fmt.startswith("Mono1") and data.dtype != np.uint8:  # Mono10/12/16 -> 8 bit
                    bits = int("".join(c for c in fmt[4:] if c.isdigit()) or 16)
                    return (data.reshape(h, w) >> (bits - 8)).astype(np.uint8)
                raise CameraError(f"GenICam：不支持的像素格式 {fmt}")
        except Exception as e:
            if "timeout" in str(e).lower():
                return None
            raise

    def set_feature(self, name: str, value) -> None:
        if self._ia is not None:
            setattr(self._ia.remote_device.node_map, name, value)

    def get_feature(self, name: str):
        if self._ia is None:
            return None
        return getattr(self._ia.remote_device.node_map, name).value

    def software_trigger(self) -> None:
        if self._ia is not None:
            self._ia.remote_device.node_map.TriggerSoftware.execute()


def enumerate_genicam(cti: str) -> list:
    """通过 GenTL 驱动枚举相机（需要 harvesters）。"""
    from .discovery import GigEDevice
    try:
        from harvesters.core import Harvester
    except ImportError as e:
        raise CameraError("GenICam 支持需要 `pip install harvesters`") from e
    h = Harvester()
    h.add_file(str(cti))
    h.update()
    out = []
    for i, d in enumerate(h.device_info_list):
        out.append(GigEDevice(ip=getattr(d, "ip_address", "") or "", mac="", manufacturer=getattr(d, "vendor", "") or "",
                              model=getattr(d, "model", "") or "", version=getattr(d, "version", "") or "",
                              serial=getattr(d, "serial_number", "") or "", user_name=getattr(d, "user_defined_name", "") or "",
                              source="genicam", extra={"index": i, "cti": str(cti)}))
    h.reset()
    return out

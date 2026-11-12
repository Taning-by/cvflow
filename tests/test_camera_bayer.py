"""Bayer 去马赛克的颜色方向：海康和 GenICam 后端都要把红还原成红。

OpenCV 的 Bayer 常量和 GenICam 的像素格式名错开一个像素，同名套用会红蓝反色。
这里按 GenICam 定义（格式名 = 左上 2x2 的排列）从已知颜色的图采样出马赛克，再看转换结果。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from cvflow.camera import hik_cam
from cvflow.camera.genicam_cam import _BAYER

# 左上 2x2 的颜色排列（GenICam / MVS 的定义）
_LAYOUT = {"RG": "RGGB", "GR": "GRBG", "GB": "GBRG", "BG": "BGGR"}
_HIK_CODE = {"GR": 0x01080008, "RG": 0x01080009, "GB": 0x0108000A, "BG": 0x0108000B}
_CH = {"B": 0, "G": 1, "R": 2}

# 三块纯色（BGR）：红、绿、蓝
_COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}


def _scene() -> np.ndarray:
    img = np.zeros((32, 96, 3), np.uint8)
    for i, bgr in enumerate(_COLORS.values()):
        img[:, i * 32:(i + 1) * 32] = bgr
    return img


def _mosaic(img: np.ndarray, pattern: str) -> np.ndarray:
    layout = _LAYOUT[pattern]
    out = np.empty(img.shape[:2], np.uint8)
    for k, c in enumerate(layout):
        dy, dx = divmod(k, 2)
        out[dy::2, dx::2] = img[dy::2, dx::2, _CH[c]]
    return out


def _assert_colors(bgr: np.ndarray) -> None:
    for i, (name, want) in enumerate(_COLORS.items()):
        got = bgr[16, i * 32 + 16]       # 色块中心，远离边界插值
        assert tuple(int(v) for v in got) == want, f"{name}: 得到 BGR {tuple(got)}，应为 {want}"


@pytest.mark.parametrize("pattern", ["RG", "GR", "GB", "BG"])
def test_hik_bayer_keeps_red_red(pattern):
    raw = _mosaic(_scene(), pattern)
    out = hik_cam._convert(raw.ravel(), raw.shape[1], raw.shape[0], _HIK_CODE[pattern])
    _assert_colors(out)


@pytest.mark.parametrize("pattern", ["RG", "GR", "GB", "BG"])
def test_genicam_bayer_keeps_red_red(pattern):
    raw = _mosaic(_scene(), pattern)
    out = cv2.cvtColor(raw, getattr(cv2, _BAYER[f"Bayer{pattern}8"]))
    _assert_colors(out)

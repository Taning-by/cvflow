"""界面文字表。

``tr(text)`` 把代码中的英文源字符串映射为中文；表里没有的字符串原样返回，
因此动态拼接的文字不会出错。全部翻译集中在 ``cvflow/ui/i18n_zh_CN.py``。
"""
from __future__ import annotations

from .i18n_zh_CN import TABLE as _table


def tr(text: str) -> str:
    return _table.get(text, text)


def init() -> str:
    return "zh_CN"

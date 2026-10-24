"""界面文字表。

``tr(text)`` 把代码中的英文源字符串映射为中文；表里没有的字符串原样返回，
因此动态拼接的文字不会出错。全部翻译集中在 ``cvflow/ui/i18n_zh_CN.py``。
"""
from __future__ import annotations

from .i18n_zh_CN import CHOICES as _choices
from .i18n_zh_CN import TABLE as _table


def tr(text: str) -> str:
    return _table.get(text, text)


def tr_choice(param: str, value: str) -> str:
    """下拉框里某个选项的显示文字；表里没有就原样显示。

    按「参数名.取值」查：同名参数在不同节点上含义不同，光看取值会串味。
    **只管显示**——存进方案文件、以及代码里比较的，始终是原始值。
    """
    return _choices.get(f"{param}.{value}", value)


def init() -> str:
    return "zh_CN"

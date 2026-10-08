"""``choice`` 类型参数的候选来源。

有些参数的候选**运行时才知道**：方案里配了哪些通信设备、设备上有哪些数据点、有哪些解析规则、
有哪几条流程。这些不能像 ``enum`` 那样写死在节点定义里，所以节点只声明一个来源名，
界面渲染时到这里取候选，渲染成**可编辑**下拉框——能从列表里挑，也能手填（配置顺序不限，
先写节点后建设备也不会丢）。

插件可以注册自己的来源：``register_options("my.things", lambda node: [...])``。
"""
from __future__ import annotations

from typing import Callable

from ..core.node import Node

#: 来源名 → 取候选的函数。函数拿到当前节点（可以据此联动，比如数据点跟着设备变）。
Provider = Callable[[Node], list[str]]
OPTION_PROVIDERS: dict[str, Provider] = {}


def register_options(name: str, provider: Provider | None = None):
    """注册一个候选来源。既能直接调用，也能当装饰器用。"""
    if provider is None:
        def decorate(fn: Provider) -> Provider:
            OPTION_PROVIDERS[name] = fn
            return fn
        return decorate
    OPTION_PROVIDERS[name] = provider
    return provider


def options_for(name: str, node: Node) -> list[str]:
    """取候选。来源不存在或取值出错时返回空列表——下拉框照样能手填，不该因此报错。"""
    provider = OPTION_PROVIDERS.get(name)
    if provider is None:
        return []
    try:
        return [str(x) for x in provider(node) if str(x)]
    except Exception:
        return []


def _manager():
    from ..comm.manager import get_manager
    return get_manager()


@register_options("comm.devices")
def _devices(node: Node) -> list[str]:
    """通信设备。显示名字，存的是**稳定 id**，所以改名不会让节点失去引用。"""
    mgr = _manager()
    return [] if mgr is None else [d.id for d in mgr.devices.values()]


@register_options("comm.register_devices")
def _register_devices(node: Node) -> list[str]:
    """支持数据点的设备（Modbus / MC / S7）。"""
    mgr = _manager()
    if mgr is None:
        return []
    from ..comm.modbus import _PointHost
    return [d.id for d in mgr.devices.values() if isinstance(d, _PointHost)]


@register_options("comm.points")
def _points(node: Node) -> list[str]:
    """当前节点所选设备上的数据点；没选设备就把所有设备的数据点都列出来。"""
    mgr = _manager()
    if mgr is None:
        return []
    from ..comm.modbus import _PointHost
    ref = str(node.values.get("device") or "")
    dev = mgr.resolve(ref) if ref else None
    if isinstance(dev, _PointHost):
        return dev.points.names()
    names: list[str] = []
    for d in mgr.devices.values():
        if isinstance(d, _PointHost):
            names += [n for n in d.points.names() if n not in names]
    return names


@register_options("comm.parse_rules")
def _parse_rules(node: Node) -> list[str]:
    mgr = _manager()
    return [] if mgr is None else list(mgr.parse_rules)


@register_options("comm.format_rules")
def _format_rules(node: Node) -> list[str]:
    mgr = _manager()
    return [] if mgr is None else list(mgr.format_rules)


@register_options("comm.handshakes")
def _handshakes(node: Node) -> list[str]:
    mgr = _manager()
    return [] if mgr is None else [h.name for h in mgr.handshakes]


@register_options("comm.peers")
def _peers(node: Node) -> list[str]:
    """当前节点所选设备上已连接的对端（TCP 服务端的客户端、UDP 的来源）。"""
    mgr = _manager()
    if mgr is None:
        return []
    dev = mgr.resolve(str(node.values.get("device") or ""))
    return [] if dev is None else list(dev.peers)


@register_options("flows")
def _flows(node: Node) -> list[str]:
    mgr = _manager()
    return [] if mgr is None else list(mgr.runners)


def display_label(name: str, value: str, node: Node) -> str:
    """下拉框里给候选加可读标签：设备 id 旁边显示设备名。"""
    if name not in ("comm.devices", "comm.register_devices"):
        return value
    mgr = _manager()
    if mgr is None:
        return value
    dev = mgr.resolve(value)
    return f"{dev.name}" if dev is not None else value

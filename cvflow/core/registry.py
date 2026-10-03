"""Node registry and plugin discovery.

Built-in operators register themselves with the ``@register`` decorator when
``cvflow.operators`` is imported. User plugins are plain ``.py`` files in a
plugin directory; every ``Node`` subclass defined in such a file is registered,
so adding an algorithm never requires touching the platform code.
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from .node import Node

log = logging.getLogger("cvflow.registry")


class NodeRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[Node]] = {}
        self._sources: dict[str, str] = {}      # type_id -> origin ("builtin" or plugin path)
        self._builtins_loaded = False

    # ---- registration ----
    def register(self, cls: type[Node], source: str = "builtin") -> type[Node]:
        if not cls.type_id:
            raise ValueError(f"{cls.__name__} 没有定义 type_id")
        if cls.type_id in self._classes and self._classes[cls.type_id] is not cls:
            log.info("替换节点类型 %s (%s)", cls.type_id, source)
        self._classes[cls.type_id] = cls
        self._sources[cls.type_id] = source
        return cls

    def unregister(self, type_id: str) -> None:
        self._classes.pop(type_id, None)
        self._sources.pop(type_id, None)

    # ---- lookup ----
    def has(self, type_id: str) -> bool:
        return type_id in self._classes

    def get(self, type_id: str) -> type[Node]:
        try:
            return self._classes[type_id]
        except KeyError:
            raise KeyError(f"未知的节点类型 {type_id!r}，插件是否已加载？") from None

    def create(self, type_id: str, node_id: str | None = None, name: str | None = None,
               values: dict | None = None) -> Node:
        node = self.get(type_id)(node_id=node_id, name=name)
        if values:
            node.update(values)
        return node

    def node_from_dict(self, d: dict) -> Node:
        return self.get(d["type"]).from_dict(d)

    def all(self) -> list[type[Node]]:
        return sorted(self._classes.values(), key=lambda c: (c.category, c.label))

    def categories(self) -> dict[str, list[type[Node]]]:
        out: dict[str, list[type[Node]]] = {}
        for cls in self.all():
            out.setdefault(cls.category, []).append(cls)
        return out

    def source_of(self, type_id: str) -> str:
        return self._sources.get(type_id, "")

    # ---- loading ----
    def load_builtins(self) -> None:
        if not self._builtins_loaded:
            importlib.import_module("cvflow.operators")
            self._builtins_loaded = True

    def load_plugin_file(self, path: str | Path) -> list[type[Node]]:
        path = Path(path).resolve()
        mod_name = f"cvflow_plugin_{path.stem}_{abs(hash(str(path))) & 0xFFFF:04x}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载插件 {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
        found: list[type[Node]] = []
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, Node) and obj is not Node \
                    and obj.__module__ == mod_name and obj.type_id:
                self.register(obj, source=str(path))
                found.append(obj)
        log.info("插件 %s：%d 个节点类型", path.name, len(found))
        return found

    def load_plugin_dir(self, directory: str | Path) -> list[type[Node]]:
        directory = Path(directory)
        found: list[type[Node]] = []
        if not directory.is_dir():
            log.warning("插件目录 %s 不存在", directory)
            return found
        for py in sorted(directory.glob("*.py")):
            if py.name.startswith("_"):
                continue
            try:
                found.extend(self.load_plugin_file(py))
            except Exception as e:  # keep loading the others
                log.error("插件 %s 加载失败：%s", py.name, e)
        return found


registry = NodeRegistry()


def register(cls: type[Node]) -> type[Node]:
    """Class decorator registering a built-in node type."""
    return registry.register(cls)

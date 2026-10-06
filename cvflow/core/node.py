"""Node model: ports, parameter descriptors and the Node base class.

A node is the unit of work in a flow. Subclasses declare their interface with
class attributes and implement ``process``::

    @register
    class Threshold(Node):
        type_id = "preprocess.threshold"
        category = "Preprocess"
        label = "Threshold"
        inputs = [Port("image", DataType.IMAGE)]
        outputs = [Port("image", DataType.IMAGE)]
        params = [Param("thresh", 128, "int", min=0, max=255)]

        def process(self, ctx, inputs):
            ...
            return {"image": out_img}

Anything a user can write in Python (an ONNX model, a PyTorch network, a
reinforcement-learning agent) fits this interface, which is what makes the
platform open: algorithms are plugins, not a fixed operator catalogue.
"""
from __future__ import annotations

import copy
import enum
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import DataType, Rect, to_jsonable

if TYPE_CHECKING:  # pragma: no cover
    from .engine import RunContext


class NodeStatus(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    OK = "ok"
    NG = "ng"
    ERROR = "error"
    SKIPPED = "skipped"


class NodeError(Exception):
    """Raise from ``process`` for a clean, user-facing failure message."""


@dataclass
class Port:
    name: str
    dtype: DataType = DataType.ANY
    label: str = ""
    optional: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name
        if not isinstance(self.dtype, DataType):
            self.dtype = DataType(self.dtype)


PARAM_KINDS = ("int", "float", "bool", "string", "enum", "file", "dir", "rect", "code",
               "color", "list", "json")


@dataclass
class Param:
    """Declarative parameter description; the UI builds editors from these."""

    name: str
    default: Any = None
    kind: str = "float"
    label: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list = field(default_factory=list)
    description: str = ""
    advanced: bool = False
    filter: str = ""  # file-dialog filter for kind == "file"

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"参数 {self.name!r}：未知类型 {self.kind!r}")
        if not self.label:
            self.label = self.name.replace("_", " ").title()
        if self.kind == "enum" and not self.choices:
            raise ValueError(f"参数 {self.name!r}：enum 类型需要 choices")
        self.default = self.coerce(self.default)

    def coerce(self, value: Any) -> Any:
        """Convert and clamp ``value`` so stored parameters are always well typed."""
        k = self.kind
        if k == "int":
            v = int(round(float(value))) if value is not None else 0
            return self._clamp(v)
        if k == "float":
            v = float(value) if value is not None else 0.0
            return self._clamp(v)
        if k == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if k in ("string", "file", "dir", "code", "color"):
            return "" if value is None else str(value)
        if k == "enum":
            return value if value in self.choices else self.choices[0]
        if k == "rect":
            if value is None:
                return None
            if isinstance(value, Rect):
                return value.to_dict()
            if isinstance(value, dict) and {"x", "y", "w", "h"} <= set(value):
                return {"x": float(value["x"]), "y": float(value["y"]), "w": float(value["w"]),
                        "h": float(value["h"]), "angle": float(value.get("angle", 0.0))}
            raise ValueError(f"参数 {self.name!r}：无效的矩形 {value!r}")
        if k == "list":
            return list(value) if value is not None else []
        if k == "json":
            return copy.deepcopy(value)
        return value

    def _clamp(self, v):
        if self.min is not None and v < self.min:
            v = type(v)(self.min)
        if self.max is not None and v > self.max:
            v = type(v)(self.max)
        return v

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "label": self.label, "default": self.default,
                "min": self.min, "max": self.max, "step": self.step, "choices": list(self.choices),
                "description": self.description, "advanced": self.advanced}


class Node:
    """Base class for all algorithm / IO nodes. See module docstring."""

    # ---- class-level interface declaration (override in subclasses) ----
    type_id: str = ""
    category: str = "Misc"
    label: str = ""
    description: str = ""
    inputs: list[Port] = []
    outputs: list[Port] = []
    params: list[Param] = []

    #: 旧参数名 → 现在的名字。方案文件里可能还存着旧名字，读进来时翻译过去。
    legacy_params: dict[str, str] = {}

    #: 能不能单独触发这一路（取一帧图、沿本路跑到深度学习节点汇合）。
    #: 置 True 的节点会在画布上带一个可点的触发图标，产线上同一个动作由 PLC 报文触发。
    triggerable: bool = False
    color: str = "#4a6fa5"      # title colour hint for the editor
    is_judge: bool = False      # if True, a False "ok" output marks the node (and run) NG

    def __init__(self, node_id: str | None = None, name: str | None = None) -> None:
        self.id: str = node_id or uuid.uuid4().hex[:8]
        self.name: str = name or self.label or self.type_id
        self.values: dict[str, Any] = {p.name: copy.deepcopy(p.default) for p in self.params}
        self.enabled: bool = True
        self.position: list[float] = [0.0, 0.0]
        self.status: NodeStatus = NodeStatus.IDLE
        self.last_error: str = ""
        self.last_time_ms: float = 0.0
        # Persistent, per-node runtime state (model weights path, learning statistics,
        # frame counters ...). Saved with the solution when JSON-serialisable.
        self.state: dict[str, Any] = {}
        self._is_setup: bool = False

    # ---- parameters ----
    def param_def(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{self.type_id}：没有参数 {name!r}")

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def set(self, name: str, value: Any) -> None:
        name = self.legacy_params.get(name, name)
        p = self.param_def(name)
        self.values[name] = p.coerce(value)
        self.on_param_changed(name, self.values[name])

    def update(self, values: dict[str, Any]) -> None:
        for k, v in values.items():
            k = self.legacy_params.get(k, k)
            if any(p.name == k for p in self.params):
                self.set(k, v)

    def rect(self, name: str) -> Rect | None:
        return Rect.from_dict(self.values.get(name))

    def on_param_changed(self, name: str, value: Any) -> None:  # noqa: B027
        """Hook for nodes that cache derived data (e.g. reload a template image)."""

    # ---- lifecycle ----
    def set_enabled(self, enabled: bool) -> None:
        """启用 / 禁用节点。禁用的节点在流程里会被跳过，顺带让它把占用的资源放掉。"""
        enabled = bool(enabled)
        if enabled == self.enabled:
            return
        self.enabled = enabled
        self.on_enabled_changed(enabled)

    def on_enabled_changed(self, enabled: bool) -> None:  # noqa: B027
        """可选钩子：节点被启用/禁用时调用。禁用时适合把重资源（模型权重、相机连接）放掉。"""

    def preload(self) -> None:  # noqa: B027
        """可选钩子：参数选好之后提前把重资源准备好（例如把模型加载进显存）。

        界面在用户改完参数、或打开方案之后会在后台线程调用它，目的是让第一次运行不用再等加载。
        所以实现必须能在后台线程里跑，并且在"已经准备好"时直接返回；失败就抛异常，由调用方记日志，
        真正运行时还会照常再试一次。
        """

    def setup(self, ctx: "RunContext | None" = None) -> None:  # noqa: B027
        """Acquire resources (open camera, load model). Called once before running."""

    def teardown(self) -> None:  # noqa: B027
        """Release resources acquired in ``setup``."""

    def reset_state(self) -> None:
        self.state.clear()

    def process(self, ctx: "RunContext", inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    # ---- introspection ----
    @classmethod
    def input_port(cls, name: str) -> Port | None:
        return next((p for p in cls.inputs if p.name == name), None)

    @classmethod
    def output_port(cls, name: str) -> Port | None:
        return next((p for p in cls.outputs if p.name == name), None)

    def get_input_port(self, name: str) -> Port | None:
        return next((p for p in self.inputs if p.name == name), None)

    def get_output_port(self, name: str) -> Port | None:
        return next((p for p in self.outputs if p.name == name), None)

    @classmethod
    def describe(cls) -> dict:
        return {
            "type": cls.type_id, "category": cls.category, "label": cls.label,
            "description": cls.description,
            "inputs": [{"name": p.name, "dtype": p.dtype.value, "optional": p.optional} for p in cls.inputs],
            "outputs": [{"name": p.name, "dtype": p.dtype.value} for p in cls.outputs],
            "params": [p.to_dict() for p in cls.params],
        }

    # ---- serialisation ----
    def to_dict(self) -> dict:
        d = {
            "id": self.id, "type": self.type_id, "name": self.name, "enabled": self.enabled,
            "position": [float(self.position[0]), float(self.position[1])],
            "values": to_jsonable(self.values),
        }
        try:
            state = to_jsonable(self.state)
            if state:
                d["state"] = state
        except Exception:  # pragma: no cover - defensive
            pass
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        node = cls(node_id=d.get("id"), name=d.get("name"))
        node.enabled = bool(d.get("enabled", True))
        pos = d.get("position") or [0, 0]
        node.position = [float(pos[0]), float(pos[1])]
        for k, v in (d.get("values") or {}).items():
            try:
                node.set(k, v)
            except (KeyError, ValueError):
                pass  # tolerate parameters removed in newer node versions
        node.state = dict(d.get("state") or {})
        return node

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} id={self.id}>"

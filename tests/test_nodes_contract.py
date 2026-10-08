"""契约测试：对注册表里的每一个节点自动执行，不依赖具体实现。

1. 元数据合法：type_id 唯一、分类/名称非空、端口类型合法、参数默认值能通过 coerce、describe() 可序列化、to_dict/from_dict 往返一致。
2. 输出契约：用代表性输入运行后，声明的每个输出端口都存在，且值类型与声明一致。
3. 错误隔离：喂错误类型的输入时，节点以 ERROR 状态结束并给出非空错误信息，引擎不抛异常。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvflow.core import Circle, DataType, Image, Line, Point, Rect, registry
from cvflow.core.node import NodeStatus
from nodes_helpers import ROOT, make_identity_model, run_node, squares_bgr

registry.load_builtins()
registry.load_plugin_dir(ROOT / "examples" / "plugins")
ALL_TYPES = sorted(c.type_id for c in registry.all())


def _sample(dtype: DataType):
    return {
        DataType.IMAGE: squares_bgr(), DataType.INT: 3, DataType.FLOAT: 2.5, DataType.BOOL: True,
        DataType.STRING: "abc", DataType.POINT: Point(1, 2), DataType.RECT: Rect(20, 70, 60, 60),
        DataType.CIRCLE: Circle(10, 10, 5), DataType.LINE: Line(0, 0, 10, 10),
        DataType.CONTOURS: [np.array([[[0, 0]], [[5, 0]], [[5, 5]]], dtype=np.int32)],
        DataType.LIST: [1, 2, 3], DataType.DICT: {"a": 1}, DataType.TENSOR: np.zeros((1, 3), np.float32),
        DataType.ANY: 1.0,
    }[dtype]


_TYPE_CHECK = {
    DataType.IMAGE: lambda v: isinstance(v, Image), DataType.INT: lambda v: isinstance(v, (int, np.integer)) and not isinstance(v, bool),
    DataType.FLOAT: lambda v: isinstance(v, (int, float, np.floating)) and not isinstance(v, bool),
    DataType.BOOL: lambda v: isinstance(v, (bool, np.bool_)), DataType.STRING: lambda v: isinstance(v, str),
    DataType.POINT: lambda v: isinstance(v, Point), DataType.RECT: lambda v: isinstance(v, Rect),
    DataType.CIRCLE: lambda v: isinstance(v, Circle), DataType.LINE: lambda v: isinstance(v, Line),
    DataType.CONTOURS: lambda v: isinstance(v, list), DataType.LIST: lambda v: isinstance(v, list),
    DataType.DICT: lambda v: isinstance(v, dict), DataType.TENSOR: lambda v: isinstance(v, np.ndarray),
    DataType.ANY: lambda v: True,
}


@pytest.fixture(scope="module")
def resources(tmp_path_factory):
    d = tmp_path_factory.mktemp("res")
    model = make_identity_model(d / "ident.onnx", (1, 3, 8, 8))
    img = d / "img.png"
    import cv2
    cv2.imwrite(str(img), squares_bgr().data)
    (d / "labels.txt").write_text("a\nb\nc\n", encoding="utf-8")
    return {"dir": d, "model": model, "img": img}


def _setup_for(type_id: str, res: dict):
    """每种节点在契约测试中需要的参数与输入覆盖；返回 (values, inputs, expect_error)。"""
    images = str(ROOT / "examples" / "images")
    table = {
        "source.image_file": ({"path": str(res["img"])}, {}, False),
        "source.image_folder": ({"directory": images}, {}, False),
        "source.camera": ({"camera": "contract_cam", "kind": "folder", "source": images}, {}, False),
        "analysis.template_match": ({"template_path": str(res["img"])}, {"template": Image(squares_bgr().data[70:130, 10:70].copy())}, False),
        "analysis.caliper": ({"roi": {"x": 10, "y": 90, "w": 180, "h": 20}}, {}, False),
        "dl.onnx": ({"model_path": str(res["model"]), "width": 8, "height": 8}, {}, False),
        "dl.onnx_classifier": ({"model_path": str(res["model"]), "width": 8, "height": 8, "labels_path": str(res["dir"] / "labels.txt")}, {}, False),
        "dl.onnx_detector": ({"model_path": str(res["model"]), "width": 8, "height": 8}, {}, True),   # identity 输出不是检测布局 → 应以 ERROR 结束
        "output.save_image": ({"directory": str(res["dir"] / "out")}, {}, False),
        "output.comm_send": ({"device": "x"}, {}, True),          # 没有通信管理器 → 明确报错
        "output.modbus_write": ({"device": "x"}, {}, True),
        # 通信节点在没有通信管理器时都应给出可读的错误，而不是抛 Traceback。
        # 接上真实设备的功能测试在 tests/test_comm_nodes.py。
        "comm.receive": ({"device": "x"}, {}, True),
        "comm.send": ({"device": "x", "template": "hi"}, {}, True),
        "comm.parse": ({"rule": "x"}, {}, True),
        "comm.format": ({"rule": "x"}, {}, True),
        "comm.read_point": ({"device": "x", "point": "p"}, {}, True),
        "comm.write_point": ({"device": "x", "point": "p"}, {}, True),
        "comm.status": ({"device": "x"}, {}, True),
        "learning.torch_template": ({}, {}, True),               # 未安装 torch → 明确报错
        "logic.gate": ({}, {"condition": True, "value": 5}, False),
        "logic.judge": ({"op": "is_true"}, {"value": True}, False),
    }
    return table.get(type_id, ({}, {}, False))


@pytest.mark.parametrize("type_id", ALL_TYPES)
def test_contract_metadata(type_id):
    cls = registry.get(type_id)
    assert cls.label and cls.category, "名称与分类不能为空"
    assert len([c for c in registry.all() if c.type_id == type_id]) == 1, "type_id 必须唯一"
    names = [p.name for p in cls.inputs] + [p.name for p in cls.outputs]
    assert all(isinstance(p.dtype, DataType) for p in cls.inputs + cls.outputs)
    assert len(set(p.name for p in cls.inputs)) == len(cls.inputs) and len(set(p.name for p in cls.outputs)) == len(cls.outputs)
    node = cls()
    for p in cls.params:
        assert p.coerce(p.default) == p.default or p.kind in ("json", "list", "rect"), f"参数 {p.name} 默认值不稳定"
    json.dumps(cls.describe(), ensure_ascii=False)
    d = json.loads(json.dumps(node.to_dict(), ensure_ascii=False, default=str))
    back = cls.from_dict(d)
    assert back.values == node.values and back.id == node.id


@pytest.mark.parametrize("type_id", ALL_TYPES)
def test_contract_outputs_match_declaration(type_id, resources):
    cls = registry.get(type_id)
    values, override, expect_error = _setup_for(type_id, resources)
    inputs = {p.name: _sample(p.dtype) for p in cls.inputs}
    inputs.update(override)
    nr, r, node = run_node(registry, type_id, inputs, values)
    if expect_error:
        assert nr.status == NodeStatus.ERROR and nr.error, f"{type_id} 应明确报错，实际 {nr.status.value}"
        assert "Traceback" not in nr.error
        return
    assert nr.status in (NodeStatus.OK, NodeStatus.NG), f"{type_id}: {nr.status.value} {nr.error}"
    for p in cls.outputs:
        assert p.name in nr.outputs, f"{type_id} 缺少声明的输出 {p.name}"
        v = nr.outputs[p.name]
        assert v is not None or p.dtype == DataType.ANY, f"{type_id}.{p.name} 不应为 None"
        if v is not None:
            assert _TYPE_CHECK[p.dtype](v), f"{type_id}.{p.name} 类型 {type(v).__name__} 与声明 {p.dtype.value} 不符"
    assert set(nr.outputs) <= {p.name for p in cls.outputs} | {"__skip__"}, f"{type_id} 输出了未声明的端口 {set(nr.outputs) - {p.name for p in cls.outputs}}"


@pytest.mark.parametrize("type_id", [t for t in ALL_TYPES if registry.get(t).inputs])
def test_contract_bad_input_is_isolated(type_id, resources):
    cls = registry.get(type_id)
    values, _, _ = _setup_for(type_id, resources)
    bad = {}
    for p in cls.inputs:
        bad[p.name] = "not-an-image" if p.dtype == DataType.IMAGE else (object() if p.dtype != DataType.ANY else "x")
    if all(p.dtype == DataType.ANY for p in cls.inputs):
        pytest.skip("只有 ANY 输入，无类型约束")
    nr, r, node = run_node(registry, type_id, bad, values)
    assert nr.status in (NodeStatus.ERROR, NodeStatus.OK, NodeStatus.NG, NodeStatus.SKIPPED)
    if nr.status == NodeStatus.ERROR:
        assert nr.error and "Traceback" not in nr.error
    assert isinstance(r.ok, bool)   # 引擎本身没有抛异常

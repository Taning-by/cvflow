"""Decision and data-flow operators: judge, expressions, variables, formatting, gating."""
from __future__ import annotations

import math
import time

from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType

_COLOR = "#00695c"

_SAFE_FUNCS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_FUNCS.update({"abs": abs, "min": min, "max": max, "round": round, "len": len, "int": int,
                    "float": float, "str": str, "bool": bool, "sum": sum, "any": any, "all": all})


def safe_eval(expr: str, names: dict):
    code = compile(expr, "<expr>", "eval")
    for n in code.co_names:
        if n not in names and n not in _SAFE_FUNCS:
            raise NodeError(f"表达式中有未知名称 {n!r}")
    return eval(code, {"__builtins__": {}}, {**_SAFE_FUNCS, **names})


@register
class Judge(Node):
    type_id = "logic.judge"
    category = "Logic"
    label = "Judge"
    description = "把值和上下限比较，结果决定本次运行 OK/NG。"
    color = _COLOR
    is_judge = True
    inputs = [Port("value", DataType.ANY)]
    outputs = [Port("ok", DataType.BOOL), Port("value", DataType.ANY), Port("reason", DataType.STRING)]
    params = [Param("op", "in_range", "enum", choices=["in_range", "==", "!=", "<", "<=", ">", ">=", "is_true", "contains"]),
              Param("low", 0.0, "float"), Param("high", 100.0, "float"),
              Param("text", "", "string", description="Expected text for == / != / contains on strings"),
              Param("name", "", "string", description="Name used in the NG reason")]

    def process(self, ctx, inputs):
        v = inputs["value"]
        op = self.get("op")
        lo, hi = float(self.get("low")), float(self.get("high"))
        txt = self.get("text")
        if op == "is_true":
            ok = bool(v)
        elif op == "contains":
            ok = str(txt) in str(v)
        elif isinstance(v, str) or (txt and op in ("==", "!=")):
            ok = (str(v) == str(txt)) if op == "==" else (str(v) != str(txt)) if op == "!=" else False
        else:
            x = float(v)
            ok = {"in_range": lo <= x <= hi, "==": x == lo, "!=": x != lo, "<": x < lo, "<=": x <= lo,
                  ">": x > lo, ">=": x >= lo}[op]
        name = self.get("name") or self.name
        reason = "" if ok else (f"{name}：{v!r} 不满足 {op} [{lo}, {hi}]" if op == "in_range" else f"{name}：{v!r} 不满足 {op} {txt or lo}")
        return {"ok": ok, "value": v, "reason": reason}


@register
class Expression(Node):
    type_id = "logic.expression"
    category = "Logic"
    label = "Expression"
    description = "对输入 a..d 求一个 Python 表达式，可用 math 函数。"
    color = _COLOR
    inputs = [Port(n, DataType.ANY, optional=True) for n in "abcd"]
    outputs = [Port("value", DataType.ANY)]
    params = [Param("expression", "a", "string")]

    def process(self, ctx, inputs):
        names = {k: v for k, v in inputs.items()}
        names["vars"] = ctx.variables.snapshot()
        return {"value": safe_eval(self.get("expression"), names)}


@register
class SetVariable(Node):
    type_id = "logic.set_variable"
    category = "Logic"
    label = "Set Variable"
    color = _COLOR
    inputs = [Port("value", DataType.ANY)]
    outputs = [Port("value", DataType.ANY)]
    params = [Param("name", "var1", "string")]

    def process(self, ctx, inputs):
        ctx.variables.set(self.get("name"), inputs["value"])
        return {"value": inputs["value"]}


@register
class GetVariable(Node):
    type_id = "logic.get_variable"
    category = "Logic"
    label = "Get Variable"
    color = _COLOR
    outputs = [Port("value", DataType.ANY)]
    params = [Param("name", "var1", "string"), Param("default", "", "string")]

    def process(self, ctx, inputs):
        return {"value": ctx.variables.get(self.get("name"), self.get("default"))}


@register
class Format(Node):
    type_id = "logic.format"
    category = "Logic"
    label = "Format Text"
    description = "按模板拼字符串，如 'RESULT,{a},{b:.2f}'。"
    color = _COLOR
    inputs = [Port(n, DataType.ANY, optional=True) for n in "abcd"]
    outputs = [Port("text", DataType.STRING)]
    params = [Param("template", "{a}", "string")]

    def process(self, ctx, inputs):
        try:
            return {"text": self.get("template").format(**inputs, vars=ctx.variables.snapshot(), run_id=ctx.run_id)}
        except (KeyError, ValueError, IndexError) as e:
            raise NodeError(f"模板有误：{e}") from e


@register
class Gate(Node):
    type_id = "logic.gate"
    category = "Logic"
    label = "Gate"
    description = "condition 为真时透传 value，否则跳过下游。"
    color = _COLOR
    inputs = [Port("condition", DataType.ANY), Port("value", DataType.ANY, optional=True)]
    outputs = [Port("value", DataType.ANY)]
    params = [Param("invert", False, "bool")]

    def process(self, ctx, inputs):
        cond = bool(inputs["condition"]) != bool(self.get("invert"))
        if not cond:
            return {"__skip__": "gate closed"}
        return {"value": inputs.get("value")}


@register
class Delay(Node):
    type_id = "logic.delay"
    category = "Logic"
    label = "Delay"
    color = _COLOR
    inputs = [Port("value", DataType.ANY, optional=True)]
    outputs = [Port("value", DataType.ANY)]
    params = [Param("ms", 10, "int", min=0, max=60000)]

    def process(self, ctx, inputs):
        time.sleep(int(self.get("ms")) / 1000.0)
        return {"value": inputs.get("value")}


@register
class Counter(Node):
    type_id = "logic.counter"
    category = "Logic"
    label = "Counter"
    description = "统计运行次数。接了 count_if 时只在它为真时计数。"
    color = _COLOR
    inputs = [Port("count_if", DataType.ANY, optional=True)]
    outputs = [Port("count", DataType.INT)]
    params = [Param("reset", False, "bool", description="Tick to reset on next run")]

    def process(self, ctx, inputs):
        if self.get("reset"):
            self.state["count"] = 0
            self.values["reset"] = False
        links = ctx.graph.input_links(self.id)
        inc = 1 if ("count_if" not in links or bool(inputs.get("count_if"))) else 0
        self.state["count"] = int(self.state.get("count", 0)) + inc
        return {"count": self.state["count"]}

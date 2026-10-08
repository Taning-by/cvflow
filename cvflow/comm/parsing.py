"""报文解析（对端 → 流程变量 / 流程输入参数）与结果格式化（流程结果 → 文本 / JSON / 二进制）。

两个方向都**不使用 eval**：
* 解析：按声明的方式取字段（分隔符下标、正则分组、JSON 路径、定长切片），再按声明的类型转换。
* 格式化：字段来源只认一张固定的取值表（``status`` / ``out.名字`` / ``var.名字`` /
  ``node[节点名].端口`` / ``field.名字`` / ``literal:常量``），文本模板用 ``str.format``
  的受限命名空间渲染，没有任何表达式求值。
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict, dataclass, field
from re import error
from typing import Any

from .framing import decode_text, encode_text

# ------------------------------------------------------------------ 标量编解码
#: 字节流里的标量类型 → struct 格式字符（字节序单独拼）
SCALAR_FORMATS = {"int8": "b", "uint8": "B", "int16": "h", "uint16": "H",
                  "int32": "i", "uint32": "I", "float32": "f", "float64": "d"}
SCALAR_SIZES = {k: struct.calcsize(v) for k, v in SCALAR_FORMATS.items()}

VALUE_TYPES = ("string", "int", "float", "bool")
BINARY_TYPES = ("string", "bool", *SCALAR_FORMATS)


class ParseError(Exception):
    """报文不符合解析规则。调用方应把它当成"参数错误"回给对端，而不是当成软件故障。"""


class FormatError(Exception):
    pass


def pack_scalar(value: Any, dtype: str, endian: str = "big") -> bytes:
    if dtype not in SCALAR_FORMATS:
        raise FormatError(f"未知的二进制类型 {dtype!r}")
    fmt = (">" if endian == "big" else "<") + SCALAR_FORMATS[dtype]
    if dtype in ("float32", "float64"):
        return struct.pack(fmt, float(value))
    v = int(round(float(value)))
    lo, hi = _int_range(dtype)
    return struct.pack(fmt, max(lo, min(hi, v)))


def unpack_scalar(data: bytes, dtype: str, endian: str = "big") -> Any:
    if dtype not in SCALAR_FORMATS:
        raise ParseError(f"未知的二进制类型 {dtype!r}")
    fmt = (">" if endian == "big" else "<") + SCALAR_FORMATS[dtype]
    size = SCALAR_SIZES[dtype]
    if len(data) < size:
        raise ParseError(f"{dtype} 需要 {size} 字节，只有 {len(data)} 字节")
    return struct.unpack(fmt, data[:size])[0]


def _int_range(dtype: str) -> tuple[int, int]:
    bits = SCALAR_SIZES[dtype] * 8
    if dtype.startswith("u"):
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def coerce(text: Any, dtype: str, scale: float = 1.0) -> Any:
    """把取到的原始文本/数值转成声明的类型。失败时抛 ParseError，带上看得懂的原因。"""
    if dtype == "string":
        return str(text)
    if dtype == "bool":
        s = str(text).strip().lower()
        if s in ("1", "true", "yes", "on", "ok"):
            return True
        if s in ("0", "false", "no", "off", "ng", ""):
            return False
        raise ParseError(f"{text!r} 不是布尔值（可用 1/0、true/false、OK/NG）")
    try:
        v = float(str(text).strip())
    except (TypeError, ValueError):
        raise ParseError(f"{text!r} 不是数字") from None
    v *= float(scale or 1.0)
    if dtype == "int":
        return int(round(v))
    return v


# ------------------------------------------------------------------ 解析
PARSE_KINDS = ("delimited", "regex", "json", "fixed_fields", "whole")
PARSE_LABELS = {"delimited": "按分隔符切分", "regex": "正则分组", "json": "JSON 字段",
                "fixed_fields": "定长切片", "whole": "整条报文"}


@dataclass
class ParseField:
    """一个要取出来的字段。按所属解析方式用到不同的定位项。"""

    name: str = "value"
    index: int = 0            # delimited：第几段（0 开始）
    group: str = ""           # regex：分组名或分组号
    path: str = ""            # json：点号路径，支持 a.b[0]
    start: int = 0            # fixed_fields：起始字节
    length: int = 0           # fixed_fields：字节数
    dtype: str = "string"
    scale: float = 1.0
    required: bool = True
    default: Any = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ParseField":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class ParseRule:
    id: str = ""
    name: str = "parse"
    kind: str = "delimited"
    separator: str = ","
    pattern: str = ""
    encoding: str = "utf-8"
    strip: bool = True
    to_variables: bool = True      # 解析结果是否同时写入全局变量
    fields: list[ParseField] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in PARSE_KINDS:
            raise ValueError(f"未知的解析方式 {self.kind!r}，可用：{PARSE_KINDS}")
        self.fields = [f if isinstance(f, ParseField) else ParseField.from_dict(f) for f in self.fields]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fields"] = [f.to_dict() for f in self.fields]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ParseRule":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known["fields"] = [ParseField.from_dict(f) for f in d.get("fields", [])]
        return cls(**known)

    # ---- 解析 ----
    def parse(self, data: bytes) -> dict[str, Any]:
        if self.kind == "fixed_fields":
            return self._parse_fixed(data)
        text = decode_text(data, self.encoding)
        if self.strip:
            text = text.strip()
        if self.kind == "whole":
            name = self.fields[0].name if self.fields else "message"
            dtype = self.fields[0].dtype if self.fields else "string"
            return {name: coerce(text, dtype, self.fields[0].scale if self.fields else 1.0)}
        if self.kind == "delimited":
            return self._parse_delimited(text)
        if self.kind == "regex":
            return self._parse_regex(text)
        return self._parse_json(text)

    def _value(self, f: ParseField, raw: Any, where: str) -> Any:
        if raw is None:
            if f.required:
                raise ParseError(f"字段 {f.name!r}：{where}")
            return f.default
        try:
            return coerce(raw, f.dtype, f.scale)
        except ParseError as e:
            if f.required:
                raise ParseError(f"字段 {f.name!r}：{e}") from None
            return f.default

    def _parse_delimited(self, text: str) -> dict[str, Any]:
        sep = self.separator or ","
        parts = [p.strip() if self.strip else p for p in text.split(sep)]
        out: dict[str, Any] = {}
        for f in self.fields:
            raw = parts[f.index] if 0 <= f.index < len(parts) else None
            out[f.name] = self._value(f, raw, f"报文只有 {len(parts)} 段，取不到第 {f.index} 段")
        return out

    def _parse_regex(self, text: str) -> dict[str, Any]:
        try:
            rx = re.compile(self.pattern)
        except re.error as e:
            raise ParseError(f"正则有误：{e}") from None
        m = rx.search(text)
        if m is None:
            raise ParseError(f"报文 {text!r} 不匹配正则 {self.pattern!r}")
        out: dict[str, Any] = {}
        for f in self.fields:
            key: Any = f.group or f.name
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            try:
                raw = m.group(key)
            except (IndexError, error):          # 分组号越界 / 分组名不存在
                raw = None
            out[f.name] = self._value(f, raw, f"正则里没有分组 {f.group or f.name!r}")
        return out

    def _parse_json(self, text: str) -> dict[str, Any]:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            raise ParseError(f"不是合法 JSON：{e.msg}（第 {e.lineno} 行第 {e.colno} 列）") from None
        out: dict[str, Any] = {}
        for f in self.fields:
            raw = _json_path(doc, f.path or f.name)
            out[f.name] = self._value(f, raw, f"JSON 里没有 {f.path or f.name!r}")
        return out

    def _parse_fixed(self, data: bytes) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in self.fields:
            end = f.start + (f.length or 0)
            if f.length <= 0 or end > len(data):
                out[f.name] = self._value(f, None, f"报文 {len(data)} 字节，取不到 [{f.start}, {end})")
                continue
            chunk = data[f.start:end]
            if f.dtype in SCALAR_FORMATS:
                try:
                    raw = unpack_scalar(chunk, f.dtype)
                except ParseError as e:
                    if f.required:
                        raise ParseError(f"字段 {f.name!r}：{e}") from None
                    out[f.name] = f.default
                    continue
                out[f.name] = raw * f.scale if f.scale not in (1.0, 0) else raw
            else:
                out[f.name] = self._value(f, decode_text(chunk, self.encoding).strip(), "")
        return out


def _json_path(doc: Any, path: str) -> Any:
    """``a.b[0].c`` 这种点号 + 下标路径。取不到返回 None。"""
    cur = doc
    for part in re.split(r"\.", path):
        if not part:
            continue
        m = re.match(r"^([^\[\]]*)((?:\[\d+\])*)$", part)
        if m is None:
            return None
        key, idx = m.group(1), m.group(2)
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        for i in re.findall(r"\[(\d+)\]", idx):
            if not isinstance(cur, (list, tuple)) or int(i) >= len(cur):
                return None
            cur = cur[int(i)]
    return cur


# ------------------------------------------------------------------ 格式化
FORMAT_KINDS = ("text", "json", "binary")
FORMAT_LABELS = {"text": "文本", "json": "JSON", "binary": "二进制"}

#: 字段来源的固定前缀。除此之外不解释任何表达式。
SOURCE_PREFIXES = ("out.", "var.", "node[", "field.", "literal:")
SOURCE_SPECIALS = ("status", "ok", "ng", "run_id", "flow", "duration_ms", "error",
                   "request_id", "error_code", "device", "peer")


@dataclass
class FormatField:
    name: str = "value"
    source: str = ""          # 见 SOURCE_PREFIXES / SOURCE_SPECIALS
    dtype: str = "string"     # text/json: string|int|float|bool；binary: 另加 SCALAR_FORMATS
    decimals: int = 3
    scale: float = 1.0
    width: int = 0            # 文本补齐宽度 / 二进制字符串字节数
    pad: str = " "
    align: str = "left"       # left | right
    endian: str = "big"
    default: Any = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FormatField":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def resolve_source(source: str, result: Any = None, variables: dict | None = None,
                   extra: dict | None = None) -> Any:
    """按固定取值表解析一个字段来源。认不出来的来源返回 None（不报错，发送不该因此中断）。"""
    s = str(source or "").strip()
    extra = extra or {}
    variables = variables or {}
    if not s:
        return None
    if s.startswith("literal:"):
        return s[len("literal:"):]
    if s in SOURCE_SPECIALS:
        if s in extra:
            return extra[s]
        if result is None:
            return None
        return {"status": getattr(result, "status_text", ""), "ok": int(bool(getattr(result, "passed", False))),
                "ng": int(not getattr(result, "passed", True)), "run_id": getattr(result, "run_id", 0),
                "flow": getattr(result, "flow", ""), "duration_ms": getattr(result, "duration_ms", 0.0),
                "error": getattr(result, "error", "")}.get(s)
    if s.startswith("out."):
        return (getattr(result, "outputs", {}) or {}).get(s[4:])
    if s.startswith("var."):
        return variables.get(s[4:])
    if s.startswith("field."):
        return (extra.get("fields") or {}).get(s[len("field."):])
    m = re.match(r"^node\[([^\]]+)\]\.(.+)$", s)
    if m and result is not None and hasattr(result, "node"):
        nr = result.node(m.group(1))
        return None if nr is None else (nr.outputs or {}).get(m.group(2))
    return None


@dataclass
class FormatRule:
    id: str = ""
    name: str = "format"
    kind: str = "text"
    template: str = ""             # kind == "text" 且非空时优先用模板
    separator: str = ","
    prefix: str = ""
    suffix: str = ""
    terminator: str = "\\n"
    encoding: str = "utf-8"
    json_indent: int = 0
    missing: str = ""              # 取不到值时填什么（文本模式）
    fields: list[FormatField] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in FORMAT_KINDS:
            raise ValueError(f"未知的格式化方式 {self.kind!r}，可用：{FORMAT_KINDS}")
        self.fields = [f if isinstance(f, FormatField) else FormatField.from_dict(f) for f in self.fields]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fields"] = [f.to_dict() for f in self.fields]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FormatRule":
        d = dict(d or {})
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known["fields"] = [FormatField.from_dict(f) for f in d.get("fields", [])]
        return cls(**known)

    # ---- 渲染 ----
    def render(self, result: Any = None, variables: dict | None = None,
               extra: dict | None = None) -> bytes:
        if self.kind == "binary":
            return self._render_binary(result, variables, extra)
        if self.kind == "json":
            return self._render_json(result, variables, extra)
        return self._render_text(result, variables, extra)

    def _one(self, f: FormatField, result, variables, extra) -> Any:
        v = resolve_source(f.source, result, variables, extra)
        if v is None or v == "":
            # 取不到值：数值字段退回 0（二进制必须有确定字节），文本字段退回声明的默认值
            if f.dtype in SCALAR_FORMATS:
                return 0
            return f.default
        if f.dtype in ("int", "float", *SCALAR_FORMATS):
            try:
                v = float(v) * float(f.scale or 1.0)
            except (TypeError, ValueError):
                return f.default
            if f.dtype in ("int", "int8", "uint8", "int16", "uint16", "int32", "uint32"):
                v = int(round(v))
        elif f.dtype == "bool":
            v = bool(v)
        return v

    def _text_of(self, f: FormatField, value: Any) -> str:
        if value is None or value == "":
            s = str(self.missing)
        elif f.dtype == "float":
            s = f"{float(value):.{max(0, int(f.decimals))}f}"
        elif f.dtype == "bool":
            s = "1" if value else "0"
        else:
            s = str(value)
        if f.width > 0:
            pad = (f.pad or " ")[:1]
            s = s.rjust(f.width, pad) if f.align == "right" else s.ljust(f.width, pad)
        return s

    def _render_text(self, result, variables, extra) -> bytes:
        from .manager import format_template      # 延迟导入，避免循环
        if self.template:
            text = format_template(self.template, result, variables, **(extra or {}))
            return encode_text(text, self.encoding)
        parts = [self._text_of(f, self._one(f, result, variables, extra)) for f in self.fields]
        body = (self.separator or "").join(parts)
        text = f"{self.prefix}{body}{self.suffix}"
        return encode_text(text, self.encoding) + _term(self.terminator)

    def _render_json(self, result, variables, extra) -> bytes:
        doc: dict[str, Any] = {}
        for f in self.fields:
            v = self._one(f, result, variables, extra)
            if f.dtype == "float" and v is not None:
                v = round(float(v), max(0, int(f.decimals)))
            doc[f.name] = v
        text = json.dumps(doc, ensure_ascii=False, indent=self.json_indent or None)
        return encode_text(self.prefix + text + self.suffix, self.encoding) + _term(self.terminator)

    def _render_binary(self, result, variables, extra) -> bytes:
        out = bytearray(_term(self.prefix))
        for f in self.fields:
            v = self._one(f, result, variables, extra)
            if f.dtype == "string":
                raw = encode_text("" if v is None else str(v), self.encoding)
                out += raw[:f.width].ljust(f.width, b"\x00") if f.width > 0 else raw
            elif f.dtype == "bool":
                out += b"\x01" if v else b"\x00"
            else:
                out += pack_scalar(0 if v is None else v, f.dtype, f.endian)
        out += _term(self.suffix) + _term(self.terminator)
        return bytes(out)


def _term(text: str) -> bytes:
    from .framing import unescape
    return unescape(text or "")

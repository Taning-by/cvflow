"""Modbus 数据点表：数据区、地址换算、数据类型、字节序与字序。

**地址**：内部一律用**协议地址**（从 0 开始，就是报文里那个数）。界面上同时显示**参考地址**
（40001 这一套，从 1 开始，带数据区前缀），两者的换算集中在 ``reference_text`` /
``parse_reference``，不在别处重复实现。

| 数据区 | 参考地址 | 位/字 | 读功能码 | 写功能码 | 远程可写 |
|---|---|---|---|---|---|
| ``coil``     线圈           | 1…     | 位 | 01 | 05 / 15 | 是 |
| ``discrete`` 离散输入       | 10001… | 位 | 02 | —       | 否（协议本身没有写功能码）|
| ``holding``  保持寄存器     | 40001… | 字 | 03 | 06 / 16 | 是 |
| ``input``    输入寄存器     | 30001… | 字 | 04 | —       | 否 |

**字节序与字序**是两件事，这里用一个四字母布局同时表达（以 32 位值的大端字节 A B C D 为基准）：

| 布局 | 寄存器 0 | 寄存器 1 | 含义 |
|---|---|---|---|
| ``ABCD`` | AB | CD | 标准大端，不交换 |
| ``CDAB`` | CD | AB | **寄存器间字交换**（很多 PLC 的默认） |
| ``BADC`` | BA | DC | **寄存器内字节交换** |
| ``DCBA`` | DC | BA | 两者都交换（完全小端） |

16 位类型只受字节交换影响（``ABCD``/``CDAB`` 等价，``BADC``/``DCBA`` 等价）；字符串按字节
铺进寄存器，只受字节交换影响，字交换对它没有定义。
"""
from __future__ import annotations

import re
import struct
from dataclasses import asdict, dataclass, field
from typing import Any

from .framing import decode_text, encode_text

AREAS = ("coil", "discrete", "holding", "input")

AREA_INFO: dict[str, dict[str, Any]] = {
    "coil":     {"label": "线圈 (0x)", "short": "0x", "ref_base": 1, "bits": True, "writable": True,
                 "fc_read": 1, "fc_write": 5, "fc_write_multi": 15, "max_read": 2000},
    "discrete": {"label": "离散输入 (1x)", "short": "1x", "ref_base": 10001, "bits": True, "writable": False,
                 "fc_read": 2, "fc_write": 0, "fc_write_multi": 0, "max_read": 2000},
    "input":    {"label": "输入寄存器 (3x)", "short": "3x", "ref_base": 30001, "bits": False, "writable": False,
                 "fc_read": 4, "fc_write": 0, "fc_write_multi": 0, "max_read": 125},
    "holding":  {"label": "保持寄存器 (4x)", "short": "4x", "ref_base": 40001, "bits": False, "writable": True,
                 "fc_read": 3, "fc_write": 6, "fc_write_multi": 16, "max_read": 125},
}

DTYPES = ("bool", "int16", "uint16", "int32", "uint32", "float32", "string")
DTYPE_LABELS = {"bool": "布尔", "int16": "有符号 16 位", "uint16": "无符号 16 位",
                "int32": "有符号 32 位", "uint32": "无符号 32 位", "float32": "单精度浮点",
                "string": "字符串"}
WORD_DTYPES = {"int16": 1, "uint16": 1, "int32": 2, "uint32": 2, "float32": 2}

LAYOUTS = ("ABCD", "BADC", "CDAB", "DCBA")
LAYOUT_LABELS = {"ABCD": "ABCD 大端（不交换）", "BADC": "BADC 寄存器内字节交换",
                 "CDAB": "CDAB 寄存器间字交换", "DCBA": "DCBA 字节与字都交换"}

DIRECTIONS = ("read", "write", "read_write")
DIRECTION_LABELS = {"read": "读（cvflow ← 对端）", "write": "写（cvflow → 对端）", "read_write": "读写"}

_STRUCT = {"int16": ">h", "uint16": ">H", "int32": ">i", "uint32": ">I", "float32": ">f"}


class MapError(Exception):
    pass


# ----------------------------------------------------------------- 地址换算
def reference_address(area: str, address: int) -> int:
    """协议地址（0 起）→ 参考地址（40001 这一套）。"""
    _check_area(area)
    return AREA_INFO[area]["ref_base"] + int(address)


def reference_text(area: str, address: int) -> str:
    _check_area(area)
    info = AREA_INFO[area]
    return f"{reference_address(area, address)}（{info['short']}，协议地址 {int(address)}）"


def parse_reference(text: str | int, default_area: str = "holding") -> tuple[str, int]:
    """界面/方案里写的地址 → ``(数据区, 协议地址)``。

    接受四种写法：``10``（按 default_area 的协议地址）、``40011``（参考地址）、
    ``4x10`` / ``0x·`` 这种区号前缀、``holding:10`` 这种显式写法。
    """
    if isinstance(text, int):
        return _check_area(default_area), int(text)
    s = str(text).strip().lower()
    if not s:
        raise MapError("地址为空")
    if ":" in s:
        area, _, rest = s.partition(":")
        area = {"0x": "coil", "1x": "discrete", "3x": "input", "4x": "holding"}.get(area, area)
        return _check_area(area), _int(rest)
    m = re.match(r"^([014])x\s*(\d+)$", s)
    if m:
        return {"0": "coil", "1": "discrete", "3": "input", "4": "holding"}[m.group(1)], int(m.group(2))
    n = _int(s)
    for area in ("holding", "input", "discrete"):          # 参考地址区间
        base = AREA_INFO[area]["ref_base"]
        if base <= n <= base + 65535:
            return area, n - base
    return _check_area(default_area), n


def _int(s: str) -> int:
    try:
        return int(str(s).strip(), 0)
    except ValueError:
        raise MapError(f"{s!r} 不是地址") from None


def _check_area(area: str) -> str:
    if area not in AREA_INFO:
        raise MapError(f"未知的数据区 {area!r}，可用：{', '.join(AREAS)}")
    return area


# ----------------------------------------------------------------- 字节序
def words_for(dtype: str, count: int = 1, length: int = 0) -> int:
    """一个数据点占几个寄存器（位数据区按"几个位"算）。"""
    if dtype == "bool":
        return max(1, int(count))
    if dtype == "string":
        n = int(length or 0)
        if n <= 0:
            raise MapError("字符串数据点必须指定字节数")
        return (n + 1) // 2
    if dtype not in WORD_DTYPES:
        raise MapError(f"未知的数据类型 {dtype!r}，可用：{', '.join(DTYPES)}")
    return WORD_DTYPES[dtype] * max(1, int(count))


def _apply_layout(words: list[int], layout: str, swap_words: bool) -> list[int]:
    """把标准大端的寄存器序列按布局重排。``swap_words`` 为假时只做寄存器内字节交换。"""
    if layout not in LAYOUTS:
        raise MapError(f"未知的字节序布局 {layout!r}，可用：{', '.join(LAYOUTS)}")
    out = list(words)
    if swap_words and layout in ("CDAB", "DCBA") and len(out) == 2:
        out = [out[1], out[0]]
    if layout in ("BADC", "DCBA"):
        out = [((w & 0xFF) << 8) | ((w >> 8) & 0xFF) for w in out]
    return out


def encode_value(value: Any, dtype: str, layout: str = "ABCD", length: int = 0,
                 encoding: str = "ascii") -> list[int]:
    """一个值 → 寄存器列表（``bool`` 返回 ``[0]`` 或 ``[1]``，由调用方决定走线圈还是寄存器）。"""
    if dtype == "bool":
        return [1 if _truthy(value) else 0]
    if dtype == "string":
        n = int(length or 0)
        if n <= 0:
            raise MapError("字符串数据点必须指定字节数")
        raw = encode_text("" if value is None else str(value), encoding)[:n].ljust(n, b"\x00")
        if n % 2:
            raw += b"\x00"
        words = list(struct.unpack(f">{len(raw) // 2}H", raw))
        return _apply_layout(words, layout, swap_words=False)
    if dtype not in _STRUCT:
        raise MapError(f"未知的数据类型 {dtype!r}")
    if dtype == "float32":
        packed = struct.pack(_STRUCT[dtype], float(value))
    else:
        v = int(round(float(value)))
        lo, hi = _int_limits(dtype)
        packed = struct.pack(_STRUCT[dtype], max(lo, min(hi, v)))
    words = list(struct.unpack(f">{len(packed) // 2}H", packed))
    return _apply_layout(words, layout, swap_words=True)


def decode_value(words: list[int], dtype: str, layout: str = "ABCD", length: int = 0,
                 encoding: str = "ascii") -> Any:
    """寄存器列表 → 值。``words`` 必须正好是 ``words_for`` 个寄存器。"""
    if dtype == "bool":
        return bool(words and words[0])
    regs = _apply_layout(list(words), layout, swap_words=(dtype != "string"))
    if dtype == "string":
        raw = struct.pack(f">{len(regs)}H", *[w & 0xFFFF for w in regs])
        n = int(length or len(raw))
        return decode_text(raw[:n], encoding).split("\x00", 1)[0]
    need = WORD_DTYPES[dtype]
    if len(regs) < need:
        raise MapError(f"{dtype} 需要 {need} 个寄存器，只有 {len(regs)} 个")
    raw = struct.pack(f">{need}H", *[w & 0xFFFF for w in regs[:need]])
    return struct.unpack(_STRUCT[dtype], raw)[0]


def layout_example(dtype: str, layout: str) -> str:
    """给界面用的字节示例：说明这个布局实际发出去的字节顺序。"""
    if dtype == "bool":
        return "位数据不涉及字节序"
    if dtype in ("int16", "uint16"):
        order = "A B" if layout in ("ABCD", "CDAB") else "B A"
        return f"寄存器0 = {order}（大端值 0x1234 → {'12 34' if order == 'A B' else '34 12'}）"
    if dtype == "string":
        order = "原序" if layout in ("ABCD", "CDAB") else "每个寄存器内两字节交换"
        return f'字符串按字节铺开，{order}（"ABCD" → {"41 42 43 44" if order == "原序" else "42 41 44 43"}）'
    mapping = {"ABCD": ("A B", "C D", "12 34 56 78"), "BADC": ("B A", "D C", "34 12 78 56"),
               "CDAB": ("C D", "A B", "56 78 12 34"), "DCBA": ("D C", "B A", "78 56 34 12")}
    r0, r1, demo = mapping[layout]
    return f"寄存器0 = {r0}，寄存器1 = {r1}（大端值 0x12345678 → {demo}）"


def _int_limits(dtype: str) -> tuple[int, int]:
    return {"int16": (-32768, 32767), "uint16": (0, 65535),
            "int32": (-2147483648, 2147483647), "uint32": (0, 4294967295)}[dtype]


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "ok")
    return bool(value)


# ----------------------------------------------------------------- 数据点
@dataclass
class DataPoint:
    """一个命名的 Modbus 数据点。名字在一个设备内唯一，流程和规则都按名字引用。"""

    name: str = "point"
    area: str = "holding"
    address: int = 0
    dtype: str = "int16"
    count: int = 1                  # bool 可以是连续多个位；其它类型 1
    length: int = 0                 # string 的字节数
    direction: str = "read"
    poll_ms: int = 0                # 0 = 不轮询（只在被读写时访问）
    scale: float = 1.0              # 读：寄存器值 × scale；写：值 ÷ scale
    layout: str = "ABCD"
    encoding: str = "ascii"         # string 用
    variable: str = ""              # 绑定的全局变量；读到新值时写入
    description: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        self.validate()

    # ---- 校验 ----
    def validate(self) -> None:
        if not str(self.name).strip():
            raise MapError("数据点名称不能为空")
        _check_area(self.area)
        if self.dtype not in DTYPES:
            raise MapError(f"数据点 {self.name!r}：未知的数据类型 {self.dtype!r}")
        if self.direction not in DIRECTIONS:
            raise MapError(f"数据点 {self.name!r}：未知的读写方向 {self.direction!r}")
        if self.layout not in LAYOUTS:
            raise MapError(f"数据点 {self.name!r}：未知的字节序 {self.layout!r}")
        if not 0 <= int(self.address) <= 65535:
            raise MapError(f"数据点 {self.name!r}：地址 {self.address} 超出 0…65535")
        if int(self.count) < 1:
            raise MapError(f"数据点 {self.name!r}：数量必须 ≥ 1")
        bits = AREA_INFO[self.area]["bits"]
        if bits and self.dtype not in ("bool",):
            raise MapError(f"数据点 {self.name!r}：{AREA_INFO[self.area]['label']} 只能放布尔值")
        if not bits and self.dtype == "bool" and int(self.count) > 1:
            raise MapError(f"数据点 {self.name!r}：寄存器区的布尔值数量只能是 1")
        if self.dtype == "string" and int(self.length) <= 0:
            raise MapError(f"数据点 {self.name!r}：字符串必须指定字节数")
        if self.writes and not AREA_INFO[self.area]["writable"]:
            raise MapError(f"数据点 {self.name!r}：{AREA_INFO[self.area]['label']}"
                           f"按 Modbus 协议不可写，只能把方向设为“读”")
        end = int(self.address) + self.words - 1
        if end > 65535:
            raise MapError(f"数据点 {self.name!r}：占到地址 {end}，超出 65535")
        if int(self.poll_ms) < 0:
            raise MapError(f"数据点 {self.name!r}：轮询周期不能是负数")

    # ---- 派生 ----
    @property
    def words(self) -> int:
        return words_for(self.dtype, self.count, self.length)

    @property
    def reads(self) -> bool:
        return self.direction in ("read", "read_write")

    @property
    def writes(self) -> bool:
        return self.direction in ("write", "read_write")

    @property
    def bits(self) -> bool:
        return bool(AREA_INFO[self.area]["bits"])

    @property
    def reference(self) -> str:
        return reference_text(self.area, self.address)

    def layout_hint(self) -> str:
        return layout_example(self.dtype, self.layout)

    # ---- 值换算 ----
    def decode(self, words: list[int] | list[bool]) -> Any:
        if self.bits:
            vals = [bool(w) for w in words[:max(1, int(self.count))]]
            return vals[0] if int(self.count) == 1 else vals
        v = decode_value([int(w) for w in words], self.dtype, self.layout, self.length, self.encoding)
        if self.dtype in ("int16", "uint16", "int32", "uint32", "float32") and self.scale not in (1.0, 0):
            return v * float(self.scale)
        return v

    def encode(self, value: Any) -> list[int]:
        if self.bits:
            if isinstance(value, (list, tuple)):
                return [1 if _truthy(v) else 0 for v in value]
            return [1 if _truthy(value) else 0]
        v = value
        if self.dtype in ("int16", "uint16", "int32", "uint32", "float32") and self.scale not in (1.0, 0):
            v = float(value) / float(self.scale)
        return encode_value(v, self.dtype, self.layout, self.length, self.encoding)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DataPoint":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass
class PointValue:
    """数据点的最新值。``valid`` 为假表示还没读到、或连接断了之后已失效。"""

    value: Any = None
    words: list[int] = field(default_factory=list)
    updated: float = 0.0
    valid: bool = False
    stale: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {"value": self.value, "updated": self.updated, "valid": self.valid,
                "stale": self.stale, "error": self.error}


@dataclass
class ReadBlock:
    """把相邻数据点合并成一次请求。``points`` 里是 ``(数据点, 块内起始偏移)``。"""

    area: str
    address: int
    count: int
    points: list[tuple[DataPoint, int]] = field(default_factory=list)


class DataPointTable:
    """一个设备的数据点集合：按名字查、校验唯一性、把相邻点合并成读请求。"""

    def __init__(self, points: list[DataPoint] | None = None) -> None:
        self._points: list[DataPoint] = []
        for p in points or []:
            self.add(p)

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    @property
    def points(self) -> list[DataPoint]:
        return list(self._points)

    def add(self, point: DataPoint) -> DataPoint:
        point.validate()
        if any(p.name == point.name for p in self._points):
            raise MapError(f"数据点名称 {point.name!r} 重复")
        self._points.append(point)
        return point

    def replace(self, index: int, point: DataPoint) -> None:
        point.validate()
        if any(i != index and p.name == point.name for i, p in enumerate(self._points)):
            raise MapError(f"数据点名称 {point.name!r} 重复")
        self._points[index] = point

    def remove(self, name: str) -> None:
        self._points = [p for p in self._points if p.name != name]

    def get(self, name: str) -> DataPoint | None:
        return next((p for p in self._points if p.name == name), None)

    def names(self) -> list[str]:
        return [p.name for p in self._points]

    def to_list(self) -> list[dict]:
        return [p.to_dict() for p in self._points]

    @classmethod
    def from_list(cls, items: list[dict] | None) -> "DataPointTable":
        t = cls()
        for d in items or []:
            t.add(DataPoint.from_dict(d))
        return t

    # ---- 轮询编排 ----
    def read_blocks(self, max_gap: int = 8, only_polled: bool = False) -> list[ReadBlock]:
        """把要读的数据点按数据区排序、合并成尽量少的请求。

        ``max_gap`` 是允许"顺手多读"的空洞寄存器数：地址相邻或只隔几个的点合并成一次请求比
        分开两次更快。单次请求不超过该数据区的协议上限（寄存器 125 个、位 2000 个）。
        """
        blocks: list[ReadBlock] = []
        for area in AREAS:
            pts = [p for p in self._points
                   if p.enabled and p.area == area and p.reads and (p.poll_ms > 0 or not only_polled)]
            if not pts:
                continue
            pts.sort(key=lambda p: (int(p.address), p.name))
            limit = int(AREA_INFO[area]["max_read"])
            cur: ReadBlock | None = None
            for p in pts:
                start, end = int(p.address), int(p.address) + p.words
                if cur is not None:
                    new_end = max(cur.address + cur.count, end)
                    if start <= cur.address + cur.count + max_gap and new_end - cur.address <= limit:
                        cur.count = new_end - cur.address
                        cur.points.append((p, start - cur.address))
                        continue
                cur = ReadBlock(area=area, address=start, count=end - start, points=[(p, 0)])
                blocks.append(cur)
        return blocks

    def validate_all(self) -> list[str]:
        errors, seen = [], set()
        for p in self._points:
            try:
                p.validate()
            except MapError as e:
                errors.append(str(e))
            if p.name in seen:
                errors.append(f"数据点名称 {p.name!r} 重复")
            seen.add(p.name)
        return errors

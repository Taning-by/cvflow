"""报文分帧与文本编码。

TCP 和串口都是**字节流**：一次 ``recv`` 既可能只拿到半条报文（半包），也可能一次拿到好几条
（粘包）。所以接收路径上统一经过 ``Framer``：喂进任意长度的字节块，吐出**完整报文**的列表。

四种分帧方式（``framing`` 配置项）：

* ``delimiter``     分隔符，例如 ``\\n`` / ``\\r\\n``。最常见的文本协议。
* ``fixed``         定长，每 ``frame_length`` 字节一条。
* ``length_prefix`` 长度前缀：报文头里有一个长度字段，需指明它的偏移、字节数、字节序，
                    以及长度**算的是什么**（见 ``LENGTH_SEMANTICS``）。
* ``raw``           不分帧，一次 ``recv`` 直接当一条。**不保证报文完整性**，只适合调试或
                    对端保证一次写一条的场合。

另有三道保护，防止畸形对端把内存吃光或让解析永久错位：
``max_frame`` 单条上限、``rx_buffer_limit`` 缓冲上限、``frame_timeout_s`` 不完整报文超时。

UDP 不走 Framer：一个数据报天然就是一条报文，边界由协议本身保证。
"""
from __future__ import annotations

import re
import struct
import time
from dataclasses import dataclass, field
from typing import Any

FRAMING_KINDS = ("delimiter", "fixed", "length_prefix", "raw")
FRAMING_LABELS = {
    "delimiter": "分隔符（如 LF / CRLF）",
    "fixed": "定长",
    "length_prefix": "长度前缀",
    "raw": "原始字节（不保证完整性）",
}

#: 长度字段算的是哪一段字节。
LENGTH_SEMANTICS = ("payload", "total", "after_field")
LENGTH_SEMANTIC_LABELS = {
    "payload": "仅数据区（报文头之后的字节数）",
    "total": "整条报文（含报文头）",
    "after_field": "长度字段之后的全部字节",
}

TEXT_ENCODINGS = ("utf-8", "ascii", "gbk", "gb18030", "latin-1")

_HEX_SPLIT = re.compile(r"[\s,;:]+")


class FramingError(Exception):
    """分帧无法继续（报文超限、长度字段非法）。调用方应重置缓冲并记录错误。"""


# --------------------------------------------------------------------- 编解码
def unescape(text: str) -> bytes:
    r"""把界面里写的 ``\n`` ``\r\n`` ``\x02`` 变成真正的字节。

    走 ``unicode_escape`` 再按 latin-1 落字节，所以 ``\xNN`` 得到的就是那一个字节。
    非转义的非 ASCII 字符（有人直接填中文分隔符）按 UTF-8 保留。
    """
    if not text:
        return b""
    try:
        return text.encode("utf-8").decode("unicode_escape").encode("latin-1")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text.encode("utf-8", errors="replace")


def to_hex(data: bytes, sep: str = " ") -> str:
    return data.hex(sep) if sep else data.hex()


def parse_hex(text: str) -> bytes:
    """解析用户输入的十六进制：``01 02 0A`` / ``0102 0a`` / ``0x01,0x02`` 都接受。"""
    cleaned = "".join(t[2:] if t.lower().startswith("0x") else t
                      for t in _HEX_SPLIT.split(text.strip()) if t)
    if not cleaned:
        return b""
    if len(cleaned) % 2:
        raise ValueError("十六进制位数必须是偶数（每两位一个字节）")
    try:
        return bytes.fromhex(cleaned)
    except ValueError as e:
        raise ValueError(f"不是合法的十六进制：{e}") from None


def decode_text(data: bytes, encoding: str = "utf-8") -> str:
    """解码成文本。解不开时退回十六进制显示，而不是抛异常——调试面板要永远有东西可看。"""
    try:
        return data.decode(encoding or "utf-8")
    except (UnicodeDecodeError, LookupError):
        return to_hex(data)


def encode_text(text: str, encoding: str = "utf-8") -> bytes:
    try:
        return text.encode(encoding or "utf-8", errors="replace")
    except LookupError:
        return text.encode("utf-8", errors="replace")


# --------------------------------------------------------------------- 配置
@dataclass
class FramingConfig:
    kind: str = "delimiter"
    delimiter: str = "\\n"          # 允许转义写法
    frame_length: int = 16          # kind == "fixed"
    length_offset: int = 0          # 长度字段在报文头里的字节偏移
    length_size: int = 2            # 长度字段字节数：1 / 2 / 4
    length_endian: str = "big"      # big / little
    length_counts: str = "payload"  # LENGTH_SEMANTICS
    header_size: int = 0            # 报文头总字节数；0 表示 length_offset + length_size
    keep_header: bool = False       # 交给上层的报文是否带报文头
    max_frame: int = 65536
    buffer_limit: int = 1_048_576
    timeout_s: float = 0.0          # 不完整报文超时，0 关闭

    def __post_init__(self) -> None:
        if self.kind not in FRAMING_KINDS:
            raise ValueError(f"未知的分帧方式 {self.kind!r}，可用：{FRAMING_KINDS}")
        if self.length_counts not in LENGTH_SEMANTICS:
            raise ValueError(f"未知的长度含义 {self.length_counts!r}，可用：{LENGTH_SEMANTICS}")
        if self.kind == "length_prefix":
            if self.length_size not in (1, 2, 4):
                raise ValueError("长度字段只能是 1、2 或 4 个字节")
            if self.length_offset < 0:
                raise ValueError("长度字段偏移不能是负数")
            if self.header_size and self.header_size < self.length_offset + self.length_size:
                raise ValueError("报文头字节数小于长度字段的结束位置")
        if self.kind == "fixed" and self.frame_length < 1:
            raise ValueError("定长分帧的报文长度必须 ≥ 1")
        if self.max_frame < 1:
            raise ValueError("单条报文上限必须 ≥ 1")
        if self.buffer_limit < self.max_frame:
            self.buffer_limit = self.max_frame

    # ---- 派生量 ----
    @property
    def header_bytes(self) -> int:
        if self.kind != "length_prefix":
            return 0
        return self.header_size or (self.length_offset + self.length_size)

    @property
    def delimiter_bytes(self) -> bytes:
        return unescape(self.delimiter)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "FramingConfig":
        """从设备配置字典构造。

        老方案文件里只有 ``terminator``：有 ``terminator`` 而没有 ``framing`` 时，
        空串按 ``raw``、非空按 ``delimiter``，和改造前的行为一致。
        """
        kind = str(cfg.get("framing") or "").strip()
        term = cfg.get("terminator", None)
        if not kind:
            kind = "delimiter" if (term is None or str(term) != "") else "raw"
        return cls(
            kind=kind,
            delimiter="\\n" if term is None else str(term),
            frame_length=int(cfg.get("frame_length", 16) or 16),
            length_offset=int(cfg.get("length_offset", 0) or 0),
            length_size=int(cfg.get("length_size", 2) or 2),
            length_endian=str(cfg.get("length_endian", "big") or "big"),
            length_counts=str(cfg.get("length_counts", "payload") or "payload"),
            header_size=int(cfg.get("header_size", 0) or 0),
            keep_header=bool(cfg.get("keep_header", False)),
            max_frame=int(cfg.get("max_frame", 65536) or 65536),
            buffer_limit=int(cfg.get("rx_buffer_limit", 1_048_576) or 1_048_576),
            timeout_s=float(cfg.get("frame_timeout_s", 0.0) or 0.0),
        )

    def describe(self) -> str:
        if self.kind == "delimiter":
            d = self.delimiter_bytes
            return f"分隔符 {to_hex(d) or '(空)'}"
        if self.kind == "fixed":
            return f"定长 {self.frame_length} 字节"
        if self.kind == "length_prefix":
            return (f"长度前缀：偏移 {self.length_offset}，{self.length_size} 字节，"
                    f"{'大端' if self.length_endian == 'big' else '小端'}，"
                    f"{LENGTH_SEMANTIC_LABELS[self.length_counts]}，报文头 {self.header_bytes} 字节")
        return "原始字节"


# --------------------------------------------------------------------- 分帧器
@dataclass
class _Stat:
    frames: int = 0
    dropped_bytes: int = 0
    errors: list[str] = field(default_factory=list)


class Framer:
    """把字节流切成完整报文。线程不安全：每条连接各用一个实例。"""

    def __init__(self, config: FramingConfig | None = None) -> None:
        self.config = config or FramingConfig()
        self._buf = bytearray()
        self._since = 0.0            # 缓冲里攒下第一个不完整字节的时刻
        self.stat = _Stat()

    # ---- 状态 ----
    @property
    def pending(self) -> int:
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()
        self._since = 0.0

    def take_errors(self) -> list[str]:
        errs, self.stat.errors = self.stat.errors, []
        return errs

    def _fail(self, msg: str, drop: int | None = None) -> None:
        self.stat.errors.append(msg)
        if drop is None:
            self.stat.dropped_bytes += len(self._buf)
            self._buf.clear()
            self._since = 0.0
        else:
            self.stat.dropped_bytes += drop
            del self._buf[:drop]

    # ---- 主流程 ----
    def feed(self, chunk: bytes, now: float | None = None) -> list[bytes]:
        """喂入一块字节，返回这次能切出来的完整报文（可能 0 条，也可能多条）。"""
        now = time.monotonic() if now is None else now
        cfg = self.config
        if cfg.kind == "raw":
            if not chunk:
                return []
            if len(chunk) > cfg.max_frame:
                self.stat.errors.append(f"报文 {len(chunk)} 字节超过上限 {cfg.max_frame}，已截断")
                chunk = chunk[:cfg.max_frame]
            self.stat.frames += 1
            return [bytes(chunk)]

        if chunk:
            if not self._buf:
                self._since = now
            self._buf += chunk
        frames: list[bytes] = []
        while True:
            frame = self._next(now)
            if frame is None:
                break
            frames.append(frame)
            self.stat.frames += 1
        if not self._buf:
            self._since = 0.0
        elif frames:
            self._since = now        # 刚切出过报文，剩下的是新报文的开头，超时重新计时
        self._guard(now)
        return frames

    def check_timeout(self, now: float | None = None) -> list[str]:
        """定时调用：不完整报文攒太久就丢掉，避免一条坏报文把后面全部拖错位。"""
        now = time.monotonic() if now is None else now
        cfg = self.config
        if cfg.timeout_s > 0 and self._buf and self._since and now - self._since >= cfg.timeout_s:
            self._fail(f"不完整报文超过 {cfg.timeout_s:g} 秒未收齐（{len(self._buf)} 字节），已丢弃")
        return self.take_errors()

    # ---- 内部 ----
    def _guard(self, now: float) -> None:
        cfg = self.config
        if len(self._buf) > cfg.buffer_limit:
            self._fail(f"接收缓冲超过上限 {cfg.buffer_limit} 字节，已清空（对端是否一直不发分隔符？）")
            return
        if cfg.kind == "delimiter" and len(self._buf) > cfg.max_frame:
            # 分隔符模式下还能自救：丢掉超限的那一段，保留末尾，等下一个分隔符重新对齐
            keep = max(0, len(cfg.delimiter_bytes) - 1)
            drop = len(self._buf) - keep
            self._fail(f"{drop} 字节内未出现分隔符，超过单条上限 {cfg.max_frame}，已丢弃", drop=drop)
        if cfg.timeout_s > 0 and self._buf and self._since and now - self._since >= cfg.timeout_s:
            self._fail(f"不完整报文超过 {cfg.timeout_s:g} 秒未收齐（{len(self._buf)} 字节），已丢弃")

    def _next(self, now: float) -> bytes | None:
        cfg = self.config
        if cfg.kind == "delimiter":
            term = cfg.delimiter_bytes
            if not term:                       # 配了分隔符模式却留空：退化成原始模式
                if not self._buf:
                    return None
                out = bytes(self._buf)
                self._buf.clear()
                return out
            idx = self._buf.find(term)
            if idx < 0:
                return None
            frame = bytes(self._buf[:idx])
            del self._buf[:idx + len(term)]
            return frame if frame else self._next(now)   # 跳过空帧（连续两个分隔符）
        if cfg.kind == "fixed":
            n = cfg.frame_length
            if len(self._buf) < n:
                return None
            frame = bytes(self._buf[:n])
            del self._buf[:n]
            return frame
        if cfg.kind == "length_prefix":
            return self._next_length_prefixed()
        return None

    def _next_length_prefixed(self) -> bytes | None:
        cfg = self.config
        head = cfg.header_bytes
        need_for_len = cfg.length_offset + cfg.length_size
        if len(self._buf) < max(head, need_for_len):
            return None
        raw = bytes(self._buf[cfg.length_offset:need_for_len])
        fmt = {1: "B", 2: "H", 4: "I"}[cfg.length_size]
        declared = struct.unpack((">" if cfg.length_endian == "big" else "<") + fmt, raw)[0]
        if cfg.length_counts == "payload":
            total = head + declared
        elif cfg.length_counts == "total":
            total = declared
        else:  # after_field
            total = need_for_len + declared
        if total < head or total > cfg.max_frame:
            # 长度字段不可信：继续按它走只会一路错位，丢弃整个缓冲重新同步
            self._fail(f"长度字段声明 {declared}（整条 {total} 字节），超出 [{head}, {cfg.max_frame}] 范围，"
                       f"已丢弃缓冲重新同步")
            return None
        if len(self._buf) < total:
            return None
        frame = bytes(self._buf[:total]) if cfg.keep_header else bytes(self._buf[head:total])
        del self._buf[:total]
        return frame


def build_length_prefixed(payload: bytes, config: FramingConfig) -> bytes:
    """按长度前缀配置给数据区加上报文头（发送方向）。"""
    cfg = config
    head = cfg.header_bytes
    need = cfg.length_offset + cfg.length_size
    if cfg.length_counts == "payload":
        value = len(payload)
    elif cfg.length_counts == "total":
        value = head + len(payload)
    else:
        value = head + len(payload) - need
    fmt = {1: "B", 2: "H", 4: "I"}[cfg.length_size]
    limit = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}[cfg.length_size]
    if value > limit:
        raise FramingError(f"报文 {len(payload)} 字节，长度字段（{cfg.length_size} 字节）装不下 {value}")
    packed = struct.pack((">" if cfg.length_endian == "big" else "<") + fmt, value)
    header = bytearray(head)
    header[cfg.length_offset:need] = packed
    return bytes(header) + payload


def frame_for_send(payload: bytes, config: FramingConfig) -> bytes:
    """按分帧配置把要发的数据区补成一条完整报文。"""
    cfg = config
    if cfg.kind == "delimiter":
        term = cfg.delimiter_bytes
        return payload if (not term or payload.endswith(term)) else payload + term
    if cfg.kind == "fixed":
        if len(payload) > cfg.frame_length:
            raise FramingError(f"定长分帧为 {cfg.frame_length} 字节，报文有 {len(payload)} 字节")
        return payload.ljust(cfg.frame_length, b"\x00")
    if cfg.kind == "length_prefix":
        return build_length_prefixed(payload, cfg)
    return payload

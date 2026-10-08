"""分帧、编码、报文解析与结果格式化、Modbus 数据点映射的黑盒测试。

对分帧只从"喂字节 → 吐报文"这一个口观察：半包、粘包、连续多帧、畸形报文、超限与超时。
对数据点映射逐个布局核对**实际字节**，不只是往返一致（往返一致掩盖得了字节序写错）。
"""
from __future__ import annotations

import json
import struct

import pytest

from cvflow.comm.framing import (Framer, FramingConfig, FramingError, build_length_prefixed,
                                 decode_text, frame_for_send, parse_hex, to_hex, unescape)
from cvflow.comm.modbus_map import (AREA_INFO, DataPoint, DataPointTable, MapError, decode_value,
                                    encode_value, layout_example, parse_reference,
                                    reference_address, reference_text, words_for)
from cvflow.comm.parsing import (FormatField, FormatRule, ParseError, ParseField, ParseRule,
                                 pack_scalar, resolve_source, unpack_scalar)


# ============================================================ 分帧
def test_framing_delimiter_handles_partial_glued_and_multiple():
    f = Framer(FramingConfig(kind="delimiter", delimiter="\\n"))
    assert f.feed(b"AB") == []                       # 半包：一条报文分两次到
    assert f.feed(b"C\n") == [b"ABC"]
    assert f.feed(b"D\nE\nF") == [b"D", b"E"]        # 粘包：一次 recv 里三条半
    assert f.pending == 1
    assert f.feed(b"\n") == [b"F"]
    assert f.feed(b"\n\n\nX\n") == [b"X"]            # 连续分隔符产生的空帧被跳过
    assert f.feed(b"") == []


def test_framing_delimiter_crlf_and_custom_bytes():
    f = Framer(FramingConfig(kind="delimiter", delimiter="\\r\\n"))
    assert f.feed(b"A\nB\r\n") == [b"A\nB"]          # 单独的 \n 不是分隔符
    f2 = Framer(FramingConfig(kind="delimiter", delimiter="\\x03"))
    assert f2.feed(b"\x02hi\x03\x02there\x03") == [b"\x02hi", b"\x02there"]


def test_framing_fixed_length():
    f = Framer(FramingConfig(kind="fixed", frame_length=4))
    assert f.feed(b"ABCDEFG") == [b"ABCD"]
    assert f.feed(b"H") == [b"EFGH"]
    assert f.feed(b"12345678") == [b"1234", b"5678"]


@pytest.mark.parametrize("counts,extra_header", [("payload", 0), ("total", 0), ("after_field", 0),
                                                 ("payload", 6)])
def test_framing_length_prefix_semantics(counts, extra_header):
    cfg = FramingConfig(kind="length_prefix", length_offset=2, length_size=2,
                        length_counts=counts, header_size=extra_header)
    payload = b"hello"
    frame = build_length_prefixed(payload, cfg)
    f = Framer(cfg)
    # 一次性喂进去
    assert f.feed(frame) == [payload]
    # 逐字节喂进去，结果必须一样（半包的极端情形）
    f.reset()
    out = []
    for i in range(len(frame)):
        out += f.feed(frame[i:i + 1])
    assert out == [payload]
    # 两条粘在一起
    f.reset()
    assert f.feed(frame + frame) == [payload, payload]


def test_framing_length_prefix_little_endian_and_keep_header():
    cfg = FramingConfig(kind="length_prefix", length_offset=0, length_size=4,
                        length_endian="little", keep_header=True)
    frame = build_length_prefixed(b"xy", cfg)
    assert frame == struct.pack("<I", 2) + b"xy"
    assert Framer(cfg).feed(frame) == [frame]        # keep_header：报文带上报文头交给上层


def test_framing_length_prefix_bad_length_resyncs_and_reports():
    cfg = FramingConfig(kind="length_prefix", length_offset=0, length_size=2, max_frame=64)
    f = Framer(cfg)
    assert f.feed(struct.pack(">H", 9999) + b"junk") == []
    errs = f.take_errors()
    assert errs and "长度字段" in errs[0] and f.pending == 0   # 丢弃缓冲重新同步
    assert f.feed(build_length_prefixed(b"ok", cfg)) == [b"ok"]


def test_framing_raw_mode_is_one_recv_one_frame():
    f = Framer(FramingConfig(kind="raw"))
    assert f.feed(b"anything\x00\xff") == [b"anything\x00\xff"]
    assert f.feed(b"") == []


def test_framing_max_frame_and_buffer_limits():
    f = Framer(FramingConfig(kind="delimiter", delimiter="\\n", max_frame=16, buffer_limit=16))
    assert f.feed(b"x" * 40) == []
    errs = f.take_errors()
    assert errs and ("上限" in errs[0])
    assert f.feed(b"short\n") == [b"short"]          # 丢掉超限那段之后还能继续用


def test_framing_incomplete_timeout():
    cfg = FramingConfig(kind="delimiter", delimiter="\\n", timeout_s=0.5)
    f = Framer(cfg)
    assert f.feed(b"half", now=100.0) == []
    assert f.check_timeout(now=100.2) == []
    errs = f.check_timeout(now=100.6)
    assert errs and "未收齐" in errs[0] and f.pending == 0


def test_framing_for_send_adds_terminator_or_prefix():
    d = FramingConfig(kind="delimiter", delimiter="\\r\\n")
    assert frame_for_send(b"A", d) == b"A\r\n"
    assert frame_for_send(b"A\r\n", d) == b"A\r\n"    # 已经带了就不重复加
    fx = FramingConfig(kind="fixed", frame_length=4)
    assert frame_for_send(b"AB", fx) == b"AB\x00\x00"
    with pytest.raises(FramingError):
        frame_for_send(b"ABCDE", fx)
    lp = FramingConfig(kind="length_prefix", length_size=1, max_frame=300)
    assert frame_for_send(b"hi", lp) == b"\x02hi"
    with pytest.raises(FramingError):
        frame_for_send(b"x" * 300, lp)


def test_framing_config_from_legacy_terminator():
    assert FramingConfig.from_config({"terminator": "\\n"}).kind == "delimiter"
    assert FramingConfig.from_config({"terminator": ""}).kind == "raw"       # 老方案空分隔符=不分帧
    assert FramingConfig.from_config({}).kind == "delimiter"
    cfg = FramingConfig.from_config({"framing": "fixed", "frame_length": 8})
    assert cfg.kind == "fixed" and cfg.frame_length == 8


def test_encoding_helpers():
    assert unescape("\\r\\n") == b"\r\n" and unescape("\\x02") == b"\x02"
    assert parse_hex("0x01, 02 0a") == b"\x01\x02\x0a"
    assert parse_hex("") == b""
    with pytest.raises(ValueError):
        parse_hex("012")
    assert to_hex(b"\x01\xab") == "01 ab"
    assert decode_text("型号".encode("gbk"), "gbk") == "型号"
    assert decode_text(b"\xff\xfe", "utf-8") == "ff fe"      # 解不开退回十六进制


# ============================================================ 解析
def test_parse_delimited_with_types_and_missing_field():
    rule = ParseRule(kind="delimited", separator=",", fields=[
        ParseField(name="cmd", index=0), ParseField(name="request_id", index=1, dtype="int"),
        ParseField(name="limit", index=2, dtype="float", required=False, default=0.0)])
    assert rule.parse(b"TRIGGER,1001,2.5") == {"cmd": "TRIGGER", "request_id": 1001, "limit": 2.5}
    assert rule.parse(b"TRIGGER,1001")["limit"] == 0.0
    with pytest.raises(ParseError) as e:
        rule.parse(b"TRIGGER")
    assert "request_id" in str(e.value)
    with pytest.raises(ParseError):
        rule.parse(b"TRIGGER,abc")                   # 不是数字 → 参数错误


def test_parse_regex_named_and_numbered_groups():
    rule = ParseRule(kind="regex", pattern=r"^(?P<cmd>[A-Z]+),(?P<id>\d+)$", fields=[
        ParseField(name="cmd", group="cmd"), ParseField(name="request_id", group="id", dtype="int")])
    assert rule.parse(b"TRIGGER,7") == {"cmd": "TRIGGER", "request_id": 7}
    with pytest.raises(ParseError):
        rule.parse(b"nope")
    numbered = ParseRule(kind="regex", pattern=r"(\w+)-(\d+)", fields=[
        ParseField(name="a", group="1"), ParseField(name="b", group="2", dtype="int")])
    assert numbered.parse(b"part-42") == {"a": "part", "b": 42}
    broken = ParseRule(kind="regex", pattern="(", fields=[ParseField(name="x", group="1")])
    with pytest.raises(ParseError):
        broken.parse(b"x")


def test_parse_json_paths_and_fixed_fields():
    rule = ParseRule(kind="json", fields=[
        ParseField(name="id", path="req.id", dtype="int"),
        ParseField(name="model", path="req.items[1].name"),
        ParseField(name="gone", path="nope", required=False, default="-")])
    doc = json.dumps({"req": {"id": 5, "items": [{"name": "a"}, {"name": "b"}]}}).encode()
    assert rule.parse(doc) == {"id": 5, "model": "b", "gone": "-"}
    with pytest.raises(ParseError):
        rule.parse(b"{not json")
    fixed = ParseRule(kind="fixed_fields", fields=[
        ParseField(name="hdr", start=0, length=2),
        ParseField(name="n", start=2, length=2, dtype="uint16"),
        ParseField(name="f", start=4, length=4, dtype="float32")])
    raw = b"ST" + struct.pack(">H", 513) + struct.pack(">f", 1.5)
    assert fixed.parse(raw) == {"hdr": "ST", "n": 513, "f": 1.5}


def test_parse_bool_and_whole():
    r = ParseRule(kind="whole", fields=[ParseField(name="msg")])
    assert r.parse(b"  hello  ") == {"msg": "hello"}
    b = ParseRule(kind="delimited", fields=[ParseField(name="flag", index=0, dtype="bool")])
    assert b.parse(b"OK")["flag"] is True and b.parse(b"NG")["flag"] is False
    with pytest.raises(ParseError):
        b.parse(b"maybe")


def test_parse_rule_roundtrip():
    rule = ParseRule(kind="delimited", name="r", fields=[ParseField(name="a", index=1, dtype="int")])
    again = ParseRule.from_dict(json.loads(json.dumps(rule.to_dict())))
    assert again.parse(b"x,5") == {"a": 5} and again.name == "r"


# ============================================================ 格式化
def test_format_text_fields_with_width_decimals_and_missing():
    rule = FormatRule(kind="text", separator=",", prefix="RESULT,", terminator="\\n", missing="-",
                      fields=[FormatField(name="id", source="request_id", dtype="int"),
                              FormatField(name="st", source="status"),
                              FormatField(name="w", source="out.width", dtype="float", decimals=3),
                              FormatField(name="pad", source="literal:A", width=4, align="right", pad="0"),
                              FormatField(name="gone", source="var.nope")])
    out = rule.render(None, {}, {"request_id": 1001, "status": "OK", "out": {}})
    assert out == b"RESULT,1001,OK,-,000A,-\n"


def test_format_text_template_mode_and_json_and_binary():
    tpl = FormatRule(kind="text", template="{status},{request_id},{field.model}\\n")
    assert tpl.render(None, {}, {"status": "NG", "request_id": 3, "fields": {"model": "A1"}}) == b"NG,3,A1\n"
    js = FormatRule(kind="json", terminator="", fields=[
        FormatField(name="req", source="request_id", dtype="int"),
        FormatField(name="ok", source="status")])
    assert json.loads(js.render(None, {}, {"request_id": 9, "status": "OK"})) == {"req": 9, "ok": "OK"}
    bina = FormatRule(kind="binary", prefix="\\x02", terminator="\\x03", fields=[
        FormatField(name="id", source="request_id", dtype="uint16"),
        FormatField(name="x", source="out.x", dtype="float32"),
        FormatField(name="tag", source="literal:AB", dtype="string", width=4)])
    data = bina.render(None, {}, {"request_id": 258})
    assert data == b"\x02" + struct.pack(">H", 258) + struct.pack(">f", 0.0) + b"AB\x00\x00" + b"\x03"


def test_format_resolve_source_is_restricted():
    # 认得的来源
    assert resolve_source("literal:x") == "x"
    assert resolve_source("var.a", None, {"a": 1}) == 1
    assert resolve_source("field.b", None, {}, {"fields": {"b": 2}}) == 2
    # 不认得的来源一律 None，不会去求值任何表达式
    assert resolve_source("__import__('os').system") is None
    assert resolve_source("1+1") is None
    assert resolve_source("os.environ") is None


def test_pack_unpack_scalars_clamp():
    assert unpack_scalar(pack_scalar(-5, "int16"), "int16") == -5
    assert unpack_scalar(pack_scalar(99999, "int16"), "int16") == 32767      # 饱和截断
    assert unpack_scalar(pack_scalar(1.25, "float32"), "float32") == 1.25
    assert pack_scalar(1, "uint16", "little") == b"\x01\x00"


# ============================================================ Modbus 数据点
def test_reference_address_conversion_both_ways():
    assert reference_address("holding", 10) == 40011
    assert reference_address("coil", 0) == 1
    assert reference_address("input", 4) == 30005
    assert reference_address("discrete", 1) == 10002
    assert "协议地址 10" in reference_text("holding", 10)
    assert parse_reference("40011") == ("holding", 10)
    assert parse_reference("30005") == ("input", 4)
    assert parse_reference("10002") == ("discrete", 1)
    assert parse_reference("4x10") == ("holding", 10)
    assert parse_reference("0x3") == ("coil", 3)
    assert parse_reference("holding:10") == ("holding", 10)
    assert parse_reference(10) == ("holding", 10)
    assert parse_reference("10", "coil") == ("coil", 10)
    with pytest.raises(MapError):
        parse_reference("abc")


@pytest.mark.parametrize("layout,expected", [
    ("ABCD", "12 34 56 78"), ("BADC", "34 12 78 56"),
    ("CDAB", "56 78 12 34"), ("DCBA", "78 56 34 12")])
def test_layout_actual_bytes_for_32bit(layout, expected):
    words = encode_value(0x12345678, "int32", layout)
    raw = b"".join(w.to_bytes(2, "big") for w in words)
    assert to_hex(raw) == expected
    assert decode_value(words, "int32", layout) == 0x12345678
    assert expected.replace(" ", " ") in layout_example("int32", layout)


@pytest.mark.parametrize("layout,expected", [("ABCD", "12 34"), ("CDAB", "12 34"),
                                             ("BADC", "34 12"), ("DCBA", "34 12")])
def test_layout_16bit_only_byte_swap_applies(layout, expected):
    """16 位值只受**寄存器内字节交换**影响，字交换对它没有意义。"""
    words = encode_value(0x1234, "uint16", layout)
    assert to_hex(words[0].to_bytes(2, "big")) == expected
    assert decode_value(words, "uint16", layout) == 0x1234


def test_float32_and_string_and_words_for():
    assert decode_value(encode_value(-2.5, "float32", "CDAB"), "float32", "CDAB") == -2.5
    assert words_for("float32") == 2 and words_for("int16") == 1 and words_for("string", length=7) == 4
    assert words_for("bool", count=8) == 8
    s = encode_value("AB", "string", "ABCD", length=4)
    assert decode_value(s, "string", "ABCD", length=4) == "AB"
    swapped = encode_value("ABCD", "string", "BADC", length=4)
    assert b"".join(w.to_bytes(2, "big") for w in swapped) == b"BADC"
    with pytest.raises(MapError):
        words_for("string")


def test_datapoint_validation_rejects_bad_configurations():
    with pytest.raises(MapError):        # 输入寄存器不可写
        DataPoint(name="a", area="input", dtype="int16", direction="write")
    with pytest.raises(MapError):        # 离散输入不可写
        DataPoint(name="a", area="discrete", dtype="bool", direction="read_write")
    with pytest.raises(MapError):        # 位区放不了 int16
        DataPoint(name="a", area="coil", dtype="int16")
    with pytest.raises(MapError):        # 地址越界
        DataPoint(name="a", area="holding", address=65535, dtype="int32")
    with pytest.raises(MapError):        # 名称为空
        DataPoint(name=" ", area="holding")
    with pytest.raises(MapError):        # 字符串必须给长度
        DataPoint(name="a", area="holding", dtype="string")
    with pytest.raises(MapError):
        DataPoint(name="a", area="nowhere")
    ok = DataPoint(name="x", area="holding", address=10, dtype="float32", layout="CDAB")
    assert ok.words == 2 and ok.reads and not ok.writes and "40011" in ok.reference


def test_datapoint_scale_applies_both_ways():
    p = DataPoint(name="w", area="holding", address=0, dtype="int16", scale=0.01, direction="read_write")
    assert p.encode(12.34) == [1234]
    assert abs(p.decode([1234]) - 12.34) < 1e-9


def test_datapoint_table_dedups_and_merges_read_blocks():
    t = DataPointTable()
    t.add(DataPoint(name="a", area="holding", address=0, dtype="int16", poll_ms=20))
    t.add(DataPoint(name="b", area="holding", address=1, dtype="int32", poll_ms=20))
    t.add(DataPoint(name="far", area="holding", address=500, dtype="int16", poll_ms=20))
    t.add(DataPoint(name="c", area="coil", address=2, dtype="bool", poll_ms=20))
    with pytest.raises(MapError):
        t.add(DataPoint(name="a", area="holding", address=9, dtype="int16"))
    blocks = {(b.area, b.address, b.count): [p.name for p, _ in b.points] for b in t.read_blocks()}
    assert blocks[("holding", 0, 3)] == ["a", "b"]       # 相邻的合并成一次请求
    assert blocks[("holding", 500, 1)] == ["far"]        # 离得远的单独一次
    assert blocks[("coil", 2, 1)] == ["c"]
    assert t.names() == ["a", "b", "far", "c"] and t.get("b").dtype == "int32"
    t.remove("far")
    assert t.get("far") is None and len(t) == 3


def test_read_blocks_respect_protocol_limits():
    t = DataPointTable()
    for i in range(0, 300, 2):
        t.add(DataPoint(name=f"p{i}", area="holding", address=i, dtype="int32", poll_ms=10))
    blocks = t.read_blocks()
    assert all(b.count <= AREA_INFO["holding"]["max_read"] for b in blocks)
    assert len(blocks) >= 3


# ============================================================ Modbus RTU（协议层 + 伪终端互通）
def test_rtu_crc_matches_known_values():
    """CRC-16/Modbus 的已知值核对（多项式 0xA001，初值 0xFFFF，低字节先发）。"""
    from cvflow.comm.modbus_client import crc16
    assert crc16(b"") == 0xFFFF
    assert crc16(b"\x01\x03\x00\x00\x00\x01") == 0x0A84      # 手册例：站 1 读保持寄存器 0
    assert crc16(b"\x01\x06\x00\x04\xfb\x2e") == 0x270B      # 站 1 写寄存器 4 = -1234
    # 低字节先发：整条帧的 CRC 自校验（把 CRC 也算进去，结果必为 0）
    frame = b"\x01\x03\x00\x00\x00\x01" + struct.pack("<H", 0x0A84)
    assert crc16(frame) == 0


@pytest.mark.parametrize("fc,head,expected", [
    (3, b"\x01\x03\x04", 9),          # 站号+功能码+字节数+4 数据+CRC2
    (1, b"\x01\x01\x01", 6),
    (6, b"\x01\x06", 8),              # 写单个寄存器的应答是回显
    (16, b"\x01\x10", 8),
    (0x83, b"\x01\x83", 5),           # 异常响应：站号+功能码+异常码+CRC2
    (3, b"\x01\x03", -1),             # 还不够：需要字节数那一字节
    (0x40, b"\x01\x40", -2),          # 不支持的功能码
])
def test_rtu_expected_frame_length(fc, head, expected):
    from cvflow.comm.modbus_client import expected_rtu_length
    assert expected_rtu_length(fc, head) == expected


def test_rtu_pdu_limits_are_enforced():
    from cvflow.comm.base import CommError
    from cvflow.comm.modbus_client import pdu_read, pdu_write_coils, pdu_write_registers
    with pytest.raises(CommError):
        pdu_read(3, 0, 200)                      # 功能码 03 单次最多 125 个
    with pytest.raises(CommError):
        pdu_read(1, 0, 3000)                     # 功能码 01 单次最多 2000 个
    with pytest.raises(CommError):
        pdu_write_registers(0, [0] * 200)        # 功能码 16 单次最多 123 个
    with pytest.raises(CommError):
        pdu_read(3, 65530, 10)                   # 越过 65535
    assert pdu_write_coils(0, [True, False])[0] == 15
    assert pdu_write_registers(0, [1, 2])[0] == 16


class _RtuSlave:
    """伪终端上的最小 RTU 从站：功能码 03/06/16，自己算 CRC，站号不符不应答。

    请求帧是**变长**的（功能码 16 带一段数据），所以按功能码推导整条请求该有多少字节，
    不能按固定长度切——这正是 RTU 没有长度字段带来的实际问题。
    """

    def __init__(self, master_fd: int, registers: int = 16):
        import threading
        self.fd = master_fd
        self.regs = [0] * registers
        self.requests: list[bytes] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _request_length(buf: bytes) -> int:
        """按功能码算出整条请求的字节数；不够判断时返回 -1。"""
        if len(buf) < 2:
            return -1
        fc = buf[1]
        if fc in (1, 2, 3, 4, 5, 6):
            return 8
        if fc in (15, 16):
            if len(buf) < 7:
                return -1
            return 7 + buf[6] + 2
        return 8

    def _run(self) -> None:
        import os
        from cvflow.comm.modbus_client import crc16
        buf = b""
        while not self.stop.is_set():
            try:
                chunk = os.read(self.fd, 256)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while True:
                need = self._request_length(buf)
                if need < 0 or len(buf) < need:
                    break
                frame, buf = buf[:need], buf[need:]
                self.requests.append(frame)
                resp = self._handle(frame)
                if resp:
                    os.write(self.fd, resp + struct.pack("<H", crc16(resp)))

    def _handle(self, frame: bytes) -> bytes:
        from cvflow.comm.modbus_client import crc16
        if crc16(frame[:-2]) != struct.unpack("<H", frame[-2:])[0]:
            return b""                      # CRC 不对就不应答，主站会超时
        unit, fc = frame[0], frame[1]
        if unit != 1:
            return b""                      # 站号不符：按规范不应答
        addr, count = struct.unpack(">HH", frame[2:6])
        if fc == 3:
            if addr + count > len(self.regs):
                return bytes([unit, fc | 0x80, 2])
            body = struct.pack(f">{count}H", *self.regs[addr:addr + count])
            return bytes([unit, 3, count * 2]) + body
        if fc == 6:
            self.regs[addr] = count          # 功能码 06 的第二个字段是值
            return frame[:6]
        if fc == 16:
            nbytes = frame[6]
            values = struct.unpack(f">{count}H", frame[7:7 + nbytes])
            self.regs[addr:addr + count] = list(values)
            return frame[:6]
        return bytes([unit, fc | 0x80, 1])

    def close(self) -> None:
        self.stop.set()
        self.thread.join(2)


def test_rtu_client_roundtrip_over_pty():
    """Linux 伪终端上和一个按手册拼帧的 RTU 从站互通：读、写单个、写多个、CRC、站号校验。"""
    import os
    import platform
    if platform.system() != "Linux":
        pytest.skip("伪终端只在 Linux 上可用")
    pytest.importorskip("serial")
    import pty

    from cvflow.comm.base import CommError
    from cvflow.comm.modbus import ModbusRtuClientDevice
    from cvflow.comm.modbus_client import crc16

    master, slave_fd = pty.openpty()
    port = os.ttyname(slave_fd)
    slave = _RtuSlave(master)
    dev = ModbusRtuClientDevice("rtu", {"port": port, "baudrate": 19200, "unit_id": 1,
                                        "poll_ms": 0, "watch_count": 0, "request_timeout_s": 2.0,
                                        "read_retries": 0, "write_retries": 0})
    assert dev.load_points([
        {"name": "v", "area": "holding", "address": 4, "dtype": "int16", "direction": "read_write"},
        {"name": "f", "area": "holding", "address": 6, "dtype": "float32", "layout": "CDAB",
         "direction": "read_write"},
        {"name": "txt", "area": "holding", "address": 8, "dtype": "string", "length": 6,
         "direction": "read_write"}]) == []
    try:
        # 功能码 06：单个寄存器，有符号数
        dev.write_point("v", -1234)
        assert dev.read_point("v") == -1234
        assert slave.regs[4] == (-1234 & 0xFFFF)
        # 功能码 16：两个寄存器，CDAB 字序
        dev.write_point("f", 1.5)
        assert dev.read_point("f") == 1.5
        assert slave.regs[6] == 0x0000 and slave.regs[7] == 0x3FC0     # 低字在前
        # 字符串
        dev.write_point("txt", "AB")
        assert dev.read_point("txt") == "AB"
        ok, msg = dev.test_connection()
        assert ok and "数据点" in msg
        # 请求帧都带着正确的 CRC（从站校验过才会应答）
        assert slave.requests
        assert all(crc16(f[:-2]) == struct.unpack("<H", f[-2:])[0] for f in slave.requests)
        # 站号不符时从站不应答 → 主站超时报错，而不是把别人的应答当成自己的
        dev.config["unit_id"] = 9      # 配置是站号的唯一来源，改配置下一次请求就生效
        with pytest.raises(CommError):
            dev.read_point("v")
        dev.config["unit_id"] = 1
        assert dev.read_point("v") == -1234          # 改回来立刻恢复
    finally:
        slave.close()
        dev.disconnect()
        os.close(master)
        os.close(slave_fd)

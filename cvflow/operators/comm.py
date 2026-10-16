"""通信节点：在流程里收发报文、解析与格式化、读写数据点、查状态。

这些节点**不自己建连接**：一律通过 ``get_manager()`` 找到全局通信管理器，再按设备 id（或
设备名）取设备。后台监听线程是唯一读套接字的一方，收到的报文复制进各个收件箱，
``Comm Receive`` 从自己的收件箱取——所以节点和监听线程不会抢读同一个连接。

本次触发的请求编号、来源对端和解析出来的字段，在握手接受请求的瞬间就冻结进了触发载荷。
要在流程里用这些参数，用 ``Parse Message``（解析报文）节点，来源选 ``trigger``：
配了解析规则就按规则再解一遍报文，**留空**就直接给出冻结的那一份（含请求编号）。
同一条流程并发排队也不会串——这是它和「读数据点」「读变量」的区别，那两个拿的是此刻的最新值。

采集分类里的「触发」节点（``source.trigger``）只管触发，不带数据：它声明一条路的入口，
由通信面板的触发规则按节点触发（见 ``operators.source``）。
"""
from __future__ import annotations

import json

from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType

_COLOR = "#5d4037"


def _manager():
    from ..comm.manager import get_manager
    mgr = get_manager()
    if mgr is None:
        raise NodeError("通信管理器未启动（请先打开方案并进入运行模式）")
    return mgr


def _value_count(value) -> int:
    """数组数据点读出来是列表，给出里面有几个值；单个值算 1。"""
    return len(value) if isinstance(value, (list, tuple)) else (0 if value is None else 1)


def _device(mgr, ref: str):
    dev = mgr.resolve(ref)
    if dev is None:
        known = "、".join(d.name for d in mgr.devices.values()) or "（没有配置设备）"
        raise NodeError(f"找不到通信设备 {ref!r}。已配置：{known}")
    if not dev.enabled:
        raise NodeError(f"设备「{dev.name}」已禁用")
    return dev


@register
class CommReceive(Node):
    type_id = "comm.receive"
    category = "Communication"
    label = "Comm Receive"
    triggerable = True
    description = ("Take one message from a device's inbox (filled by the background listener). "
                   "Set timeout 0 to use only what already arrived.")
    color = _COLOR
    outputs = [Port("text", DataType.STRING), Port("data", DataType.ANY, label="bytes"),
               Port("peer", DataType.STRING), Port("device", DataType.STRING),
               Port("got", DataType.BOOL)]
    params = [Param("device", "", "choice", options="comm.devices",
                    description="通信设备；留空表示任意设备"),
              Param("timeout_ms", 0, "int", min=0, max=600000,
                    description="等待一条报文的毫秒数；0 表示只取已经到达的"),
              Param("queue_size", 16, "int", min=1, max=1000, description="收件箱容量，满了丢最旧的"),
              Param("on_empty", "skip", "enum", choices=["skip", "error", "empty"],
                    description="没有报文时：跳过下游 / 报错 / 输出空字符串")]

    def _inbox(self):
        mgr = _manager()
        return mgr.inbox(f"node:{self.id}", str(self.get("device") or ""),
                         int(self.get("queue_size")))

    def setup(self, ctx=None) -> None:
        self._inbox()          # 进入运行模式就建好收件箱，别等到第一次触发才开始攒报文

    def teardown(self) -> None:
        from ..comm.manager import get_manager
        mgr = get_manager()
        if mgr is not None:
            mgr.release_inbox(f"node:{self.id}")

    def process(self, ctx, inputs):
        item = self._inbox().get(timeout=int(self.get("timeout_ms")) / 1000.0)
        if item is None:
            mode = self.get("on_empty")
            if mode == "error":
                raise NodeError("收件箱里没有报文")
            if mode == "skip":
                return {"__skip__": "没有收到报文"}
            return {"text": "", "data": b"", "peer": "", "device": "", "got": False}
        return {"text": item["text"], "data": item["data"], "peer": item["peer"],
                "device": item["device"], "got": True}


@register
class CommSend(Node):
    type_id = "comm.send"
    category = "Communication"
    label = "Comm Send"
    runs_last = True          # 汇报结论：等判定与叠加层齐了再执行
    description = "Send text or HEX to a device; can reply to the client that triggered this run."
    color = _COLOR
    inputs = [Port("text", DataType.ANY, optional=True),
              Port("after", DataType.ANY, optional=True, description="连任意输出以强制在它之后执行")]
    outputs = [Port("sent", DataType.BOOL)]
    params = [Param("device", "", "choice", options="comm.devices", description="通信设备"),
              Param("mode", "text", "enum", choices=["text", "hex"],
                    description="text 按文本编码发送；hex 把输入当十六进制字符串"),
              Param("template", "", "string",
                    description="留空用输入端口的值；也可写模板，如 {status},{out.width:.2f}"),
              Param("append", "\\n", "string", description="结束符（可写 \\n、\\r\\n，留空不加）"),
              Param("target", "origin", "enum", choices=["origin", "default", "broadcast"],
                    description="origin 回复本次请求的来源客户端"),
              Param("peer", "", "choice", options="comm.peers", advanced=True,
                    description="指定对端（填了就优先于上面的选择）")]

    def process(self, ctx, inputs):
        from ..comm.framing import parse_hex, unescape
        from ..comm.manager import format_template
        mgr = _manager()
        dev = _device(mgr, self.get("device"))
        payload = ctx.payload or {}
        template = str(self.get("template") or "")
        if template:
            text = format_template(template, None, ctx.variables.snapshot(),
                                   request_id=payload.get("request_id", 0),
                                   fields=payload.get("fields", {}))
        elif inputs.get("text") is None:
            raise NodeError("既没有连接输入，也没有填模板")
        else:
            text = str(inputs["text"])
        if self.get("mode") == "hex":
            try:
                data = parse_hex(text)
            except ValueError as e:
                raise NodeError(str(e)) from None
        else:
            data = text.encode(dev.encoding, errors="replace") + unescape(str(self.get("append") or ""))
        peer = str(self.get("peer") or "")
        if not peer and self.get("target") == "origin":
            peer = str(payload.get("peer") or "")
        if self.get("target") == "broadcast":
            peer = ""
        if peer and peer not in dev.peers:
            raise NodeError(f"来源对端 {peer} 已断开，未发送")
        return {"sent": bool(dev.send(data, peer=peer or None, raw=True))}


@register
class ParseMessage(Node):
    type_id = "comm.parse"
    category = "Communication"
    label = "Parse Message"
    description = ("Split a message into named fields with a parse rule; with source=trigger and no rule "
                   "it hands out this request's frozen parameters (request id, parsed fields).")
    color = _COLOR
    inputs = [Port("text", DataType.ANY, optional=True)]
    outputs = [Port("fields", DataType.DICT), Port("value", DataType.ANY),
               Port("ok", DataType.BOOL), Port("error", DataType.STRING)]
    params = [Param("rule", "", "choice", options="comm.parse_rules",
                    description="解析规则（通信面板里配置）。来源是「触发」、只想取本次请求已经"
                                "解析好的参数时留空即可"),
              Param("field", "", "string", description="要单独输出到 value 端口的字段名"),
              Param("source", "trigger", "enum", choices=["trigger", "input"],
                    description="trigger 取触发本次运行的那条请求（留空规则＝直接用接受请求时"
                                "冻结的参数，含请求编号）；input 解析输入端口的文本"),
              Param("on_error", "error", "enum", choices=["error", "empty"],
                    description="解析失败 / 本次不是通信触发：报错（判为异常）/ 输出空并置 ok=False")]

    def process(self, ctx, inputs):
        """来源是「触发」时优先用**冻结**的那份参数。

        这份参数是握手接受请求的瞬间存进触发载荷的，所以同一条流程并发排队也不会串——
        这正是它和「读数据点」「读变量」的区别：那两个拿到的是**此刻**的最新值，
        请求一排队就可能读到下一件的参数。
        """
        from ..comm.parsing import ParseError
        rule_name = str(self.get("rule") or "")
        name = str(self.get("field") or "")
        if self.get("source") == "trigger":
            payload = ctx.payload or {}
            data = payload.get("raw") or b""
            fields = dict(payload.get("fields") or {})
            if fields and (not rule_name or not data):
                # 触发时已经解析好了（报文按解析规则解的，寄存器触发则是数据点值与请求编号）
                return {"fields": fields, "value": fields.get(name) if name else None,
                        "ok": True, "error": ""}
            if not data:
                if self.get("on_error") == "empty":
                    return {"fields": {}, "value": None, "ok": False, "error": "本次运行不是由通信触发的"}
                raise NodeError("本次运行不是由通信触发的，没有请求参数可取"
                                "（手动运行时把「解析失败」设成 empty 就会跳过而不是报错）")
        else:
            raw = inputs.get("text")
            if raw is None:
                raise NodeError("输入端口没有文本")
            data = raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")
        mgr = _manager()
        rule = mgr.parse_rules.get(rule_name) if rule_name else None
        if rule is None:
            known = "、".join(mgr.parse_rules) or "（没有配置解析规则）"
            if not rule_name:
                raise NodeError(f"没有选择解析规则。本次触发的报文还没有解析过，"
                                f"请在通信面板「解析/格式」页配一条规则再选上。已配置：{known}")
            raise NodeError(f"找不到解析规则 {rule_name!r}。已配置：{known}")
        try:
            fields = rule.parse(bytes(data))
        except ParseError as e:
            if self.get("on_error") == "empty":
                return {"fields": {}, "value": None, "ok": False, "error": str(e)}
            raise NodeError(f"解析失败：{e}") from None
        return {"fields": fields, "value": fields.get(name) if name else None,
                "ok": True, "error": ""}


@register
class FormatResult(Node):
    type_id = "comm.format"
    category = "Communication"
    label = "Format Result"
    runs_last = True          # 汇报结论：等判定与叠加层齐了再执行
    description = "Render the run's results into text / JSON / binary with a format rule."
    color = _COLOR
    inputs = [Port("after", DataType.ANY, optional=True)]
    outputs = [Port("text", DataType.STRING), Port("data", DataType.ANY, label="bytes")]
    params = [Param("rule", "", "choice", options="comm.format_rules", description="格式化规则（通信面板里配置）")]

    def process(self, ctx, inputs):
        mgr = _manager()
        rule = mgr.format_rules.get(str(self.get("rule") or ""))
        if rule is None:
            known = "、".join(mgr.format_rules) or "（没有配置格式化规则）"
            raise NodeError(f"找不到格式化规则 {self.get('rule')!r}。已配置：{known}")
        payload = ctx.payload or {}
        # 流程还没结束，所以这里拿不到 RunResult；用当前已发布的输出现场拼一个取值视图
        view = _ResultView(ctx)
        data = rule.render(view, ctx.variables.snapshot(),
                           {"request_id": payload.get("request_id", 0),
                            "fields": payload.get("fields", {}),
                            "peer": payload.get("peer", ""), "device": payload.get("device", "")})
        from ..comm.framing import decode_text
        return {"text": decode_text(data, rule.encoding), "data": data}


class _ResultView:
    """给 ``Format Result`` 用的"当前运行"视图：只暴露格式化需要的那几个属性。"""

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self.outputs = dict(ctx.outputs)
        self.run_id = ctx.run_id
        self.flow = ctx.graph.name
        self.duration_ms = 0.0
        self.error = ""
        self.ok = True
        self.judgement = ctx.judgement
        self.variables = ctx.variables.snapshot()

    @property
    def passed(self) -> bool:
        return self.judgement is not False

    @property
    def status_text(self) -> str:
        return "NG" if self.judgement is False else "OK"

    def node(self, name: str):
        for r in self._ctx.results.values():
            if r.node_name == name:
                return r
        return None


@register
class ReadDataPoint(Node):
    type_id = "comm.read_point"
    category = "Communication"
    label = "Read Data Point"
    description = "Read a named data point (Modbus / MC / S7) configured on the device."
    color = _COLOR
    inputs = [Port("after", DataType.ANY, optional=True)]
    outputs = [Port("value", DataType.ANY), Port("valid", DataType.BOOL),
               Port("count", DataType.INT, label="几个值"), Port("error", DataType.STRING)]
    params = [Param("device", "", "choice", options="comm.register_devices", description="支持数据点的设备"),
              Param("point", "", "choice", options="comm.points", description="数据点"),
              Param("mode", "cached", "enum", choices=["cached", "fresh"],
                    description="cached 用轮询到的最新值（不额外发请求）；fresh 立刻读一次"),
              Param("on_error", "error", "enum", choices=["error", "invalid"],
                    description="读失败：报错 / 输出 valid=False")]

    def process(self, ctx, inputs):
        mgr = _manager()
        dev = _device(mgr, self.get("device"))
        name = str(self.get("point") or "")
        try:
            if self.get("mode") == "cached":
                pv = dev.point_value(name)        # type: ignore[attr-defined]
                if pv.valid and not pv.stale:
                    return {"value": pv.value, "valid": True, "count": _value_count(pv.value),
                            "error": ""}
                if pv.valid and pv.stale:
                    if self.get("on_error") == "invalid":
                        return {"value": pv.value, "valid": False, "count": _value_count(pv.value),
                                "error": pv.error or "数据已过期"}
                    raise NodeError(f"数据点 {name!r} 的值已过期：{pv.error or '连接断开'}")
            value = mgr.read_point(dev.id, name)
        except NodeError:
            raise
        except Exception as e:
            if self.get("on_error") == "invalid":
                return {"value": None, "valid": False, "count": 0, "error": str(e)}
            raise NodeError(f"读数据点 {name!r} 失败：{e}") from None
        return {"value": value, "valid": True, "count": _value_count(value), "error": ""}


@register
class WriteDataPoint(Node):
    """写一个数据点。单个值和**数组**走的是同一个节点。

    数组：数据点的「数量」填几，就一次写几个值（``float32×40`` = 80 个寄存器，一条
    功能码 16 请求写完）。输入端口接 ``[1.2, 3.4, …]``、``[[x, y], …]``（轨迹节点的点列
    直接接上来就行）或 numpy 数组都可以——嵌套会先按行摊平成一维，再按顺序铺进寄存器。
    """

    type_id = "comm.write_point"
    category = "Communication"
    label = "Write Data Point"
    runs_last = True          # 汇报结论：等判定与叠加层齐了再执行
    description = ("Write a value — or an array of values — into a named data point "
                   "(Modbus / MC / S7).")
    color = _COLOR
    inputs = [Port("value", DataType.ANY)]
    outputs = [Port("ok", DataType.BOOL), Port("count", DataType.INT, label="写了几个"),
               Port("error", DataType.STRING)]
    params = [Param("device", "", "choice", options="comm.register_devices", description="支持数据点的设备"),
              Param("point", "", "choice", options="comm.points", description="数据点"),
              Param("scale", 1.0, "float", description="写之前先乘这个系数（数组里每个值都乘）"),
              Param("on_length", "error", "enum", choices=["error", "fit"],
                    description="数组长度和数据点「数量」不符时：报错 / 自动补齐"
                                "（多的截掉，少的补 0）"),
              Param("on_error", "error", "enum", choices=["error", "false"],
                    description="写失败：报错 / 输出 ok=False")]

    def process(self, ctx, inputs):
        from ..comm.modbus_map import MapError, fit_numbers, flatten_numbers
        mgr = _manager()
        dev = _device(mgr, self.get("device"))
        name = str(self.get("point") or "")
        value = inputs["value"]
        scale = float(self.get("scale") or 1.0)
        point = self._point(dev, name)
        count = max(1, int(point.count)) if point is not None else 1
        array = point is not None and not point.bits and point.dtype != "string" and count > 1
        written = 1
        if array or isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
            # 数组：摊平 → 乘系数 → 按数据点的「数量」对齐
            try:
                vals = flatten_numbers(value)
            except MapError as e:
                raise NodeError(f"写数据点 {name!r}：{e}") from None
            if scale != 1.0:
                vals = [v * scale for v in vals]
            if array and len(vals) != count:
                if self.get("on_length") != "fit":
                    raise NodeError(f"写数据点 {name!r}：这个点要 {count} 个值，输入给了 {len(vals)} 个"
                                    f"（把「数量不符」设成「自动补齐」就多的截掉、少的补 0）")
                vals = fit_numbers(vals, count)
            value, written = vals, len(vals)
        elif scale != 1.0:
            try:
                value = float(value) * scale
            except (TypeError, ValueError):
                raise NodeError(f"{value!r} 不是数字，无法乘系数") from None
        try:
            dev.write_point(name, value)          # type: ignore[attr-defined]
        except Exception as e:
            if self.get("on_error") == "false":
                return {"ok": False, "count": 0, "error": str(e)}
            raise NodeError(f"写数据点 {name!r} 失败：{e}") from None
        return {"ok": True, "count": written, "error": ""}

    @staticmethod
    def _point(dev, name: str):
        """取数据点定义（为了知道数组长度）。不支持数据点表的设备给 None。"""
        getter = getattr(dev, "points", None)
        if getter is None:
            return None
        try:
            return getter.get(name)
        except Exception:
            return None


@register
class CommStatus(Node):
    type_id = "comm.status"
    category = "Communication"
    label = "Comm Status"
    description = "Query a device's (or handshake's) live state: connected, peers, counters."
    color = _COLOR
    outputs = [Port("connected", DataType.BOOL), Port("peers", DataType.INT),
               Port("info", DataType.DICT), Port("text", DataType.STRING)]
    params = [Param("device", "", "choice", options="comm.devices", description="通信设备；留空看整体"),
              Param("handshake", "", "choice", options="comm.handshakes",
                    description="检测握手；填了就一并输出它的状态"),
              Param("require_connected", False, "bool",
                    description="未连接时报错（配合判定节点可以让流程直接判 NG）")]

    def process(self, ctx, inputs):
        mgr = _manager()
        ref = str(self.get("device") or "")
        if ref:
            dev = _device(mgr, ref)
            info = dev.info()
            connected, peers = dev.connected, len(dev.peers)
        else:
            rows = mgr.device_rows()
            info = {"devices": rows, "connected": sum(1 for r in rows if r["connected"]),
                    "total": len(rows)}
            connected = bool(rows) and all(r["connected"] for r in rows if r["enabled"])
            peers = sum(r.get("peers", 0) for r in rows)
        hs_name = str(self.get("handshake") or "")
        if hs_name:
            hs = mgr.handshake_by_name(hs_name)
            if hs is None:
                raise NodeError(f"找不到检测握手 {hs_name!r}")
            info = {**info, "handshake": hs.status_dict()}
        if self.get("require_connected") and not connected:
            raise NodeError(f"设备 {ref or '（全部）'} 未连接")
        return {"connected": connected, "peers": peers, "info": info,
                "text": json.dumps(info, ensure_ascii=False, default=str)}

"""生成三个「完整业务闭环」示例方案。

    python examples/build_handshake_demos.py

| 方案 | 角色 | 对端 | 外部脚本 |
|---|---|---|---|
| ``demo_tcp_handshake.json``  | cvflow 做 TCP 服务端 | 上位机 / 机器人 | ``sim_handshake.py tcp`` |
| ``demo_modbus_master.json``  | cvflow 做 Modbus 主站 | 模拟 PLC（从站） | ``sim_handshake.py slave`` |
| ``demo_modbus_slave.json``   | cvflow 做 Modbus 从站 | 模拟 PLC（主站） | ``sim_handshake.py master`` |

三个方案用的是**同一条视觉流程**：取图 → 灰度 → 阈值 → 开运算 → 数孔 + 两把卡尺测宽与偏转角
+ 圆查找定位，发布 ``holes`` ``x`` ``y`` ``angle`` ``width`` 五个结果。
后两个方案的**寄存器映射完全一样**，区别只在谁是主站——换角色不用改流程，也不用改映射。

详细的报文格式、寄存器映射与联调步骤见 ``docs/comm-examples.md``。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvflow.core import Graph, Solution, paths, registry  # noqa: E402

registry.load_builtins()

OUT_DIR = ROOT / "examples" / "solutions"
paths.set_base_dir(OUT_DIR)
IMAGES = "../images"


def place(node, col: int, row: int = 0):
    node.position = [40 + col * 210, 60 + row * 150]
    return node


# ---------------------------------------------------------------- 视觉流程
def build_main() -> Graph:
    """数孔 + 测宽 + 求偏转角 + 定位，发布五个结果供通信回传。"""
    g = Graph("main")
    g.description = ("取图 → 灰度 → 阈值 → 开运算 → 数孔；两把卡尺量宽度与偏转角；"
                     "圆查找给出定位坐标。发布 holes / x / y / angle / width。")
    src = place(g.add_node(registry.create("source.image_folder", name="取图",
                                           values={"directory": IMAGES, "mode": "next"})), 0)
    gray = place(g.add_node(registry.create("preprocess.color", name="灰度", values={"mode": "gray"})), 1)
    th = place(g.add_node(registry.create("preprocess.threshold", name="阈值",
                                          values={"method": "binary_inv", "thresh": 120})), 2)
    mo = place(g.add_node(registry.create("preprocess.morphology", name="开运算",
                                          values={"op": "open", "ksize": 3})), 3)
    blob = place(g.add_node(registry.create("analysis.blob", name="孔",
                                            values={"min_area": 500, "max_area": 5000})), 4)
    jc = place(g.add_node(registry.create("logic.judge", name="孔数判定",
                                          values={"op": "==", "low": 3, "name": "holes"})), 5)
    # 定位：找第一个孔的圆心当作抓取坐标
    circ = place(g.add_node(registry.create("analysis.circle", name="定位",
                                            values={"min_radius": 15, "max_radius": 40})), 4, 1)
    # 两把卡尺：上下各量一次左边缘，位置差就是偏转角
    cal_top = place(g.add_node(registry.create("analysis.caliper", name="上卡尺", values={
        "roi": {"x": 60, "y": 140, "w": 520, "h": 20}, "direction": "horizontal",
        "select": "first_last", "min_contrast": 15})), 4, 2)
    cal_bot = place(g.add_node(registry.create("analysis.caliper", name="下卡尺", values={
        "roi": {"x": 60, "y": 240, "w": 520, "h": 20}, "direction": "horizontal",
        "select": "first_last", "min_contrast": 15})), 4, 3)
    angle = place(g.add_node(registry.create("logic.expression", name="偏转角", values={
        "expression": "degrees(atan2(b - a, 100.0))"})), 5, 3)
    jw = place(g.add_node(registry.create("logic.judge", name="宽度判定", values={
        "op": "in_range", "low": 380, "high": 420, "name": "width"})), 5, 2)
    pub = place(g.add_node(registry.create("output.publish", name="发布结果", values={
        "name_a": "holes", "name_b": "x", "name_c": "y", "name_d": "angle"})), 6)
    pub2 = place(g.add_node(registry.create("output.publish", name="发布宽度", values={
        "name_a": "width"})), 6, 2)
    rnd = place(g.add_node(registry.create("output.render", name="渲染")), 6, 1)
    sav = place(g.add_node(registry.create("output.save_image", name="NG 存图", values={
        "directory": "../../captures", "when": "ng_only"})), 7, 1)

    g.add_link(src.id, "image", gray.id, "image")
    g.add_link(gray.id, "image", th.id, "image")
    g.add_link(th.id, "image", mo.id, "image")
    g.add_link(mo.id, "image", blob.id, "image")
    g.add_link(blob.id, "count", jc.id, "value")
    g.add_link(gray.id, "image", circ.id, "image")
    g.add_link(gray.id, "image", cal_top.id, "image")
    g.add_link(gray.id, "image", cal_bot.id, "image")
    g.add_link(cal_top.id, "first", angle.id, "a")
    g.add_link(cal_bot.id, "first", angle.id, "b")
    g.add_link(cal_top.id, "width", jw.id, "value")
    g.add_link(blob.id, "count", pub.id, "a")
    g.add_link(circ.id, "cx", pub.id, "b")
    g.add_link(circ.id, "cy", pub.id, "c")
    g.add_link(angle.id, "value", pub.id, "d")
    g.add_link(cal_top.id, "width", pub2.id, "a")
    g.add_link(src.id, "image", rnd.id, "image")
    g.add_link(jc.id, "ok", rnd.id, "after")
    g.add_link(rnd.id, "image", sav.id, "image")
    g.add_link(jw.id, "ok", sav.id, "after")
    return g


# ---------------------------------------------------------------- 通信配置
#: 示例 A：TCP 文本协议的解析与格式化规则。
TCP_PARSE = [{
    "name": "request", "kind": "delimited", "separator": ",", "encoding": "utf-8",
    "strip": True, "to_variables": False,
    "fields": [{"name": "cmd", "index": 0, "dtype": "string"},
               {"name": "request_id", "index": 1, "dtype": "int"}],
}]

TCP_FORMAT = [
    {"name": "result", "kind": "text", "separator": ",", "terminator": "\\n", "missing": "",
     "fields": [{"name": "head", "source": "literal:RESULT"},
                {"name": "request_id", "source": "request_id", "dtype": "int"},
                {"name": "status", "source": "status"},
                {"name": "x", "source": "out.x", "dtype": "float", "decimals": 3},
                {"name": "y", "source": "out.y", "dtype": "float", "decimals": 3},
                {"name": "angle", "source": "out.angle", "dtype": "float", "decimals": 3}]},
    {"name": "busy", "kind": "text", "separator": ",", "terminator": "\\n",
     "fields": [{"name": "head", "source": "literal:BUSY"},
                {"name": "request_id", "source": "request_id", "dtype": "int"}]},
    {"name": "error", "kind": "text", "separator": ",", "terminator": "\\n",
     "fields": [{"name": "head", "source": "literal:ERROR"},
                {"name": "request_id", "source": "request_id", "dtype": "int"},
                {"name": "code", "source": "error_code", "dtype": "int"}]},
    {"name": "result_json", "kind": "json", "terminator": "\\n", "json_indent": 0,
     "fields": [{"name": "request_id", "source": "request_id", "dtype": "int"},
                {"name": "status", "source": "status"},
                {"name": "holes", "source": "out.holes", "dtype": "int"},
                {"name": "x", "source": "out.x", "dtype": "float", "decimals": 3},
                {"name": "y", "source": "out.y", "dtype": "float", "decimals": 3},
                {"name": "angle", "source": "out.angle", "dtype": "float", "decimals": 3},
                {"name": "width", "source": "out.width", "dtype": "float", "decimals": 2}]},
]

#: 示例 B / C 共用的寄存器映射。协议地址从 0 开始；参考地址见 docs/comm-examples.md。
def register_map(direction_rw: str = "read_write") -> list[dict]:
    """``direction_rw`` 是对端写、我们读的那几个点的方向。"""
    poll = 20
    return [
        {"name": "trigger", "area": "holding", "address": 0, "dtype": "int16",
         "direction": direction_rw, "poll_ms": poll, "description": "对端写 1 触发一次检测"},
        {"name": "request_id", "area": "holding", "address": 1, "dtype": "int32",
         "layout": "CDAB", "direction": direction_rw, "poll_ms": poll,
         "description": "本次请求编号，由对端写入"},
        {"name": "ready", "area": "holding", "address": 3, "dtype": "int16", "direction": "write",
         "description": "1=可以接受新任务"},
        {"name": "busy", "area": "holding", "address": 4, "dtype": "int16", "direction": "write",
         "description": "1=正在执行"},
        {"name": "done", "area": "holding", "address": 5, "dtype": "int16", "direction": "write",
         "description": "1=结果已写完，可以读了"},
        {"name": "result", "area": "holding", "address": 6, "dtype": "int16", "direction": "write",
         "description": "1=OK，2=NG"},
        {"name": "error_code", "area": "holding", "address": 7, "dtype": "int16", "direction": "write",
         "description": "0 正常；1 忙 2 参数错 3 流程异常 4 超时 5 未运行 6 重复"},
        {"name": "done_id", "area": "holding", "address": 8, "dtype": "int32", "layout": "CDAB",
         "direction": "write", "description": "本次完成对应的请求编号"},
        {"name": "ack", "area": "holding", "address": 10, "dtype": "int16",
         "direction": direction_rw, "poll_ms": poll, "description": "对端写 1 表示已取走结果"},
        {"name": "holes", "area": "holding", "address": 20, "dtype": "int16", "direction": "write",
         "description": "孔数"},
        {"name": "x", "area": "holding", "address": 21, "dtype": "float32", "layout": "CDAB",
         "direction": "write", "description": "定位 X（像素）"},
        {"name": "y", "area": "holding", "address": 23, "dtype": "float32", "layout": "CDAB",
         "direction": "write", "description": "定位 Y（像素）"},
        {"name": "angle", "area": "holding", "address": 25, "dtype": "float32", "layout": "CDAB",
         "direction": "write", "description": "偏转角（度）"},
        {"name": "width", "area": "holding", "address": 27, "dtype": "float32", "layout": "CDAB",
         "direction": "write", "description": "宽度（像素）"},
        {"name": "heartbeat", "area": "holding", "address": 30, "dtype": "int16",
         "direction": "write", "description": "视觉在线心跳，每秒自增"},
    ]


RESULT_POINTS = [{"point": "holes", "source": "out.holes"},
                 {"point": "x", "source": "out.x"},
                 {"point": "y", "source": "out.y"},
                 {"point": "angle", "source": "out.angle"},
                 {"point": "width", "source": "out.width"}]

REGISTER_HANDSHAKE = {
    "name": "检测握手", "flow": "main", "mode": "register", "enabled": True,
    "ready": "ready", "busy": "busy", "done": "done", "result": "result",
    "error_code": "error_code", "request_id": "request_id", "done_id": "done_id", "ack": "ack",
    "ack_value": 1, "ok_value": 1, "ng_value": 2,
    "require_ack": True, "auto_reset_s": 10.0, "run_timeout_s": 5.0,
    "busy_policy": "reject", "max_queue": 1, "dedup_window_s": 3.0, "dedup_action": "ignore",
}

VARIANTS = {
    "demo_tcp_handshake": {
        "devices": [{
            "name": "上位机", "kind": "tcp_server",
            "config": {"host": "0.0.0.0", "port": 6000, "max_clients": 8,
                       "default_target": "broadcast", "framing": "delimiter", "terminator": "\\n",
                       "encoding": "utf-8", "max_frame": 4096, "frame_timeout_s": 2.0,
                       "heartbeat_s": 0.0, "heartbeat_text": "HB\\n",
                       "auto_connect": True, "connect_timeout_s": 3.0, "request_timeout_s": 1.0},
        }],
        "parse_rules": TCP_PARSE,
        "format_rules": TCP_FORMAT,
        "receive_rules": [{
            "name": "触发检测", "device": "上位机", "source": "message", "match": "startswith",
            "pattern": "TRIGGER", "action": "trigger_flow", "flow": "main", "parse": "request",
            "request_field": "request_id", "busy_policy": "reject", "max_queue": 1, "enabled": True,
        }],
        "send_rules": [],         # 结果由检测握手按请求来源回复
        "handshakes": [{
            "name": "检测握手", "device": "上位机", "flow": "main", "mode": "text", "enabled": True,
            "result_format": "result", "busy_format": "busy", "error_format": "error",
            "busy_policy": "reject", "run_timeout_s": 5.0,
            "dedup_window_s": 3.0, "dedup_action": "resend",
        }],
    },
    "demo_modbus_master": {
        "devices": [{
            "name": "PLC", "kind": "modbus_tcp_client",
            "config": {"host": "127.0.0.1", "port": 5020, "unit_id": 1, "poll_ms": 0,
                       "watch_address": 0, "watch_count": 0, "read_retries": 1, "write_retries": 0,
                       "busy_address": -1, "trigger_reset": False, "heartbeat_address": 30,
                       "auto_connect": True, "connect_timeout_s": 3.0, "request_timeout_s": 1.0,
                       "auto_reconnect": True, "reconnect_s": 2.0, "heartbeat_s": 1.0},
            "points": register_map("read_write"),
        }],
        "receive_rules": [{
            "name": "触发检测", "device": "PLC", "source": "datapoint", "match": "rising",
            "point": "trigger", "action": "trigger_flow", "flow": "main",
            "request_field": "request_id", "busy_policy": "reject", "max_queue": 1, "enabled": True,
        }],
        "send_rules": [{"name": "回传结果", "device": "PLC", "flow": "main", "when": "always",
                        "template": "", "points": RESULT_POINTS, "enabled": True}],
        "handshakes": [dict(REGISTER_HANDSHAKE, device="PLC")],
    },
    "demo_modbus_slave": {
        "devices": [{
            "name": "PLC", "kind": "modbus_tcp_server",
            "config": {"host": "0.0.0.0", "port": 5020, "unit_id": 1, "register_count": 64,
                       "input_count": 16, "coil_count": 16, "discrete_count": 16,
                       "allow_remote_write": True, "max_clients": 4,
                       "busy_address": -1, "trigger_reset": False, "heartbeat_address": 30,
                       "auto_connect": True, "heartbeat_s": 1.0},
            "points": register_map("read_write"),
        }],
        "receive_rules": [{
            "name": "触发检测", "device": "PLC", "source": "datapoint", "match": "rising",
            "point": "trigger", "action": "trigger_flow", "flow": "main",
            "request_field": "request_id", "busy_policy": "reject", "max_queue": 1, "enabled": True,
        }],
        "send_rules": [{"name": "回传结果", "device": "PLC", "flow": "main", "when": "always",
                        "template": "", "points": RESULT_POINTS, "enabled": True}],
        "handshakes": [dict(REGISTER_HANDSHAKE, device="PLC")],
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, comm in VARIANTS.items():
        sol = Solution(name)
        sol.add_flow(build_main())
        sol.variables.define("product", "PART-A", "string", "当前产品型号")
        sol.variables.define("last_request", 0, "int", "最近一次请求编号")
        sol.comm_config = comm
        out = OUT_DIR / f"{name}.json"
        sol.save(out)
        print("saved", out)


if __name__ == "__main__":
    main()

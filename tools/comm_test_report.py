"""运行通信黑盒测试并生成按功能区归类的报告（Markdown）。

    python tools/comm_test_report.py            # 写入 docs/comm-test-report.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = ["tests/test_comm_blackbox.py", "tests/test_comm_framing.py",
              "tests/test_comm_handshake.py", "tests/test_comm_nodes.py",
              "tests/test_comm_examples.py", "tests/test_comm.py", "tests/test_comm_plc.py"]

AREAS = OrderedDict([
    ("framing", "报文分帧（半包、粘包、多帧、畸形、超限、超时）"),
    ("parse", "报文解析（分隔符、正则、JSON、定长切片）"),
    ("format", "结果格式化（文本、JSON、二进制、取值来源受限）"),
    ("map", "Modbus 数据点映射（地址换算、数据类型、字节序与字序）"),
    ("tcpserver", "TCP 服务端（对端：真实套接字客户端）"),
    ("tcpclient", "TCP 客户端（对端：真实套接字服务端）"),
    ("udp", "UDP"), ("serial", "串口（Linux 伪终端回环）"),
    ("modbusserver", "Modbus TCP 从站（对端：第三方 pymodbus 主站）"),
    ("modbusclient", "Modbus TCP 主站（对端：第三方 pymodbus 从站）"),
    ("mc", "三菱 MC 协议（字节级核对 + 模拟 PLC）"),
    ("s7", "西门子 S7（对端：第三方 snap7 服务器）"),
    ("rules", "触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件"),
    ("template", "发送模板字段、格式与寄存器类型"),
    ("handshake", "检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离）"),
    ("heartbeat", "心跳（寄存器自增、文本报文、断开后停止）"),
    ("nodes", "通信节点（收发、解析、格式化、数据点读写、触发数据、状态）"),
    ("devices", "统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放）"),
    ("events", "事件总线（连接/收/发/断开的事件与载荷）"),
    ("persistence", "配置持久化与非法配置"),
    ("testconn", "连接测试（各设备类型的正反用例）"),
    ("examples", "示例方案端到端（TCP 握手、Modbus 主站、Modbus 从站）"),
    ("cli", "无界面生产模式端到端（cvflow serve）"),
])

#: 测试函数名 → 功能区。按顺序匹配，第一条命中的为准。
NAME_RULES = [
    (r"^test_comm_(?P<area>[a-z0-9]+)_", None),          # 老命名：test_comm_<功能区>_<场景>
    (r"^test_framing_|^test_encoding_helpers", "framing"),
    (r"^test_parse_", "parse"),
    (r"^test_format_|^test_pack_unpack", "format"),
    (r"^test_reference_address|^test_layout_|^test_float32_and_string|^test_datapoint_|^test_read_blocks_", "map"),
    (r"^test_text_handshake_|^test_register_handshake_|^test_register_trigger_|^test_master_handshake_", "handshake"),
    (r"^test_node_", "nodes"),
    (r"^test_device_|^test_rename_|^test_manual_disconnect|^test_references_|^test_validate_"
     r"|^test_bad_point_config|^test_shutdown_releases|^test_comm_log_", "devices"),
    (r"^test_persistence_", "persistence"),
    (r"^test_example_.*serve", "cli"),
    (r"^test_example_", "examples"),
    (r"^test_modbus_handshake|^test_mc_|^test_s7_", "handshake"),
    (r"^test_connection_tests", "testconn"),
    # tests/test_comm.py 里的早期用例，名字不带 comm 前缀
    (r"^test_tcp_server_", "tcpserver"),
    (r"^test_tcp_client_", "tcpclient"),
    (r"^test_udp_", "udp"),
    (r"^test_register_conversions", "map"),
    (r"^test_modbus_server_", "modbusserver"),
    (r"^test_modbus_client_", "modbusclient"),
    (r"^test_solution_comm_config", "persistence"),
    (r"^test_modbus_and_tcp_heartbeats", "heartbeat"),
]


def classify(name: str) -> tuple[str, str]:
    """返回 (功能区, 场景描述)。"""
    for pattern, area in NAME_RULES:
        m = re.match(pattern, name)
        if not m:
            continue
        if area is None:
            found = m.groupdict().get("area")
            if found in AREAS:
                return found, name[m.end():].replace("_", " ") or name
            continue
        scenario = re.sub(r"^test_", "", name).replace("_", " ")
        return area, scenario
    return "rules", re.sub(r"^test_", "", name).replace("_", " ")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(ROOT / "docs" / "comm-test-report.md"))
    a = ap.parse_args()
    xml = Path(tempfile.gettempdir()) / "cvflow_comm.xml"
    subprocess.call([sys.executable, "-m", "pytest", *TEST_FILES, "-q", "-p", "no:warnings",
                     f"--junitxml={xml}"], cwd=ROOT)
    rows: dict[str, list] = {k: [] for k in AREAS}
    total = passed = failed = skipped = 0
    for tc in ET.parse(xml).getroot().iter("testcase"):
        name = tc.get("name", "")
        status, msg = "通过", ""
        for child in tc:
            if child.tag in ("failure", "error"):
                status, msg = "失败", ((child.get("message") or "").strip().splitlines() or [""])[0][:120]
            elif child.tag == "skipped":
                status, msg = "跳过", (child.get("message") or "")[:120]
        total += 1; passed += status == "通过"; failed += status == "失败"; skipped += status == "跳过"
        area, scenario = classify(name)
        rows[area].append((scenario, status, msg, float(tc.get("time", 0))))
    try:
        import pymodbus, snap7  # noqa: E401
        env_text = f"Python {platform.python_version()}, pymodbus {pymodbus.__version__}, python-snap7 {snap7.__version__}, {platform.system()} {platform.release()}"
    except Exception:
        env_text = platform.platform()
    lines = ["# 通信功能黑盒测试报告", "", f"生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}　环境：{env_text}", "",
             "测试方式：把通信层当作整体，只从对端观察。对端使用真实 TCP/UDP 套接字、Linux 伪终端串口、第三方的 pymodbus 主站/从站与 snap7 服务器，"
             "以及按协议手册手工拼出的报文（三菱 MC 3E 帧、Modbus 异常响应）做字节级核对；观察点是事件总线上的事件、被触发的流程与统计、"
             "写回到对端的数据、持久化文件，以及无界面生产模式的子进程。分帧与字节序这类纯逻辑按**实际字节**核对，"
             "不只看往返一致（往返一致掩盖得了字节序写错）。", "",
             f"**合计 {total} 个用例：通过 {passed}，失败 {failed}，跳过 {skipped}。**", "",
             "| 功能区 | 场景 | 结果 | 耗时(s) | 说明 |", "|---|---|---|---|---|"]
    for key, label in AREAS.items():
        for scenario, status, msg, t in rows[key]:
            icon = {"通过": "✅", "失败": "❌", "跳过": "⚠️"}[status]
            lines.append(f"| {label} | {scenario} | {icon} {status} | {t:.1f} | {msg} |")
        if not rows[key]:
            lines.append(f"| {label} | — | ⚠️ 无用例 | | |")
    lines += ["", "## 说明", "",
              "- 串口用例依赖 Linux 伪终端，在 Windows 上跳过；西门子用例依赖 python-snap7，缺少时跳过。",
              "- 三菱 MC 的字节级核对用原始套接字扮演 PLC，直接比对客户端发出的请求帧与手册示例，不依赖项目自带的模拟器。",
              "- Modbus 从站对单元号不符的请求按规范不应答，主站表现为超时。",
              "- 示例方案用例直接加载 `examples/solutions` 里的文件，所以它们同时验证了示例本身可用。",
              "- Modbus RTU 主站在 Linux 伪终端上与一个手工拼帧的从站做了互通验证（读、写单个、写多个、字符串、"
              "CRC、站号不符不应答）；**真串口与真 PLC 还没有验证过**，见 README 的“目前的边界”。",
              "", "重新生成：`python tools/comm_test_report.py`"]
    Path(a.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[6]); print("报告已写入", a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

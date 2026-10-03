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
TEST_FILE = "tests/test_comm_blackbox.py"
AREAS = OrderedDict([
    ("tcpserver", "TCP 服务端（对端：真实套接字客户端）"), ("tcpclient", "TCP 客户端（对端：真实套接字服务端）"),
    ("udp", "UDP"), ("serial", "串口（Linux 伪终端回环）"),
    ("modbusserver", "Modbus TCP 从站（对端：第三方 pymodbus 主站）"), ("modbusclient", "Modbus TCP 主站（对端：第三方 pymodbus 从站）"),
    ("mc", "三菱 MC 协议（字节级核对 + 模拟 PLC）"), ("s7", "西门子 S7（对端：第三方 snap7 服务器）"),
    ("rules", "接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件"), ("template", "发送模板字段、格式与寄存器类型"),
    ("handshake", "PLC 握手（忙标志、触发复位）"), ("heartbeat", "心跳（寄存器自增、文本报文、断开后停止）"),
    ("events", "事件总线（连接/收/发/断开的事件与载荷）"), ("persistence", "配置持久化与非法配置"),
    ("testconn", "连接测试（各设备类型的正反用例）"), ("cli", "无界面生产模式端到端（cvflow serve）"),
])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(ROOT / "docs" / "comm-test-report.md"))
    a = ap.parse_args()
    xml = Path(tempfile.gettempdir()) / "cvflow_comm.xml"
    subprocess.call([sys.executable, "-m", "pytest", TEST_FILE, "-q", "-p", "no:warnings", f"--junitxml={xml}"], cwd=ROOT)
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
        m = re.match(r"test_comm_([a-z0-9]+)_(.*)", name)
        area = m.group(1) if m and m.group(1) in AREAS else "events"
        scenario = (m.group(2) if m else name).replace("_", " ")
        rows[area].append((scenario, status, msg, float(tc.get("time", 0))))
    try:
        import pymodbus, snap7  # noqa: E401
        env_text = f"Python {platform.python_version()}, pymodbus {pymodbus.__version__}, python-snap7 {snap7.__version__}, {platform.system()} {platform.release()}"
    except Exception:
        env_text = platform.platform()
    lines = ["# 通信功能黑盒测试报告", "", f"生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}　环境：{env_text}", "",
             "测试方式：把通信层当作整体，只从对端观察。对端使用真实 TCP/UDP 套接字、Linux 伪终端串口、第三方的 pymodbus 主站/从站与 snap7 服务器，"
             "以及按协议手册手工拼出的报文（三菱 MC 3E 帧、Modbus 异常响应）做字节级核对；观察点是事件总线上的事件、被触发的流程与统计、"
             "写回到对端的数据、持久化文件，以及无界面生产模式的子进程。", "",
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
              "", "重新生成：`python tools/comm_test_report.py`"]
    Path(a.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[6]); print("报告已写入", a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

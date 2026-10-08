"""运行界面黑盒测试并生成按功能区归类的报告（Markdown）。

    python tools/ui_test_report.py            # 写入 docs/ui-test-report.md
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
TEST_FILE = "tests/test_ui_blackbox.py"
AREAS = OrderedDict([
    ("startup", "启动与布局"), ("file", "文件菜单（新建/打开/保存/另存/最近/快捷方式/关闭）"),
    ("flow", "流程管理（新建/重命名/删除/切换/检查）"), ("palette", "节点库（筛选/双击/拖放）"),
    ("editor", "节点编辑器（选择/移动/删除/连线/右键菜单/缩放）"), ("params", "参数面板（各类控件/高级参数/代码/ROI）"),
    ("run", "运行控制（运行一次/自动运行/连续/运行模式）"), ("image", "图像窗口（显示/叠加层/缩放/像素信息）"),
    ("results", "结果面板"), ("variables", "变量面板"), ("log", "日志面板"),
    ("comm", "通信面板（设备 / 数据点 / 触发 / 发送 / 解析·格式 / 握手 / 调试）"), ("camera", "相机管理对话框"), ("plugins", "插件菜单"), ("help", "帮助"),
])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(ROOT / "docs" / "ui-test-report.md"))
    a = ap.parse_args()
    xml = Path(tempfile.gettempdir()) / "cvflow_ui.xml"
    env = dict(**__import__("os").environ, QT_QPA_PLATFORM="offscreen")
    subprocess.call([sys.executable, "-m", "pytest", TEST_FILE, "-q", "-p", "no:warnings", f"--junitxml={xml}"], cwd=ROOT, env=env)
    rows: dict[str, list] = {k: [] for k in AREAS}
    total = passed = failed = skipped = 0
    for tc in ET.parse(xml).getroot().iter("testcase"):
        name = tc.get("name", "")
        status, msg = "通过", ""
        for child in tc:
            if child.tag in ("failure", "error"):
                status, msg = "失败", (child.get("message") or "").strip().splitlines()[0][:120] if (child.get("message") or "").strip() else ""
            elif child.tag == "skipped":
                status, msg = "跳过", (child.get("message") or "")[:120]
        total += 1; passed += status == "通过"; failed += status == "失败"; skipped += status == "跳过"
        m = re.match(r"test_ui_([a-z]+)_(.*)", name)
        area = m.group(1) if m and m.group(1) in AREAS else "startup"
        scenario = (m.group(2) if m else name).replace("_", " ")
        rows[area].append((scenario, status, msg, float(tc.get("time", 0))))
    try:
        import PySide6
        env_text = f"Python {platform.python_version()}, PySide6 {PySide6.__version__}, {platform.system()} {platform.release()}, QT_QPA_PLATFORM=offscreen"
    except Exception:
        env_text = platform.platform()
    lines = ["# 界面功能黑盒测试报告", "", f"生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}　环境：{env_text}", "",
             "测试方式：只通过用户可见的操作驱动界面（菜单、工具栏、鼠标按下/拖动/释放、键盘、对话框、表格编辑），检查用户可见的结果"
             "（画布上的节点与连线、面板内容、标题栏、写出的文件、通信设备状态）。模态对话框在离屏环境里会阻塞，用桩替换并记录调用；"
             "设置存储隔离到临时目录。", "",
             f"**合计 {total} 个用例：通过 {passed}，失败 {failed}，跳过 {skipped}。**", "",
             "| 功能区 | 场景 | 结果 | 耗时(s) | 说明 |", "|---|---|---|---|---|"]
    for key, label in AREAS.items():
        for scenario, status, msg, t in rows[key]:
            icon = {"通过": "✅", "失败": "❌", "跳过": "⚠️"}[status]
            lines.append(f"| {label} | {scenario} | {icon} {status} | {t:.1f} | {msg} |")
        if not rows[key]:
            lines.append(f"| {label} | — | ⚠️ 无用例 | | |")
    lines += ["", "## 说明", "",
              "- 每个用例都会新建主窗口并打开示例方案，用例之间互不影响。",
              "- 节点编辑器的连线、移动、右键菜单通过向视图发送真实的鼠标事件完成；拖放从节点库到画布在离屏平台没有拖放会话，改为把放下事件直接交给视图。",
              "- 运行模式用例会真正启动流程线程并连接示例方案的 TCP 服务端（6000 端口）。",
              "", "重新生成：`python tools/ui_test_report.py`"]
    Path(a.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[6]); print("报告已写入", a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

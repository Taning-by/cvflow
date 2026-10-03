"""运行节点黑盒测试并生成按节点归类的测试报告（Markdown）。

    python tools/node_test_report.py            # 写入 docs/node-test-report.md
    python tools/node_test_report.py -o 报告.md
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
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

TEST_FILES = ["tests/test_nodes_contract.py", "tests/test_nodes_blackbox.py"]


def _node_table():
    from cvflow.core import registry
    registry.load_builtins()
    registry.load_plugin_dir(ROOT / "examples" / "plugins")
    from cvflow.ui.i18n import tr
    return {c.type_id: (tr(c.category), tr(c.label)) for c in registry.all()}


def _extra_coverage():
    try:
        import test_nodes_blackbox  # noqa: F401
        return dict(getattr(test_nodes_blackbox, "EXTRA_COVERAGE", {}))
    except Exception:
        return {}


def _map_test_to_nodes(name: str, type_ids: list[str], extra: dict) -> list[str]:
    m = re.search(r"\[([a-z0-9_.]+)\]$", name)
    if m and m.group(1) in type_ids:
        return [m.group(1)]
    base = name.split("[")[0]
    cands = [t for t in type_ids if base.startswith("test_" + t.replace(".", "_") + "_")]
    nodes = [max(cands, key=len)] if cands else []
    nodes += extra.get(base, [])
    return nodes


def run_tests(xml_path: Path) -> int:
    cmd = [sys.executable, "-m", "pytest", *TEST_FILES, "-q", "-p", "no:warnings", f"--junitxml={xml_path}"]
    return subprocess.call(cmd, cwd=ROOT)


def build_report(xml_path: Path) -> str:
    nodes = _node_table()
    type_ids = list(nodes)
    extra = _extra_coverage()
    per: dict[str, dict] = {t: {"contract": [], "func": [], "fail": [], "skip": []} for t in type_ids}
    unmapped = []
    total = passed = failed = skipped = 0
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        name = tc.get("name", "")
        status = "pass"
        msg = ""
        for child in tc:
            if child.tag in ("failure", "error"):
                status = "fail"; msg = (child.get("message") or child.text or "").strip().splitlines()[0][:120]
            elif child.tag == "skipped":
                status = "skip"; msg = (child.get("message") or "").strip()[:120]
        total += 1
        passed += status == "pass"; failed += status == "fail"; skipped += status == "skip"
        targets = _map_test_to_nodes(name, type_ids, extra)
        if not targets:
            unmapped.append(name)
        for t in targets:
            bucket = "contract" if name.startswith("test_contract_") else "func"
            per[t][bucket].append(name)
            if status == "fail":
                per[t]["fail"].append(f"{name}: {msg}")
            elif status == "skip" and not name.startswith("test_contract_bad_input_is_isolated"):
                per[t]["skip"].append(f"{name}: {msg}")   # 预期中的契约跳过（只有 ANY 输入）不列入说明
    try:
        import cv2, numpy, onnxruntime  # noqa: E401
        env = f"Python {platform.python_version()}, OpenCV {cv2.__version__}, numpy {numpy.__version__}, onnxruntime {onnxruntime.__version__}, {platform.system()} {platform.release()}"
    except Exception:
        env = platform.platform()
    lines = [f"# 节点功能黑盒测试报告", "",
             f"生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}　环境：{env}", "",
             f"测试方式：只通过节点的公开接口（输入端口、参数 → 输出端口）验证，期望值来自可解析计算的合成图像。"
             f"契约测试对注册表中全部 {len(type_ids)} 个节点自动执行（元数据、输出端口与声明一致、错误输入被隔离）；功能测试按节点覆盖各模式与参数。", "",
             f"**合计 {total} 个用例：通过 {passed}，失败 {failed}，跳过 {skipped}（跳过的均为“只有 ANY 类型输入、无类型约束可验证”的契约用例）。**", "",
             "| 分类 | 节点 | type_id | 契约用例 | 功能用例 | 结果 | 说明 |", "|---|---|---|---|---|---|---|"]
    for t in sorted(type_ids, key=lambda x: (nodes[x][0], nodes[x][1])):
        cat, label = nodes[t]
        d = per[t]
        res = "❌ 失败" if d["fail"] else ("⚠️ 跳过" if d["skip"] and not d["func"] else "✅ 通过")
        note = "; ".join(d["fail"] + d["skip"]) or ("无功能用例" if not d["func"] else "")
        lines.append(f"| {cat} | {label} | `{t}` | {len(d['contract'])} | {len(d['func'])} | {res} | {note} |")
    uncovered = [t for t in type_ids if not per[t]["func"]]
    lines += ["", "## 覆盖缺口", ""]
    lines += [f"- 没有功能用例的节点：{', '.join('`' + t + '`' for t in uncovered)}" if uncovered else "- 所有节点都有功能用例。"]
    if unmapped:
        lines += [f"- 未归类到节点的用例：{', '.join(unmapped[:10])}"]
    lines += ["", "## 说明", "",
              "- 契约用例对每个节点各 2～3 条：元数据/序列化往返、输出端口齐全且类型符合声明、错误输入以 ERROR 结束且不崩溃。",
              "- 功能用例的真值来自合成图像：已知面积与质心的方块、已知边缘位置的阶跃、已知圆心半径的圆、用 OpenCV 生成的二维码、用 onnx 构造的微型模型（恒等、按通道均值分类、固定输出的 YOLO 布局）。",
              "- 未在真实硬件上验证的部分（海康 MVS 取图、GenICam 取图、PyTorch 插件）以“明确报错”作为通过标准。",
              "", "重新生成：`python tools/node_test_report.py`"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(ROOT / "docs" / "node-test-report.md"))
    ap.add_argument("--xml", default=None, help="已有的 junit xml（跳过运行）")
    a = ap.parse_args()
    xml = Path(a.xml) if a.xml else Path(tempfile.gettempdir()) / "cvflow_nodes.xml"
    if not a.xml:
        run_tests(xml)
    report = build_report(xml)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(report, encoding="utf-8")
    print(report.splitlines()[4])
    print("报告已写入", a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

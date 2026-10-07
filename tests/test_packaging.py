r"""打包配置的静态检查。

Windows 安装包只能在 Windows 上构建，本机验不了；但配置文件里的低级错误完全可以在这里
几秒钟挡掉，不必等 CI 上跑十几分钟的 Windows 作业才发现。实际踩过的两个坑都在这里：

* installer.iss 里一行被意外断成两行（生成文件时反斜杠转义错了），Inno 报
  `Unrecognized parameter name "vidia\*..."`；
* spec 里的 hidden import 写错模块名，PyInstaller 只打一行 ERROR 就继续，
  打出来的程序悄悄少掉节点。
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "packaging"

#: Inno Setup 各节里合法的首个参数名
SECTION_KEYS = {
    "Files": {"Source"}, "Icons": {"Name"}, "Dirs": {"Name"}, "Registry": {"Root"},
    "Run": {"Filename"}, "Components": {"Name"}, "Tasks": {"Name"}, "Types": {"Name"},
    "Languages": {"Name"}, "UninstallDelete": {"Type"},
}


def logical_lines(text: str):
    """把 Inno 脚本的续行（行尾反斜杠）合并，返回 (节名, 整行) 列表。"""
    section = ""
    out, buf = [], ""
    for raw in text.split("\n"):
        line = raw.rstrip()
        if buf:
            buf = buf[:-1].rstrip() + " " + line.strip()
        elif line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        elif not line.strip() or line.lstrip().startswith(";") or line.lstrip().startswith("#"):
            continue
        else:
            buf = line
        if buf.endswith("\\"):
            continue
        out.append((section, buf))
        buf = ""
    if buf:
        out.append((section, buf))
    return out


def test_installer_script_entries_start_with_a_known_parameter():
    """每条目录项都要以合法的参数名开头——被转义毁掉的半截行就是这么露出来的。"""
    text = (PKG / "installer.iss").read_text(encoding="utf-8")
    bad = []
    for section, line in logical_lines(text):
        keys = SECTION_KEYS.get(section)
        if not keys:
            continue
        head = line.split(":", 1)[0].strip()
        if head not in keys:
            bad.append(f"[{section}] {line[:80]}")
    assert not bad, "这些行不是以合法参数名开头（多半是某一行被断开了）：\n" + "\n".join(bad)


def test_installer_script_has_no_dangling_define_references():
    """引用了不存在的 #define 会在编译期才报错，这里提前挡掉。"""
    text = (PKG / "installer.iss").read_text(encoding="utf-8")
    defined = set(re.findall(r"^#define\s+(\w+)", text, re.M)) | {
        "AppName", "AppVersion", "AppPublisher", "AppURL", "AppExe", "SrcDir", "OutDir"}
    used = set(re.findall(r"\{#(\w+)\}", text))
    assert used <= defined, f"引用了没定义的：{sorted(used - defined)}"


def test_installer_windows_paths_are_not_broken_across_lines():
    """`_internal\\nvidia` 这类路径必须完整落在一行里（回归：曾被 \\n 转义成换行）。"""
    text = (PKG / "installer.iss").read_text(encoding="utf-8")
    for frag in (r"_internal\nvidia\*", r"_internal\nvrtc*.dll", r"_internal\nvJitLink*.dll"):
        assert frag in text, f"{frag} 不在文件里（可能被断成了两行）"
    for line in text.split("\n"):
        assert not re.match(r"^(vidia|vrtc|vJitLink|internal)\b", line.strip()), f"这行像是被断开的：{line[:60]}"


def test_spec_hidden_imports_all_exist():
    """spec 里列的隐藏导入必须都能 import，否则打包后会少节点。"""
    text = (PKG / "cvflow.spec").read_text(encoding="utf-8")
    block = re.search(r"^hidden = \[(.*?)\]", text, re.S | re.M)
    assert block, "spec 里没找到 hidden 列表"
    mods = re.findall(r'"([\w.]+)"', block.group(1))
    assert len(mods) >= 10
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    assert not missing, f"这些模块 import 不了：{missing}"


def test_gpu_split_patterns_agree_between_installer_and_build_script():
    """安装脚本和构建脚本对"哪些文件属于 GPU 组件"的判断必须一致。

    不一致的话，构建脚本报的组件体积就是假的——而那正是构建完唯一的核对手段。
    """
    iss = (PKG / "installer.iss").read_text(encoding="utf-8")
    ps1 = (PKG / "build.ps1").read_text(encoding="utf-8-sig")
    for frag in ("cudnn", "cublas", "cudart64", "cufft", "curand", "nvrtc", "nvJitLink",
                 "onnxruntime_providers_cuda", "onnxruntime_providers_tensorrt", r"_internal\nvidia"):
        assert frag in iss, f"installer.iss 里没有 {frag}"
        assert frag in ps1, f"build.ps1 里没有 {frag}"


@pytest.mark.parametrize("name", ["build.ps1"])
def test_powershell_scripts_are_utf8_with_bom(name):
    """PowerShell 5.1 按 ANSI 读无 BOM 的脚本，中文会乱码甚至解析失败。"""
    raw = (PKG / name).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), f"{name} 缺少 UTF-8 BOM"
    assert b"\r\n" in raw, f"{name} 应该用 CRLF 换行"


#: Windows PowerShell 5.1 解析不了的 PS7 专有语法。这类错误是**解析期**的——
#: 脚本一行都不会执行，而且 CI 上用 pwsh（7.x）跑完全看不出来。
PS7_ONLY = [
    (r"\)\s*\?\.", "?. 空条件运算符"),
    (r"\?\?", "?? 空合并运算符"),
    (r"&&|\|\|", "&& / || 管道链运算符"),
    (r"-Parallel\b", "ForEach-Object -Parallel"),
    (r"\$PSStyle\b", "$PSStyle"),
    (r"ConvertFrom-Json\s+[^\n]*-AsHashtable", "ConvertFrom-Json -AsHashtable"),
    (r"\bJoin-String\b", "Join-String"),
]


@pytest.mark.parametrize("name", ["packaging/build.ps1", "install.ps1"])
def test_powershell_scripts_parse_on_windows_powershell_51(name):
    """脚本不能用 PS7 专有语法：产线和开发机上的默认 PowerShell 往往还是 5.1。

    实际踩过：build.ps1 里一个 `(Get-Command ...)?.Source` 让整个脚本在 5.1 上解析失败，
    而 CI 用 pwsh 7 跑，一点问题都没报出来。
    """
    path = Path(__file__).resolve().parent.parent / name
    text = path.read_text(encoding="utf-8-sig")
    hits = []
    for pattern, what in PS7_ONLY:
        for m in re.finditer(pattern, text):
            line = text[:m.start()].count("\n") + 1
            if text.split("\n")[line - 1].strip().startswith("#"):
                continue                                   # 注释里提到不算
            hits.append(f"{name}:{line} 用了 {what}")
    assert not hits, "Windows PowerShell 5.1 解析不了这些写法：\n" + "\n".join(hits)


@pytest.mark.parametrize("name", ["packaging/build.ps1", "install.ps1"])
def test_powershell_scripts_have_no_python_isms(name):
    """PowerShell 脚本里不能混进 Python 写法（三引号文档字符串最容易手滑）。

    三引号在 PowerShell 里不是注释，会被当成字符串拼接解析；写在 param() 前面更是直接
    解析错误——脚本一行都不执行。PowerShell 的函数注释要用 # 或者 <# #>。
    """
    quotes = chr(34) * 3
    text = (Path(__file__).resolve().parent.parent / name).read_text(encoding="utf-8-sig")
    assert quotes not in text, f"{name} 里有 Python 三引号"
    # 用了 param() 的函数，param() 必须是函数体的第一条语句
    for m in re.finditer(r"^function\s+([\w-]+)\s*\{", text, re.M):
        body = text[m.end():m.end() + 400]
        stmts = [l.strip() for l in body.split("\n") if l.strip() and not l.strip().startswith("#")]
        if any(x.startswith("param(") for x in stmts[:4]):
            assert stmts[0].startswith("param("), \
                f"{name} 的 {m.group(1)}：param() 必须是函数体第一条语句，实际是 {stmts[0][:40]!r}"

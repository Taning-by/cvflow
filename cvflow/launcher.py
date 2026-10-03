"""桌面快捷方式与启动辅助。

`cvflow shortcut` 在桌面创建一个带图标的快捷方式，点击即可打开软件，不再需要命令行：
  Windows → 桌面上的 CVFlow.lnk（指向无控制台窗口的 cvflow-gui.exe）
  Linux   → ~/Desktop/CVFlow.desktop 以及应用菜单项
  macOS   → ~/Desktop/CVFlow.command
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "ui" / "assets"


def gui_executable() -> Path:
    """当前环境里的无控制台 GUI 入口（pip 安装时由 [project.gui-scripts] 生成）。"""
    import sysconfig
    name = "cvflow-gui.exe" if platform.system() == "Windows" else "cvflow-gui"
    for d in (Path(sys.executable).parent, Path(sysconfig.get_path("scripts") or "")):  # 不解析软链接，保留虚拟环境目录
        cand = d / name
        if cand.exists():
            return cand
    found = shutil.which("cvflow-gui")
    if found:
        return Path(found)
    raise FileNotFoundError("没有找到 cvflow-gui，请先在虚拟环境里执行 pip install -e . 安装")


def desktop_dir() -> Path:
    home = Path.home()
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                                 capture_output=True, text=True, timeout=15)
            if out.stdout.strip():
                return Path(out.stdout.strip())
        except Exception:
            pass
        return home / "Desktop"
    xdg = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True) if shutil.which("xdg-user-dir") else None
    if xdg is not None and xdg.stdout.strip() and Path(xdg.stdout.strip()).is_dir():
        return Path(xdg.stdout.strip())
    return home / "Desktop"


def create_shortcut(solution: str | None = None, name: str = "CVFlow") -> Path:
    """在桌面创建快捷方式；``solution`` 给定时快捷方式直接打开该方案。返回创建的文件路径。"""
    exe = gui_executable()
    args = f'"{Path(solution).resolve()}"' if solution else ""
    system = platform.system()
    desk = desktop_dir()
    desk.mkdir(parents=True, exist_ok=True)
    if system == "Windows":
        lnk = desk / f"{name}.lnk"
        ico = ASSETS / "cvflow.ico"
        ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
              f"$s.TargetPath='{exe}';$s.Arguments='{args}';$s.WorkingDirectory='{Path.cwd()}';"
              f"$s.IconLocation='{ico}';$s.Description='CVFlow 工业视觉';$s.Save()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=30)
        return lnk
    if system == "Darwin":
        cmd = desk / f"{name}.command"
        cmd.write_text(f'#!/bin/bash\ncd "{Path.cwd()}"\n"{exe}" {args}\n', encoding="utf-8")
        cmd.chmod(0o755)
        return cmd
    # Linux / 其它 freedesktop 桌面
    content = (f"[Desktop Entry]\nType=Application\nName={name}\nComment=CVFlow 工业视觉\n"
               f"Exec=\"{exe}\" {args}\nIcon={ASSETS / 'cvflow.png'}\nTerminal=false\nCategories=Development;Graphics;\n"
               f"Path={Path.cwd()}\n")
    apps = Path.home() / ".local" / "share" / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    (apps / f"{name}.desktop").write_text(content, encoding="utf-8")
    entry = desk / f"{name}.desktop"
    entry.write_text(content, encoding="utf-8")
    entry.chmod(0o755)
    return entry

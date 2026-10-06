"""把虚拟环境清成"能干净装上 onnxruntime-gpu"的状态。给 install.sh / install.ps1 调用，也可以手工跑。

    python tools/venv_clean.py isolate   关掉 include-system-site-packages
    python tools/venv_clean.py purge     删掉环境里 onnxruntime 的残留目录与 dist-info

这两件事对应的是两个很难自己看出来的坑：

* **isolate**：虚拟环境带 ``--system-site-packages`` 建出来时，系统 site-packages 里的 CPU 版
  onnxruntime 会**盖掉**环境里的 onnxruntime-gpu（环境的路径在前，但模块名相同，谁先被找到用谁；
  环境里那份被卸掉或没装上时就轮到系统那份），而 ``pip uninstall`` 在环境内删不掉它
  （只会提示 outside environment）。症状是"装了 GPU 版，后端里却只有 CPU"。
* **purge**：onnxruntime 和 onnxruntime-gpu 装的是同一个 ``onnxruntime/`` 目录，混装过的环境里
  经常出现"dist-info 还在、模块目录已经没了"的半残状态。这时 pip 认为包已安装，
  ``pip install`` 直接跳过，于是怎么装都装不上。

只允许用虚拟环境里的 python 执行，免得误删系统文件。
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
import sysconfig


def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def isolate() -> int:
    """把 pyvenv.cfg 里的 include-system-site-packages 改成 false。"""
    cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    if not os.path.exists(cfg):
        print(f"没有 {cfg}，跳过")
        return 0
    with open(cfg, encoding="utf-8") as f:
        text = f.read()
    new = re.sub(r"(?im)^(include-system-site-packages\s*=\s*)true\s*$", r"\1false", text)
    if new == text:
        print("虚拟环境本来就是隔离的（include-system-site-packages = false）")
        return 0
    shutil.copyfile(cfg, cfg + ".bak")
    with open(cfg, "w", encoding="utf-8") as f:
        f.write(new)
    print("已关掉 include-system-site-packages（原文件备份为 pyvenv.cfg.bak）——"
          "系统 site-packages 里的 onnxruntime 不会再盖掉环境里的 GPU 版")
    return 0


def purge() -> int:
    """删掉本环境 site-packages 里 onnxruntime 的模块目录和 dist-info。"""
    removed = []
    for base in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        for pat in ("onnxruntime", "onnxruntime_gpu", "onnxruntime-*.dist-info",
                    "onnxruntime_gpu-*.dist-info", "onnxruntime*.egg-info"):
            for path in glob.glob(os.path.join(base, pat)):
                shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
                removed.append(path)
    for path in removed:
        print(f"删掉残留 {path}")
    if not removed:
        print("环境里没有 onnxruntime 残留")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    actions = {"isolate": isolate, "purge": purge}
    if len(args) != 1 or args[0] not in actions:
        print(__doc__)
        return 1
    if not _in_venv():
        print("✗ 必须用虚拟环境里的 python 执行（例如 .venv/bin/python tools/venv_clean.py "
              f"{args[0]}），拒绝在系统 python 下动文件", file=sys.stderr)
        return 1
    return actions[args[0]]()


if __name__ == "__main__":
    raise SystemExit(main())

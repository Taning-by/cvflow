"""打包版的命令行入口（有控制台）。

装完之后现场排查全靠它，尤其是：

    cvflow gpu                 看装了哪个后端、CUDA 能不能加载、认不认得显卡
    cvflow gpu 模型.onnx --bench  实测 GPU 比 CPU 快几倍、算子落在哪个后端
"""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()
    from cvflow.cli import force_utf8_output
    force_utf8_output()          # 输出被重定向时别因为中文崩掉（英文 Windows 的 cp1252）
    from cvflow.cli import main as cli_main
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

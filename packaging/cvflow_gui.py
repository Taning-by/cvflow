"""打包版的窗口入口（无控制台）。PyInstaller 从这里起步。"""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    # 打包后的程序里，子进程会重新执行这个 exe；不先冻结就会无限开窗口
    multiprocessing.freeze_support()
    from cvflow.ui import main as gui_main
    return gui_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

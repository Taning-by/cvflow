# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：一次产出两个 exe，共用同一份 _internal。

    CVFlow.exe   窗口程序，无控制台（双击用）
    cvflow.exe   命令行，有控制台（cvflow gpu 自检、serve 无界面生产模式）

CUDA / cuDNN 的 DLL 按 `nvidia/<包>/bin/` 的原布局收进来，运行时由
cvflow.operators.dl._open_cudnn_search_path() 加进 DLL 搜索路径——打包后没有
site-packages，那段代码会改从 _internal 里找。

这些 DLL 加上 onnxruntime_providers_cuda.dll 一共约 1.9 GB，安装程序把它们划成
可选的「GPU 推理支持」组件，不勾就只装基础部分（约 500 MB），软件照常跑在 CPU 上。
"""
import glob
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT))

block_cipher = None


def nvidia_binaries():
    """按原样收走 site-packages/nvidia/<包>/bin 下的 DLL，保持相对布局。

    build.ps1 -SkipGpu 会设 CVFLOW_SKIP_NVIDIA=1：只打基础部分，省掉约 1.6 GB 的下载与打包时间
    （CI 上验证打包链路用的就是这条）。
    """
    import sysconfig
    if os.environ.get("CVFLOW_SKIP_NVIDIA"):
        # 只打 ASCII：spec 跑在 PyInstaller 的进程里，输出被重定向时用的是本地编码，
        # 英文 Windows 是 cp1252，打中文会直接抛 UnicodeEncodeError
        print("spec: CVFLOW_SKIP_NVIDIA=1, skipping CUDA / cuDNN runtime libraries")
        return []
    out = []
    for root in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        for d in sorted(glob.glob(os.path.join(root, "nvidia", "*", "bin"))):
            rel = os.path.relpath(d, root)                      # nvidia/<包>/bin
            for dll in sorted(glob.glob(os.path.join(d, "*.dll"))):
                out.append((dll, rel))
    return out


#: GPU 组件独有的文件：CUDA provider 本身（Windows 上 325 MB）和 TensorRT provider。
#: 安装程序把它们和 CUDA / cuDNN 运行库一起划成可选组件；-SkipGpu 构建时干脆不收进来。
GPU_ONLY = ("providers_cuda", "providers_tensorrt")


def onnxruntime_binaries():
    libs = collect_dynamic_libs("onnxruntime")
    if os.environ.get("CVFLOW_SKIP_NVIDIA"):
        libs = [(src, dst) for src, dst in libs if not any(k in os.path.basename(src) for k in GPU_ONLY)]
    return libs


datas = collect_data_files("cvflow", includes=["ui/assets/*"])
binaries = onnxruntime_binaries() + nvidia_binaries()

hidden = [
    "cvflow.operators.source", "cvflow.operators.preprocess", "cvflow.operators.analysis",
    "cvflow.operators.dl", "cvflow.operators.logic", "cvflow.operators.output",
    "cvflow.camera.folder", "cvflow.camera.opencv_cam",
    "cvflow.comm.tcp", "cvflow.comm.serial_dev", "cvflow.comm.modbus",
    "onnxruntime", "onnxruntime.capi._pybind_state",
]

excludes = ["tkinter", "matplotlib", "pytest", "IPython", "notebook", "torch", "torchvision",
            "PySide6.QtWebEngineCore", "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization"]

gui = Analysis([str(ROOT / "packaging" / "cvflow_gui.py")], pathex=[str(ROOT)],
               binaries=binaries, datas=datas, hiddenimports=hidden, excludes=excludes,
               hookspath=[], runtime_hooks=[], cipher=block_cipher, noarchive=False)
cli = Analysis([str(ROOT / "packaging" / "cvflow_cli.py")], pathex=[str(ROOT)],
               binaries=[], datas=[], hiddenimports=hidden, excludes=excludes,
               hookspath=[], runtime_hooks=[], cipher=block_cipher, noarchive=False)

MERGE((gui, "cvflow_gui", "CVFlow"), (cli, "cvflow_cli", "cvflow"))

if os.environ.get("CVFLOW_SKIP_NVIDIA"):
    # onnxruntime 自带的 PyInstaller 钩子也会收 provider，所以要在最终清单上再滤一遍，
    # 光过滤自己加的那份不够（基础包会白白大 400 MB，而且"没装 GPU 组件"的状态不自洽）
    def _drop_gpu(toc):
        return TOC([(n, p, t) for n, p, t in toc if not any(k in os.path.basename(n) for k in GPU_ONLY)])

    gui.binaries = _drop_gpu(gui.binaries)
    cli.binaries = _drop_gpu(cli.binaries)

gui_pyz = PYZ(gui.pure, gui.zipped_data, cipher=block_cipher)
cli_pyz = PYZ(cli.pure, cli.zipped_data, cipher=block_cipher)

icon = str(ROOT / "cvflow" / "ui" / "assets" / "cvflow.ico")
icon = icon if os.path.isfile(icon) else None

gui_exe = EXE(gui_pyz, gui.scripts, [], exclude_binaries=True, name="CVFlow",
              debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
              console=False, icon=icon)
cli_exe = EXE(cli_pyz, cli.scripts, [], exclude_binaries=True, name="cvflow",
              debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
              console=True, icon=icon)

COLLECT(gui_exe, gui.binaries, gui.zipfiles, gui.datas,
        cli_exe, cli.binaries, cli.zipfiles, cli.datas,
        strip=False, upx=False, name="CVFlow")

# 安装环境（从零到 CUDA 跑起来）

这份文档只讲一件事：**按什么顺序装，一次装对，并且能确认深度学习真的跑在显卡上。**

全程只有一个关键规则：

> `onnxruntime`（CPU 版）和 `onnxruntime-gpu`（GPU 版）是**同一个 Python 模块**的两个发行包，
> 谁后装谁覆盖。要用 CUDA，就**从一开始只装 GPU 版**，不要先装 CPU 版再补。

最常见的症状是这条日志

```
请求的推理后端 cuda 不可用（当前安装的 onnxruntime 里没有这个后端），实际使用 CPUExecutionProvider
```

这是上面那条规则被违反的结果：`pip install -e ".[dev,cpu]"` 装的是 CPU 版，里面根本没有编译 CUDA 后端。
不是驱动问题，也不是模型问题，**只是装错了包**。已经装成 CPU 版、要切到 GPU 的，直接跳到
[第 3 节](#3-已经装成-cpu-版了怎么切过去)。

---

## 0. 一键装（推荐，不用记步骤）

仓库里带了安装脚本，把下面讲的检查、清理、安装、验证一次做完；装乱了加 `-Recreate` / `--recreate`
从头来，代码不受影响。

```powershell
# Windows（PowerShell，在仓库根目录执行）
git clone https://github.com/Taning-by/cvflow.git
cd cvflow
.\install.ps1                 # 有 NVIDIA 显卡：装 GPU 版并验证 CUDA
.\install.ps1 -Cpu            # 没有 NVIDIA 显卡
.\install.ps1 -Recreate       # 之前装乱了：删掉虚拟环境从头装
.\install.ps1 -Model D:\models\best.onnx   # 顺便拿真模型实测一次
```

```bash
# Linux / macOS
git clone https://github.com/Taning-by/cvflow.git
cd cvflow
./install.sh                  # 有 NVIDIA 显卡
./install.sh --cpu            # 没有 NVIDIA 显卡
./install.sh --recreate
./install.sh --model best.onnx
```

脚本做的事和下面第 2 节的手工步骤一一对应，另外多做三件只靠手敲很容易漏的：

* 装之前**检查显卡驱动版本**够不够 CUDA 12 用（不够就直接告诉你去更新驱动，不会白下 2 GB）；
* 把虚拟环境**隔离**掉（`include-system-site-packages = false`），并清掉 onnxruntime 的残留
  —— 这两样是"装了 GPU 版却只有 CPU 后端"最隐蔽的两个来源，见[第 7 节](#7-常见错误对照表)最后两行；
* 装完跑 `cvflow gpu --require-gpu`，**CUDA 没真正跑起来就以非零退出码失败**，不会假装装好了。

想知道每一步在干什么、或者脚本在你的机器上卡住了，再往下看手工步骤。

---

## 1. 装之前先确认这几件事（Windows）

这几项不满足，后面怎么装都不会成功，先花两分钟查清楚。

| 查什么 | 怎么查 | 要求 |
|---|---|---|
| NVIDIA 显卡和驱动 | `nvidia-smi` | 命令能出表格，右上角 `CUDA Version` ≥ **12.0** |
| 驱动版本 | 同上，表头 `Driver Version` | Windows ≥ **527.41**（Linux ≥ 525.60）。低于这个数去 nvidia.com 更新驱动 |
| 显卡算力 | [nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) | ≥ **5.0**（Maxwell 往后，GTX 900 系以上都满足） |
| Python | `python -V`、`python -c "import struct;print(struct.calcsize('P')*8)"` | **3.10 ~ 3.13**（Windows 上 3.14 也可以），且必须是 **64 位** |

几点说明：

* `nvidia-smi` 里的 `CUDA Version` 是**驱动支持的最高版本**，不代表你装了 CUDA Toolkit。
  **本项目不需要你安装 CUDA Toolkit**，CUDA 12 和 cuDNN 9 的运行库会由 pip 一起装进虚拟环境。
* `nvidia-smi` 不存在 → 要么机器没有 NVIDIA 显卡（走[第 4 节](#4-没有-nvidia-显卡只用-cpu)），
  要么驱动没装好，先解决驱动。
* Python 3.9 及以下不支持。3.14 在 Windows 上能装齐（Linux 的 PySide6 目前还没有 3.14 的轮子），
  产线上建议用 3.11 / 3.12 这种轮子最全的版本。
* 不用安装 TensorRT。它是额外的加速后端，不装也能用 CUDA。

---

## 1.5 机器上只有 conda 怎么办

本项目用的是标准的 `venv` + `pip`，**不要用 conda 装**。原因有两个，都是排查起来很费劲的那种：

1. conda 的 `python.exe` 依赖 `<env>\Library\bin` 里的 DLL（ssl、sqlite 这些），那个目录只在
   conda 环境**激活时**才在 `PATH` 上。基于 conda 解释器建出来的 venv 继承了这个依赖，
   以后不激活 conda 就可能报 `DLL load failed` 或 `No module named '_ssl'`。
2. conda 环境里装过 `cudnn` / `cudatoolkit` 的话，它们的 DLL 也在 `PATH` 上，
   会和 pip 装进环境的 CUDA 12 / cuDNN 9 撞版本。

所以 `install.ps1` / `install.sh` **默认拒绝**用 conda 里的解释器建 venv（靠 `sys.base_prefix`
下有没有 `conda-meta` 目录判断；conda 环境本身不是 venv，`sys.prefix` 认不出来），
并会提醒你当前是不是还有 conda 环境处于激活状态。

干净的做法，三步：

```powershell
# 1) 装一个普通 CPython（任选其一，都不会动你的 conda）
winget install Python.Python.3.12
#   或者去 python.org 下 64 位安装包，装的时候不勾 "Add to PATH" 也没关系，下一步用绝对路径

# 2) 退出 conda，让这个窗口干净
conda deactivate
conda config --set auto_activate_base false     # 以后新开的窗口不再自动进 base（可选但推荐）

# 3) 明确用那个 CPython 装
.\install.ps1 -Python "C:\Users\<你>\AppData\Local\Programs\Python\Python312\python.exe"
#   装了 py 启动器更省事：.\install.ps1 -Python "py -3.12"
```

装完确认这个 venv 跟 conda 没关系：

```powershell
.venv\Scripts\python -c "import sys;print(sys.base_prefix)"
```

打出来的路径里**不带 conda / miniconda / anaconda** 就对了。之后用 `.venv\Scripts\activate`
激活，不要再 `conda activate`。

确实非要用 conda 的解释器（比如机器上就是装不了别的 Python）：加 `-AllowConda` / `--allow-conda`
就能过，但上面那两个风险你自己要记着，出问题先想到它。

另外，本项目**不支持**直接装进 conda 环境（`conda activate` 后 `pip install -e .`）：
`tools/venv_clean.py` 的清理动作只肯在 venv 里动文件，conda 环境会被它拒绝，
混装过 onnxruntime 的 conda 环境清不干净。

---

## 2. 正确的安装顺序（Windows + NVIDIA 显卡）

下面六步照顺序执行，不要跳、不要换序。命令在 PowerShell 或 CMD 里都一样。

### 第 1 步：取代码

```powershell
git clone https://github.com/Taning-by/cvflow.git
cd cvflow
```

### 第 2 步：建虚拟环境并激活

```powershell
python -m venv .venv
.venv\Scripts\activate
```

提示符前面出现 `(.venv)` 才算激活成功。PowerShell 报“禁止运行脚本”时，
用 `.venv\Scripts\activate.bat`，或者先执行一次
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

**后面每一条 `pip` 和 `cvflow` 命令都必须在激活状态下执行**，否则装到了系统 Python 里，
出来的现象就是“明明装了却说没有”。

### 第 3 步：升级 pip

```powershell
python -m pip install -U pip
```

旧版 pip 解析不了 `onnxruntime-gpu[cuda,cudnn]` 这种带 extra 的依赖，会静默少装 CUDA 运行库。

### 第 4 步：一次装好，**只装 GPU 版**

```powershell
pip install -e ".[dev,gpu]"
```

这一条就够了。它会装：

* 核心依赖（numpy、OpenCV、PySide6 界面、pymodbus、pyserial）
* 测试依赖（pytest、onnx）
* **`onnxruntime-gpu`**，以及它的 `cuda` / `cudnn` extra 带来的
  `nvidia-cuda-runtime-cu12`、`nvidia-cublas-cu12`、`nvidia-cudnn-cu12`、
  `nvidia-cufft-cu12`、`nvidia-curand-cu12`、`nvidia-cuda-nvrtc-cu12`

装的是 **CUDA 12** 那条线（`onnxruntime-gpu>=1.21,<1.27`），这是故意卡的上限：

* onnxruntime-gpu **1.27 起换成了 CUDA 13**，要求**驱动 ≥ 580**，而且只支持**算力 7.5 以上**
  （Turing / RTX 20 系往后）——产线上还在用的 GTX 10 系、Quadro P 系（Pascal，算力 6.1）直接用不了。
* 卡在 CUDA 12 上，驱动 ≥ 527.41、算力 ≥ 5.0 就能跑，老卡新卡都覆盖，新卡也不会慢。
* 新机器确实想用 CUDA 13 版（驱动 ≥ 580、算力 ≥ 7.5）：`pip install -e ".[dev,gpu-cuda13]"`。
  两条线互斥，切换前照[第 3 节](#3-已经装成-cpu-版了怎么切过去)把 onnxruntime 卸干净。

下载量约 2 GB，第一次装会比较久。装完**不需要**设置 `PATH`、`CUDA_PATH`，
也**不需要**系统级的 CUDA Toolkit 或 cuDNN：这些库在虚拟环境里不在系统搜索路径上，
CVFlow 会在建立推理会话之前调 `onnxruntime.preload_dlls()` 把它们预加载进来。

不要做的事：

* ❌ `pip install -e ".[dev,cpu]"` 然后再装 gpu —— CPU 版会把 GPU 版的模块覆盖掉
* ❌ `pip install -r requirements.txt` —— 那个文件是给 CPU 环境用的，会把 CPU 版装回来
* ❌ 同时写 `".[cpu,gpu]"` —— 两个包会一起装上，互相覆盖

### 第 5 步：自检（这一步不要省）

```powershell
cvflow gpu
```

期望看到的输出（关键三行已标注）：

```
== 推理运行时 ==
  onnxruntime 1.23.2  C:\...\.venv\Lib\site-packages\onnxruntime
  已安装的包：onnxruntime-gpu                      ← 只有这一个，没有 onnxruntime

== 后端 ==
  包里编译进来的：TensorrtExecutionProvider、CUDAExecutionProvider、CPUExecutionProvider
  TensorrtExecutionProvider    加载失败：...nvinfer...   ← 没装 TensorRT，正常，忽略
  CUDAExecutionProvider        可用                      ← 必须是“可用”

== 显卡 ==
  0 号卡 NVIDIA GeForce RTX 4060　驱动 560.94　已用 1024/8188 MiB   ← 认得出显卡
```

标了箭头的四处任意一处不对，去[第 7 节](#7-常见错误对照表)查。

写脚本或做产线上线检查时，用 `cvflow gpu --require-gpu`：CUDA 没跑起来会返回非零退出码（2），
可以直接拿它当"环境合格"的判据。

### 第 6 步：拿真模型实测

```powershell
cvflow gpu D:\models\best.onnx
```

```
== 实测加载 ==
  实际使用的后端：CUDAExecutionProvider（0 号卡）
  加载耗时：1483 ms
  本进程显存：0 → 942 MiB（+942）        ← 显存涨了，这是跑在显卡上最硬的证据
```

**显存涨了才算真的用上了 CUDA。** 后端写着 CUDA 但显存没变，说明看的不是同一块卡（用 `--device` 指定）。

装完跑一遍测试确认环境完整：

```powershell
pytest -q
```

### 装好之后，节点怎么设

深度学习节点的**推理后端**参数保持默认的 `auto` 就行：它按 TensorRT → CUDA → CPU
挑第一个真能加载的，装了 GPU 版就会自动走 CUDA，不用改任何参数。

想强制用显卡、让装错环境时直接报警（而不是静悄悄退回 CPU），就把它填成 `cuda`：
环境不对时日志里会出现本文开头那条警告。多卡机器用**显卡编号**把不同节点分到不同卡上。

运行时日志里这一行是最终确认：

```
ONNX 检测 (YOLO)：模型 best.onnx 使用推理后端 CUDAExecutionProvider（0 号卡）
```

---

## 3. 已经装成 CPU 版了怎么切过去

先装了 `.[cpu]`、后来要用显卡，是最常见的情形。**不能直接 `pip install -e ".[dev,gpu]"`** ——
那样 `onnxruntime` 和 `onnxruntime-gpu` 会同时存在，CPU 版的模块文件还在原地，
结果仍然没有 CUDA 后端。按这四步来：

```powershell
# 1) 两个包都卸掉。反复执行到两个都显示 "not installed" 为止
#    （一个包可能被装过多次，一次 uninstall 只卸掉最上面那层）
pip uninstall -y onnxruntime onnxruntime-gpu
pip uninstall -y onnxruntime onnxruntime-gpu

# 2) 确认 site-packages 里的 onnxruntime 目录已经没了，有残留就手动删掉
python -c "import onnxruntime" 2>$null   # 应该报 ModuleNotFoundError
Remove-Item -Recurse -Force .venv\Lib\site-packages\onnxruntime   # 只在确实还存在时执行

# 3) 只装 GPU 版
pip install -e ".[dev,gpu]"

# 4) 自检
cvflow gpu
```

第 2 步容易被忽略，但它是“卸了又装还是只有 CPU 后端”的最常见原因：
pip 卸载只删它记录在案的文件，两个包混装过的目录里常有管不到的残留。

嫌麻烦就一条命令：`.\install.ps1 -Recreate`（Linux：`./install.sh --recreate`），
上面这些它全做了，还会多查驱动、多做一次验证。

手工走到底、或者反复试过都不对，最干净的办法是**把虚拟环境整个删了重来**（代码不受影响）：

```powershell
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev,gpu]"
cvflow gpu
```

---

## 4. 没有 NVIDIA 显卡（只用 CPU）

```powershell
pip install -e ".[dev,cpu]"
```

这时把节点的**推理后端**保持 `auto`。填 `cuda` 就会得到你现在看到的那条警告——
它本身不是故障，是在告诉你“你要的后端没有，已经退回 CPU 了”。

AMD / Intel 显卡、以及苹果的 M 系列芯片都走不了 `onnxruntime-gpu` 的 CUDA 后端，只能用 CPU 版。

---

## 5. Linux 下的差别

顺序完全一样，只有两处不同：

```bash
python -m venv .venv
source .venv/bin/activate          # 激活方式不同
python -m pip install -U pip
pip install -e ".[dev,gpu]"
cvflow gpu
```

* 驱动要求 ≥ 525.60，同样**不需要**系统装 CUDA Toolkit。
* 同样**不需要**配 `LD_LIBRARY_PATH`，`preload_dlls()` 会处理。
  （手动跑 `python -c "import onnxruntime"` 再建会话时会缺库，是因为绕过了 CVFlow 的预加载，不是环境坏了。）

---

## 6. 产线机器上不了网怎么装（离线包）

工控机常常是不通外网的。做法是**在一台联网的机器上按目标机器的平台把轮子全下下来**，
拷过去离线装。注意下载时要指定**目标机器**的平台和 Python 版本，不是下载机的。

联网的机器上（这里以目标机器是 Windows + Python 3.12 为例）：

```bash
# 先打出 cvflow 的包（仓库根目录执行，产物在 dist/）
pip install build && python -m build

# 再把 cvflow 和它的全部依赖（含 CUDA 12 / cuDNN 9）按 Windows 的轮子下下来，约 2 GB
pip download -d cvflow-offline \
    --platform win_amd64 --python-version 3.12 --only-binary=:all: \
    "dist/cvflow-0.1.0-py3-none-any.whl[dev,gpu]"
```

把 `cvflow-offline` 目录和仓库一起拷到产线机器，然后：

```powershell
python -m venv .venv                       # 别加 --system-site-packages
.venv\Scripts\activate
pip install --no-index --find-links cvflow-offline -e ".[dev,gpu]"
cvflow gpu --require-gpu                   # 照样要自检
```

三个容易出错的地方：

* `--platform` / `--python-version` 写的是**目标机器**的；写错了下回来的轮子装不上（会报
  "not a supported wheel on this platform"）。Linux 目标机器一般用
  `--platform manylinux_2_28_x86_64`，老系统用 `manylinux2014_x86_64`。
* 必须带 `--only-binary=:all:`，否则 pip 可能下源码包，到了离线机器上还要编译。
* `pip download` 和产线机器的 Python **小版本要对上**（3.12 的轮子不能给 3.11 用）。

公司有内网 pip 镜像的话更简单，照正常步骤装就行，只要确认镜像里有
`onnxruntime-gpu 1.21 ~ 1.26` 和那几个 `nvidia-*-cu12` 包。

---

## 7. 常见错误对照表

| 看到的信息 | 真正的原因 | 怎么修 |
|---|---|---|
| `请求的推理后端 cuda 不可用（当前安装的 onnxruntime 里没有这个后端）` | 装的是 CPU 版 `onnxruntime` | [第 3 节](#3-已经装成-cpu-版了怎么切过去) |
| `请求的推理后端 cuda 不可用（这个后端在本机加载失败）` | 包里有 CUDA 后端，但运行库加载不了：缺 `nvidia-*` 包，或驱动太老 | 确认驱动 ≥ 527.41；重装 `pip install -e ".[dev,gpu]"` 把 extra 带上 |
| `同时安装了 onnxruntime 与 onnxruntime-gpu` | 两个包混装，互相覆盖 | [第 3 节](#3-已经装成-cpu-版了怎么切过去) |
| `推理后端 TensorrtExecutionProvider 的运行库加载不了…跳过它` | 没装 TensorRT 本体 | **不用管**。`auto` 会跳过它继续用 CUDA |
| `cvflow gpu` 说“没找到 nvidia-smi” | 没有 NVIDIA 显卡，或驱动没装好 | 装/更新驱动；确实没有独显就走[第 4 节](#4-没有-nvidia-显卡只用-cpu) |
| `缺 CUDA 运行库…libcublasLt / cublasLt64_12.dll` | cuDNN 没装上（cuBLAS 由它带入） | `pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12`，或按[第 3 节](#3-已经装成-cpu-版了怎么切过去)重装 |
| 后端是 CUDA，但显存一点没涨 | 看的不是同一块卡 | `cvflow gpu 模型.onnx --device 1` 指定卡号 |
| `DLL load failed` / `No module named '_ssl'`（venv 里） | 这个 venv 是用 conda 的解释器建的，conda 环境没激活时缺 `Library\bin` 里的 DLL | 用普通 CPython 重建：`.\install.ps1 -Recreate -Python "C:\Python312\python.exe"`，见[第 1.5 节](#15-机器上只有-conda-怎么办) |
| 装完还是找不到 `cvflow` 命令 | 虚拟环境没激活 | 重新 `.venv\Scripts\activate`；或用 `python -m cvflow gpu` |
| `pip` 装的是 `onnxruntime-gpu 1.19/1.20`，CUDA 库一个没装 | 这两个版本**没有** `cuda`/`cudnn` extra | 本项目已要求 `>=1.21`；用公司内网镜像时确认镜像里有 1.21+ |
| 装的是 `onnxruntime-gpu 1.27+`，CUDA 后端却加载失败 | 1.27 起要 CUDA 13：驱动 < 580，或显卡算力 < 7.5（GTX 10 系这类 Pascal 卡） | 回到 CUDA 12 那条线：卸干净后 `pip install -e ".[dev,gpu]"`（已限制 <1.27） |
| `cvflow gpu` 第一行的路径**不在** `.venv` 里，还提示"被系统 site-packages 覆盖" | 虚拟环境是带 `--system-site-packages` 建的，系统里装过 CPU 版 onnxruntime，它盖掉了环境里的 GPU 版（`pip uninstall` 在环境内删不掉它，只说 outside environment） | `.venv\Scripts\python tools\venv_clean.py isolate` 后重装，或直接 `.\install.ps1 -Recreate` |
| `pip install` 说 requirement already satisfied，但 `import onnxruntime` 失败 / 没有 CUDA 后端 | 混装过的环境里"dist-info 还在、模块目录已经没了"，pip 以为装过就跳过 | `.venv\Scripts\python tools\venv_clean.py purge` 后重装，或 `.\install.ps1 -Recreate` |

---

## 8. 可选依赖

| extra | 装什么 | 什么时候要 |
|---|---|---|
| `dev` | pytest、onnx | 跑测试、校验模型，平时都带上 |
| `gpu` | onnxruntime-gpu(<1.27) + CUDA 12 / cuDNN 9 | 有 NVIDIA 显卡，**默认用这个**：驱动 ≥ 527.41、算力 ≥ 5.0。与 `cpu` 互斥 |
| `gpu-cuda13` | onnxruntime-gpu(≥1.27) + CUDA 13 / cuDNN 9 | 新机器（驱动 ≥ 580、算力 ≥ 7.5）才行，老卡装了跑不起来。与 `cpu`、`gpu` 互斥 |
| `cpu` | onnxruntime | 没有 NVIDIA 显卡。**与 `gpu` 互斥** |
| `genicam` | harvesters | GigE / USB3 Vision 相机（还需厂商的 `.cti` 驱动） |
| `plc` | python-snap7 | 西门子 S7 PLC |
| `dl` | torch、torchvision | 只给自己写的 PyTorch 插件用 |

两点容易混淆的地方：

* **ONNX 走 CUDA 不需要 torch。** `.[dl]` 和推理后端没有关系，别为了“用 GPU”去装它。
* Windows 上 PyPI 的 `torch` 是 **CPU 版**。真要 PyTorch 用显卡，单独从官方索引装：
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`。
  这跟 `onnxruntime-gpu` 互不干扰，可以共存。

---

## 9. 为什么是这个顺序

留个记录，免得以后又绕回去：

1. **先查驱动，再装包。** 驱动不够的话 2 GB 下载完了照样加载失败，查驱动只要几秒。
2. **先建虚拟环境，再装任何东西。** 装错了可以整个目录删掉重来，不污染系统 Python。
3. **先升 pip，再装带 extra 的依赖。** 老 pip 会把 `[cuda,cudnn]` 悄悄忽略。
4. **一次就装 GPU 版，绝不先装 CPU 版。** 两个包共用模块名，先装 CPU 再补 GPU 必然要走一遍
   “卸干净 + 删残留”，比一次装对麻烦得多。
5. **虚拟环境必须是隔离的。** 带 `--system-site-packages` 建出来的环境，会被系统里那份 CPU 版
   onnxruntime 盖掉，而且在环境内 `pip uninstall` 删不掉它——产线机器上有人全局 `pip install onnxruntime`
   过，就会撞上这个。`install.sh` / `install.ps1` 会自动把这个开关关掉。
6. **装完立刻 `cvflow gpu`，再拿真模型实测。** 后端回退到 CPU 是**静默**的——
   流程照跑、结果照出，只是慢十倍。不主动验证，就只能等到现场节拍不达标时才发现。

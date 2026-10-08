# CVFlow — 开放的流程式工业视觉平台

CVFlow 是一个用 Python 全栈实现的工业视觉软件雏形，对标 VisionMaster / VisionPro 这类"拖拽流程 + 算子 + 通信"的平台，但把**算法节点做成了普通 Python 插件**：任何能用 Python 写出来的东西（OpenCV 算子、ONNX/PyTorch 模型、强化学习策略）都可以作为一个节点进入流程，并通过 TCP / UDP / 串口 / Modbus 与 PLC、机器人交互。

```
┌──────────────────────────── cvflow.ui (PySide6) ────────────────────────────┐
│  节点编辑器   图像视图(叠加层/ROI绘制)   参数面板   结果   通信   变量   日志 │
└──────────────────────────────────────────────────────────────────────────────┘
┌─────────────── cvflow.core ───────────────┐  ┌──────── cvflow.comm ────────┐
│ types  node  registry  graph  engine      │  │ TCP client/server  UDP      │
│ runtime(FlowRunner/Solution)  events      │  │ Serial  Modbus TCP (主/从)  │
│ variables  draw  paths                    │  │ 接收规则→触发  发送规则→结果 │
└───────────────────────────────────────────┘  └─────────────────────────────┘
┌──────────── cvflow.operators ─────────────┐  ┌──────── cvflow.camera ──────┐
│ Source Preprocess Analysis DeepLearning   │  │ folder(模拟)  OpenCV        │
│ Script Logic Output   + examples/plugins  │  │ GenICam(harvesters, 可选)   │
└───────────────────────────────────────────┘  └─────────────────────────────┘
```

核心层（`cvflow.core`）不依赖 Qt，可以无界面地跑在产线工控机上。

## 安装

需要 Python 3.10–3.13（64 位），Linux / Windows / macOS 均可。

### 装成 Windows 软件（安装包）

给产线机器用的是安装程序，不需要装 Python：

```powershell
CVFlow-Setup-0.1.0-x64.exe          # 双击，下一步到底
```

一个安装包，**GPU 推理支持是可勾选的组件**：主程序约 530 MB（装完就能用，推理跑 CPU），
勾上 GPU 组件再加约 1.9 GB（onnxruntime 的 CUDA provider + CUDA 12 / cuDNN 9 运行库）。
没有 NVIDIA 显卡的机器不勾即可，软件照常用。装完可选立即跑一次自检，当场知道 CUDA 行不行。

程序装在 `C:\Program Files\CVFlow`，方案/日志/存图在 `%ProgramData%\CVFlow`（可写）。
静默安装：`CVFlow-Setup-0.1.0-x64.exe /SILENT /DIR="C:\CVFlow"`。

自己打包：见 [packaging/README.md](packaging/README.md)（PyInstaller + Inno Setup，一条命令）。

### 一键装（推荐，开发用）

脚本会检查驱动、建好隔离的虚拟环境、装对推理运行时，**并在装完实测 CUDA**，没跑起来就报错退出：

```powershell
git clone https://github.com/Taning-by/cvflow.git
cd cvflow
.\install.ps1                # Windows。有 NVIDIA 显卡就装 GPU 版（约 2 GB），装完验证 CUDA
.\install.ps1 -Cpu           # 没有 NVIDIA 显卡
.\install.ps1 -Recreate      # 之前装乱了：删掉虚拟环境从头装（代码不受影响）
```

```bash
./install.sh                 # Linux / macOS，参数同上：--cpu / --recreate / --model best.onnx
```

机器上只有 conda 的话，**别用 conda 的解释器**（脚本默认会拒绝）：装个普通 CPython，
`conda deactivate` 之后用 `-Python` 明确指定它，理由和步骤见
[docs/install.md 第 1.5 节](docs/install.md#15-机器上只有-conda-怎么办)。

不想碰系统 Python / conda，或者机器上根本没有 Python，就让**项目自带一份**：

```powershell
.\install.ps1 -BootstrapPython     # 下载便携版 CPython 3.12 到 .python\（校验 SHA256）再用它建 .venv
```

装完 `.venv` 只认项目目录里的 `.python\`，和系统里的任何 Python / conda 无关
（细节见 [docs/install.md 第 1.6 节](docs/install.md#16-让项目自带-python推荐给不想碰系统-python--conda的情形)）。

### 手工装

```bash
git clone https://github.com/Taning-by/cvflow.git
cd cvflow
python -m venv .venv               # 别加 --system-site-packages，系统里的 onnxruntime 会盖掉环境里的
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip       # 老 pip 会悄悄忽略下面的 [cuda,cudnn]
```

然后**按机器二选一装推理运行时**，这一步选错了后面要卸干净重来：

```bash
pip install -e ".[dev,gpu]"        # 有 NVIDIA 显卡：onnxruntime-gpu + CUDA 12 / cuDNN 9（约 2 GB）
pip install -e ".[dev,cpu]"        # 没有 NVIDIA 显卡
```

装完**立刻自检**，别等跑流程时才发现悄悄回退到了 CPU：

```bash
cvflow gpu                      # 装了哪个包、CUDA 后端能不能加载、认不认得显卡
cvflow gpu 你的模型.onnx         # 实测：实际用哪个后端、显存涨了多少
cvflow gpu --require-gpu        # 给脚本/上线检查用：CUDA 没跑起来就返回非零退出码
cvflow gpu 你的模型.onnx --bench # 实测 GPU 比 CPU 快几倍，并列出每个算子落在哪个后端
pytest -q
```

**完整步骤、自检输出怎么看、常见错误对照表见 [docs/install.md](docs/install.md)。**

### 为什么必须二选一

`onnxruntime`（CPU 版）与 `onnxruntime-gpu`（GPU 版）是**同一个 Python 模块**的两个发行包，谁后装谁覆盖。
典型症状就是"装了 GPU 版却只有 CPU 后端"，日志里长这样：

```
请求的推理后端 cuda 不可用（当前安装的 onnxruntime 里没有这个后端），实际使用 CPUExecutionProvider
```

所以**用显卡就从一开始只装 `.[gpu]`**，不要先装 `.[cpu]` 再补。已经装错了的话：

```bash
pip uninstall -y onnxruntime onnxruntime-gpu       # 反复执行到两个都显示未安装
python -c "import onnxruntime"                     # 应报 ModuleNotFoundError；有残留目录就手动删掉
pip install -e ".[dev,gpu]"
cvflow gpu
```

同理，GPU 环境下**不要**再执行 `pip install -r requirements.txt`，那个文件是基础依赖清单，不含推理运行时。

GPU 版连同 CUDA 12 与 cuDNN 9 的 pip 包一起装好，**系统不需要另外装 CUDA Toolkit / cuDNN，也不用配
`LD_LIBRARY_PATH` / `PATH`**：这些库不在系统搜索路径里，软件会在建立推理会话前自动预加载它们。
只需要显卡驱动够新（Windows ≥ 527.41，Linux ≥ 525.60，即 `nvidia-smi` 右上角 `CUDA Version` ≥ 12.0）。
TensorRT 不用装，`auto` 会跳过它直接用 CUDA。

`.[gpu]` 走的是 **CUDA 12** 那条线（`onnxruntime-gpu>=1.21,<1.27`），故意没放开到 CUDA 13：1.27 起的
CUDA 13 版要求驱动 ≥ 580、显卡算力 ≥ 7.5，产线上常见的 GTX 10 系、Quadro P 系（Pascal）直接用不了。
卡在 CUDA 12 上，驱动 ≥ 527.41、算力 ≥ 5.0 就能跑，新卡也不会慢。新机器想用 CUDA 13 版：`.[dev,gpu-cuda13]`。

其它可选依赖：`.[genicam]`（harvesters，GigE/USB3 Vision 相机）、`.[plc]`（python-snap7，西门子 S7）。

**没有 torch 的 extra，这是故意的。** ONNX 走 CUDA 用不到 torch，而 torch 自带一整套 CUDA / cuDNN，
装进同一个环境后谁先加载谁当家（onnxruntime 发现 torch 已导入会主动让位），版本不匹配时推理会
悄悄退回 CPU。要用 PyTorch 插件模板就单独装，最好另开一个虚拟环境：
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`。

## 运行

```bash
cvflow nodes                                          # 列出 40 个内置节点
cvflow run examples/solutions/demo_holes.json -n 10   # 终端跑示例流程
cvflow gui examples/solutions/demo_holes.json         # 打开桌面程序
cvflow serve examples/solutions/demo_holes.json       # 无界面生产模式
cvflow gpu                                           # 自检推理环境（后端 / 显卡 / 实测加载）
pytest -q                                             # 35 个测试，界面测试 offscreen 运行
```

不安装也可以用 `python -m cvflow …` 直接运行。

方案文件里的路径（图像文件夹、模型、模板、存图目录、插件目录）都相对于方案文件所在目录解析，整个方案目录可以随意移动或从 git 克隆。

示例方案 `demo_holes.json`：

* 流程 **main**：文件夹取图 → 灰度 → 阈值 → 开运算 → Blob 计数(3 个孔) → 判定；同时用边缘卡尺测量工件宽度 → 判定；结果发布、叠加渲染、NG 时存图。
* 流程 **learning**：`examples/plugins/bandit_threshold.py` 里的 ε-greedy bandit 节点根据上一轮的奖励调整阈值，演示"带状态的学习型节点"如何接进流程。
* 通信：TCP 服务端监听 6000 端口，收到以 `TRIG` 开头的报文就触发 main，结束后按模板 `{status},{out.holes},{out.width:.1f}\n` 回发结果。

用 `nc` 模拟 PLC：先 `cvflow serve examples/solutions/demo_holes.json`，再 `printf 'TRIG\n' | nc -q1 127.0.0.1 6000`。

## 用图标打开软件

安装后虚拟环境里会生成无控制台窗口的入口 `cvflow-gui`（Windows 下是 `.venv\Scripts\cvflow-gui.exe`），直接双击就能打开。更方便的是让程序在桌面放一个带图标的快捷方式：

```bash
cvflow shortcut                                   # 桌面快捷方式，打开软件
cvflow shortcut examples/solutions/demo_holes.json  # 快捷方式直接打开指定方案
```

Windows 生成 `CVFlow.lnk`，Linux 生成 `CVFlow.desktop`（同时加入应用菜单），macOS 生成 `CVFlow.command`。界面里"文件 → 创建桌面快捷方式"也能做同样的事。不带方案启动时会自动打开上次使用的方案，"文件 → 最近打开"里有最近 8 个方案。

## 界面操作

界面分五个区：顶部是项目与执行工具栏（新建/打开/保存 · 流程选择 · 运行一次/连续运行/运行模式），
左侧是可搜索的算法工具箱，中央是**图像**与**流程**两个主工作区（可拖动分隔条调整比例、点右上角按钮
最大化其中一个、用「视图」菜单或流程区的按钮切换左右／上下排列，比例与排列会被记住），右侧是当前
选中节点的参数与本次输出，底部是结果／通信／变量／日志，状态栏显示当前帧、执行统计与所处模式。

* 左侧节点库拖到画布，或双击加到视图中心；从输出端口拖到输入端口连线，Delete 删除，F 适配视图（焦点在流程区时），按住右键（或中键、Alt+左键）拖动平移，滚轮缩放。
* 选中节点后右侧参数面板自动生成；ROI 类参数点 **Draw** 后在图像上拖矩形。
* "Auto-run on change" 开启时改参数立即重跑；**Run once (F5)** 手动跑一次；"Continuous" 按间隔循环。
* **Start run mode (F9)**：连接所有通信设备、启动所有流程的工作线程，此时流程被锁定，触发来自 PLC/定时器/手动。
* 底部 Communication 页：设备（TCP/UDP/串口/Modbus）、接收规则（触发流程或设变量）、发送规则（文本模板或 Modbus 寄存器写入）、报文监视与手动发送。

## 相机采集

菜单 **相机 → 相机管理**（工具栏也有入口），和 VisionMaster 的相机管理一样：

* **搜索相机**：用 GigE Vision 发现协议在本机每个网卡上广播，列出网络上所有 GigE 相机的 IP、MAC、厂商、型号、序列号、用户名，并标出"不在本机网段"的相机。搜索不依赖任何 SDK；装了海康 MVS SDK 或迈德威视 SDK 时会同时用各自的 SDK 枚举（含 USB3 相机），填了 GenTL 驱动时也会用 GenICam 枚举。
* **按 IP 添加**：向指定 IP 单播发现请求，没响应也可以先手动加入。
* **连接测试**：读相机的 GigE Vision 版本寄存器，报告在线与否和耗时；海康相机还会尝试用 SDK 打开。
* **强制 IP**：相机和网卡不在同一网段时，按 MAC 给相机下发临时 IP。
* **添加到流程**：生成一个"相机"节点，取图方式自动选择海康 MVS（`hik`）或 GenICam（`genicam`）。

相机节点参数：触发模式（沿用 / 自由采集 / 软触发 / 硬件触发）、曝光（微秒）、增益、超时，以及任意 GenICam 特性的 JSON。

### 支持哪些相机

| 取图方式 | 适用 | 需要装什么 |
|---|---|---|
| `folder` | 用一个文件夹里的图片模拟相机，开发调试用 | 无 |
| `opencv` | UVC 摄像头、视频流（RTSP 等） | 无 |
| `hik` | 海康机器人（HIKROBOT）GigE / USB3 | MVS SDK，`MVCAM_SDK_PATH` 指向 `MVS/Development/Samples/Python/MvImport` |
| `mindvision` | **迈德威视（MindVision）GigE / USB3** | 厂商驱动 + SDK，`MVSDK_PATH` 或节点参数「迈德威视 SDK 目录」指向 `mvsdk.py` 所在目录 |
| `genicam` | 任何符合 GenICam/GenTL 的相机 | `pip install harvesters` + 厂商的 `.cti` 驱动 |

**迈德威视和海康这类相机走的是厂商自己的协议**，不是 USB3 Vision 标准，所以 `opencv` 和
`genicam` 两条路都接不上，必须用对应的 SDK 后端。接法见下面「接一台迈德威视相机」。

### 不确定相机能不能接？先问一句

```bash
cvflow cameras            # 每条取图路线分别试一遍，告诉你谁认得这台相机
cvflow cameras --gige     # 再加上网络相机的广播搜索（几秒）
```

输出会分段列出 OpenCV（各后端能打开哪些设备号）、迈德威视、海康、GigE 的情况，
并直接告诉你相机节点该怎么填。

### 接一台迈德威视相机

1. 装厂商的**相机驱动 + SDK**，用它自带的演示软件先确认相机能出图——这一步不通的话，
   任何软件都接不上。
2. 找到 **`mvsdk.py`**（Python 封装）。**注意它在厂商的「Python Demo / SDK 开发包」里**，
   只有 DLL 的那个运行时目录（`MVCAMSDK.dll`、`MVImageProcess.dll`、`MindVision.ax` 之类）
   是不带它的。
3. 在 CVFlow 里加一个**相机节点**，参数这样填：
   * 取图方式 `mindvision`
   * 相机来源：留空取第一台；多台时填**序列号**、**友好名**或**索引**（`0`、`1`…）
   * 高级参数「**迈德威视 SDK 目录**」：填 `mvsdk.py` 所在的目录（或设环境变量 `MVSDK_PATH`，
     两者都没有时会去几个常见安装路径找）
   * 触发模式：`off` 自由采集 / `software` 软触发 / `hardware` 硬件触发（PLC 接线触发）
4. 点节点上的**触发图标**取一帧，确认出图。

**用演示软件占着相机的话 CVFlow 打不开它** —— 这类 SDK 一般是独占的，先把厂商软件关掉。

**拿不到 `mvsdk.py` 时还有一条路**：迈德威视的驱动会注册一个 DirectShow 滤镜
（SDK 目录里的 `MindVision.ax`）。这种情况下用 `opencv` 取图方式也可能直接能用——
把**采集后端**设成 `dshow`（Windows 默认的 MSMF 只认 UVC 设备，看不见厂商滤镜），
相机来源填设备号（`cvflow cameras` 会告诉你哪些号能打开）。
这条路拿不到曝光/增益/硬触发的精细控制，但先把图取进来调流程够用了。

## PLC 联动

通信面板分七页：**设备 / 数据点 / 触发 / 发送 / 解析·格式 / 握手 / 调试**。界面线程不做任何阻塞
通信，收发与轮询都在通信层自己的后台线程里，面板按 250 ms 批量刷新，所以高频通信不会卡界面。

> **第一次配通信，直接看 [docs/comm-manual.md](docs/comm-manual.md)**：从零配通的逐步操作、
> 分帧与字节序怎么定、握手时序图、故障对照表。下面是功能概览。

### 在界面里配一条通信（点哪里）

通信面板在主窗口**下方的「通信」页签**里。从零配一条「上位机发请求 → 视觉检测 → 回结果」：

1. **设备** 页 → `添加` → 选「TCP 服务端」→ 监听 `0.0.0.0`、端口 `6000` → 「报文分帧」页确认分隔符是 `\n` → 确定。
2. **解析/格式** 页左边「解析规则」→ `添加` → 名称 `请求`，方式「按分隔符切分」，字段表里写
   `cmd`（第 0 段）、`request_id`（第 1 段，整数）→ 确定。
3. 同一页右边「格式化规则」→ `添加` → 名称 `结果`，格式「文本」，字段表里挑 `literal:RESULT`、
   `request_id`、`status`、`out.<你发布的名字>` → 确定。
4. **触发** 页 → `添加` → 设备选刚建的那台，条件「前缀匹配」`TRIGGER`，解析规则选 `请求`，
   动作「触发流程」→ 选流程，流程忙时选「拒绝并回复忙」→ 确定。
5. **握手** 页 → `添加` → 设备与流程选好，模式「文本报文」，结果回复选 `结果` → 确定。
6. **设备** 页 → `检查配置`（引用有问题会在这里一次性列出来）→ 工具栏点 **进入运行模式**。
7. **调试** 页看收发：手动发一条 `TRIGGER,1001` 试试，或者让对端发。

寄存器类设备（Modbus/MC/S7）多一步：建完设备到 **数据点** 页给地址起名字（`trigger`、
`request_id`、`ready`、`busy`、`done`…），之后触发规则和握手都从下拉框里选这些名字，不用再记地址。
对话框里会显示**参考地址**（40001 那一套）和**字节序的实际字节示例**，拿对端读到的字节一比就知道选哪个。

流程里要主动收发时，节点库的「通信」分类里有 7 个节点；它们的**设备、数据点、规则参数都是下拉
候选**（显示名字、存稳定 id），所以改设备名不会让流程失去引用，也不用手敲名称。

### 实际支持的协议与角色

| 设备类型 | 角色 | 状态 |
|---|---|---|
| TCP 客户端 | 主动连对端 | 可用 |
| TCP 服务端 | 监听，多客户端，**可按客户端定向发送 / 回复请求来源 / 广播** | 可用 |
| UDP | 保留来源地址，可回复来源 | 可用 |
| 串口 RS-232/485 | 端口枚举、文本与二进制 | 可用（本机只有 `/dev/ttyS0`，**没有真机验证**） |
| Modbus TCP 主站 | 主动读写对端 | 可用（与 pymodbus 双向互通测试） |
| Modbus TCP 从站 | 对端读写本机，四个数据区独立（0x/1x/3x/4x），功能码 01/02/03/04/05/06/15/16 | 可用（与 pymodbus 双向互通测试） |
| Modbus RTU 主站 | 串口，CRC 校验、帧长推导、请求串行化 | 可用（Linux 伪终端上与手工拼帧的从站互通验证），**没有真串口/真 PLC 验证** |
| 三菱 MC（3E 二进制） | 主站 | 只在模拟器上验证过 |
| 西门子 S7 | 主站（DB 块交换区） | 只在 snap7 服务器上验证过 |

还**没有**实现、因此不会出现在设备类型列表里的：Modbus RTU 从站、Modbus ASCII、
罗克韦尔 EtherNet/IP、欧姆龙 FINS、OPC UA、PROFINET。普通 TCP 不会被标成任何工业协议。

### 报文分帧（TCP 与串口）

TCP 是字节流：一次 `recv` 可能只有半条报文，也可能有好几条。四种分帧方式：

| 方式 | 配置 |
|---|---|
| 分隔符 | `\n` / `\r\n` / `\x03`，任意转义写法 |
| 定长 | 每 N 字节一条 |
| 长度前缀 | 长度字段的偏移、字节数（1/2/4）、字节序、**算的是哪一段**（仅数据区 / 整条报文 / 长度字段之后）、报文头字节数、是否把报文头交给上层 |
| 原始字节 | 一次 `recv` 一条，**不保证报文完整性**，界面上也这么写 |

另有三道保护：单条上限、接收缓冲上限、不完整报文超时。文本编码支持 UTF-8 / ASCII / GBK /
GB18030 / Latin-1，解不开时调试页自动按十六进制显示。UDP 不分帧——一个数据报就是一条报文。

### 数据点表（Modbus / MC / S7）

给地址起名字：**流程、触发规则和检测握手都按名字引用，改地址不用改流程**。

每个数据点可配：数据区（线圈 0x / 离散输入 1x / 输入寄存器 3x / 保持寄存器 4x）、协议地址、
数据类型（Bool / Int16 / UInt16 / Int32 / UInt32 / Float32 / 定长字符串）、数量、读写方向、
轮询周期、缩放系数、**字节序与字序**、绑定的全局变量。

* 内部一律用**协议地址**（从 0 开始），界面同时显示 40001 那一套**参考地址**，换算只有一处实现。
* 字节序用四字母布局表达：`ABCD` 不交换、`CDAB` **寄存器间字交换**、`BADC` **寄存器内字节交换**、
  `DCBA` 两者都交换。编辑对话框里显示**实际字节示例**（`CDAB` → `0x12345678` 发出去是 `56 78 12 34`）。
* 地址、数量、方向、越界都在保存时校验；离散输入与输入寄存器按协议本来就没有写功能码，
  想把整张表锁成只读还有一个 `allow_remote_write` 开关。
* 轮询与流程里的主动读写**共用同一把传输锁**，不会出现"拿到上一次请求的应答"，RTU 半双工也不会撞车。
* 断线后数据点标记为**过期**，流程里读到过期值会明确报错（或按配置输出 `valid=False`）。

### 检测握手

把一台设备和一条流程绑成完整闭环：**外部请求 → 接受任务 → 执行 → 发布结果 → 对端确认**。
信号包括 Ready / Busy / Done / OK-NG / ErrorCode / 请求编号 / 完成编号 / 确认位。三条硬保证：

1. **先结果、后 Done**：发送规则写完测量值，才写完成标志与完成编号。
2. **结果绑定请求**：接受任务的瞬间冻结请求编号、输入参数和**来源对端**，TCP 服务端因此总能回给
   正确的那个客户端；完成编号让对端确认结果属于哪一次请求。
3. **断线不重放**：记下连接代号，断线重连后那次旧任务的结果直接丢弃。

Ready 反映**真实能力**（设备已连接 + 流程在运行 + 不忙 + 不在等确认）。忙时可选拒绝 / 丢弃 /
有界排队，队列满了返回明确状态。重复请求可选重发上次结果 / 忽略 / 当成新请求。超时**只上报
错误码，不重发触发或写入**——那可能让对端执行两次物理动作。"发送成功"与"对端已确认"在
握手状态里分开记录。

错误码：`0` 正常、`1` 忙、`2` 参数错误、`3` 流程异常、`4` 超时、`5` 流程未运行、`6` 重复请求、
`7` 内部错误、`8` 判定 NG（可选）。

老方案里的简版握手（`busy_address` + `trigger_reset`）继续支持，不用改。

### 调试

「调试」页有：手动发送（文本或十六进制，可指定对端）、Modbus 手动读写与持续监视、
收发日志（时间 / 方向 / 设备 / 对端 / 字节数 / 文本 / 十六进制 / 说明），可按方向与关键字筛选、
暂停滚动、停止记录、导出 CSV。日志有容量上限，满了丢最旧的并显示丢弃条数。

每个设备都有**测试连接**：TCP 报连接耗时，Modbus/MC/S7 真读一个数据点并报告值与耗时，
串口先枚举本机端口再试开，服务端类报正在监听几个客户端。

* 结果面板里的文字可以选中复制：Ctrl+C 复制选中行，右键可复制单个单元格、整行或全部结果，内容过长时显示会截断但悬停和复制都是完整的。报错信息就显示在"值"一列。

## 没有相机和 PLC 时怎么测试

仓库自带两个模拟器，每条链路都是"打开方案 → 进入运行模式 → 运行模拟器"三步：

| 想测什么 | 第一步 | 第二步 |
|---|---|---|
| 相机搜索 / 连接测试 / 强制 IP | `python examples/sim_camera.py` | 相机 → 相机管理 → 搜索相机，或"按 IP 添加"填 127.0.0.1 |
| 相机取图 | 相机节点类型选 `folder`，来源填 `examples/images` | 运行一次 |
| TCP 报文触发与回复 | `cvflow gui examples/solutions/demo_holes.json`，进入运行模式 | `python examples/sim_plc.py tcp` |
| Modbus TCP（PLC 做主站） | `cvflow gui examples/solutions/demo_modbus.json`，进入运行模式 | `python examples/sim_plc.py modbus` |
| **完整握手：TCP 请求—应答** | `cvflow serve examples/solutions/demo_tcp_handshake.json` | `python examples/sim_handshake.py tcp` |
| **完整握手：cvflow 做 Modbus 主站** | `python examples/sim_handshake.py slave`（先起从站） | `cvflow serve examples/solutions/demo_modbus_master.json` |
| **完整握手：cvflow 做 Modbus 从站** | `cvflow serve examples/solutions/demo_modbus_slave.json` | `python examples/sim_handshake.py master` |
| 三菱 MC 协议 | `python examples/sim_plc.py mc` | `cvflow gui examples/solutions/demo_mc.json`，进入运行模式 |
| 西门子 S7 | `python examples/sim_plc.py s7` | `cvflow gui examples/solutions/demo_s7.json`，进入运行模式 |

PLC 模拟器会周期性地触发一次检测，并打印触发字是否被复位、忙标志是否出现、结果寄存器的值和心跳计数，对应软件里通信面板「调试」页和结果页。模拟相机只实现发现协议，不出图，所以取图用文件夹模拟。

后三行是**完整业务闭环**的示例：请求编号、Ready/Busy/Done、结果绑定请求、对端确认、忙时与
超时分支都走一遍。报文格式、寄存器映射与逐步说明见 [docs/comm-examples.md](docs/comm-examples.md)。

## 深度学习节点的多输入合批

三个 ONNX 节点都支持多个图像输入。把输入个数设为 N，节点会多出 `image2` 到 `imageN` 这些输入端口，以及每一路各自的输出端口，比如 `count`、`count2`、`count3`。一次运行里，所有已连接输入的图像会被拼成一个批次做一次推理，再按来源拆回各自的端口，所以结果能分清来自哪个输入。

| 参数 | 作用 | 默认 |
|---|---|---|
| 输入个数 | 图像输入端口的数量，最多 8 | 1 |
| 输入到达方式 | `sync` 整条流程一次触发、所有输入同时到达；`async` 每路相机各自被触发、在本节点汇合 | sync |
| 批次上限 | 一个批次最多几张图 | 8 |
| 合批等待（毫秒） | 等本节点其它输入到达的窗口，**只在 `async` 下生效**（`sync` 下所有输入同时到达，等待只会白白增加节拍） | 0 |
| 推理后端 | `auto` 按 TensorRT → CUDA → CPU 挑第一个能用的，也可以强制指定 | auto |
| 显卡编号 | 多卡时用哪一块，CPU 后端忽略 | 0 |
| 推理会话（高级） | 权重加载几份：`shared` 全软件一份；`exclusive` 按权重共享组/节点各一份；`auto` 在 CPU 上共享、GPU 上各一份 | auto |
| 权重共享组（高级） | 填相同名字的节点共用同一份权重（省显存，实测代价 0~6%）。**只影响权重，不影响排队** | 空 |
| 叠加层来自 | 图像窗口上画哪一路的结果 | 1 |

**合批只在一个节点自己的多路输入之间进行，跨节点不合批。** 跨节点合批要求几个节点共用一个
队列，于是互相排队、还要陪着等待窗口，实测比各自一个队列慢 6%（吃满 GPU 的模型）到 27%
（吃不满的小模型）；跨节点真正值得共享的是**权重**，那个由「权重共享组」单独决定，代价只有
0~6%。自己量一遍：`python tools/bench_sharing.py 你的模型.onnx --nodes 4`。

选中多路输入的节点时，图像窗口上方会出现一个选择框，可以切换查看哪一路输入的图像，以及该节点的输出图像。每一路的检测框、文字等叠加层都按来源标记，切到哪一路就只显示那一路的结果。

### 在 GPU 上跑

按[安装](#安装)里的 `.[dev,gpu]` 装好（`.[gpu]` 与 `.[cpu]` 二选一，装错了怎么救见 [docs/install.md](docs/install.md)），默认的 `auto` 就会优先用 GPU，不需要改任何参数。先确认包里有哪些后端：

```bash
cvflow gpu                 # 自检：装了哪些后端、能不能加载、有哪些显卡
cvflow gpu 你的模型.onnx    # 再加一步实测：这个模型实际跑在哪、加载后占多少显存
```

典型输出（显存那一行是最直接的证据——跑在 CPU 上时它不会变）：

```
  CUDAExecutionProvider        可用
  TensorrtExecutionProvider    加载失败：libnvinfer.so.10: cannot open shared object file
  实际使用的后端：CUDAExecutionProvider（0 号卡）
  本进程显存：0 → 942 MiB（+942）
```

`auto` 会自动跳过加载不了的后端：装了 GPU 版但没装 TensorRT 时，日志里会看到 `推理后端 TensorrtExecutionProvider 的运行库加载不了（libnvinfer.so.10: ...），本次运行跳过它`，然后继续用 CUDA。这一步很关键——onnxruntime 只要收到一个加载失败的后端就会把整个会话退回 CPU，连本来能用的 CUDA 一起丢掉。

列表里没有 `CUDAExecutionProvider`，说明装的是 CPU 版（或者 CPU 版把 GPU 版覆盖了）：`pip uninstall -y onnxruntime onnxruntime-gpu` 之后重新装 `.[gpu]`。列表里有它、运行时却仍然回退到 CPU，就是运行库加载失败，日志会写明缺的是哪个库，比如：

```
推理后端 CUDAExecutionProvider 无法加载（libcublasLt.so.12: cannot open shared object file...）；
缺 CUDA 运行库，可装 pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12，本次运行不再尝试
````TensorrtExecutionProvider` 还要另外安装 TensorRT 本体，一般用 `cuda` 或 `auto` 就够。

**想确认实际跑在哪里，看日志里的这一行**：

```
ONNX 检测 (YOLO)：模型 best.onnx 使用推理后端 CUDAExecutionProvider（0 号卡）
```

显示 `CPUExecutionProvider` 就说明 GPU 后端没加载起来（最常见的原因是缺 CUDA 运行库），日志里会另有一条警告说明原因。多卡机器用**显卡编号**把不同节点分到不同卡上，这是 GPU 上最确定有效的并行方式。

### 权重共享和合批共享是两件事

| | 按什么共享 | 共享意味着 |
|---|---|---|
| **推理会话**（权重） | 模型文件 + 修改时间 + 后端 + 显卡编号 + 会话归属 | 内存/显存里只有一份权重 |
| **批处理器**（队列） | 上面这些 + 预处理参数 + 批次上限 + 合批等待 + 批处理归属 | 这些节点会凑在一起合批，并且互相排队 |

会话归属由**推理会话**参数决定，默认 `auto`：

* **CPU 上共享一份**。共用会话不会让推理互相阻塞（onnxruntime 的 `Run` 本身线程安全，实测两个线程跑同一个会话能到 2.01× 加速），限制并行度的是核心数而不是会话，所以多加载一份权重只是白占内存。
* **GPU 上按合批分组/节点各加载一份**。同一个会话在 GPU 上对应一条 CUDA 流，共用就只能排队；各自一份权重才有机会用各自的流并行——代价是**显存成倍**。显存不够时会自动退回共享会话并在日志里说明，不会让流程挂掉。
* 想省显存就把这个参数改成 `shared`，想在 CPU 上也各加一份就改成 `exclusive`。

要提醒的是：GPU 上"各加载一份"只有在**单次推理占不满 GPU** 时才换得来吞吐。单路推理时用 `nvidia-smi dmon -s u` 看 sm 利用率，稳定在 95% 以上就别指望并行了，老老实实合批。

**合批等待只对跨流程有意义。** 一个节点的多路输入是一次提交进去的，同一个流程里的节点又是顺序执行的，所以没填合批分组时队列里不可能出现第二个提交者——这种情况下等待窗口直接按 0 处理，不会白白增加节拍。想验证合批是否生效，看 `batch_size` 输出；想对比合批的收益，改的是批次上限而不是等待时间。

### 多路图像不同时到达怎么办

典型场景：几台相机各自被 PLC 触发，图像不是同一时刻来的。做法是**一个深度学习节点、多路输入**，
把它的「输入到达方式」设成 **`async`**，每路相机接到自己的那个输入上，再把**合批等待**设成能接受的节拍余量：

```
相机1 ─预处理─┐
相机2 ─预处理─┤
相机3 ─预处理─┼→[深度学习节点 async]→ 判定 → 通信
相机4 ─预处理─┘   凑够批次上限 / 等待窗口到期就推理
```

* 每路相机被触发后，**只跑自己那条支路**（别的路不会被白取一帧），一直跑到深度学习节点；
* 先到的那一路成为"领队"，在窗口内等后到的；
* 凑够**批次上限**就立刻执行，不必等满窗口；
* 窗口到期就按当前已有的数量执行，**没赶上的留给下一批，谁都不会被永远卡住**；
* **只有领队那一路带着整批结果继续往下游跑**，其余各路在汇合点停住——下游（判定、存图、通信）
  因此一批只执行一次，`batch_size` 显示这一批实际有几张，没赶上的那几路输出为空。

要求各路的上游链**互不相交**（每路一个相机、一套预处理）。共用了上游节点时无法判断是哪一路被
触发，打开方案就会报错指出是哪个节点。

手工验证不用接 PLC：相机 / 图像文件 / 图像文件夹这三个源节点的**卡片标题栏里有一个触发图标**
（状态胶囊左边），点一下就只触发这一路。图标本身就是状态：空闲时是「暂停」两道竖杠，
正在触发时变成实心三角。产线上这一步由接收规则（PLC 报文）触发。

现成的例子可以直接点：

```bash
python examples/build_async_demo.py                    # 生成方案和它要用的小模型
cvflow gui examples/solutions/demo_async.json
```

点「相机1」卡片上的触发图标 → 这一路取图后在深度学习节点等着（图标变成实心三角）；
400 ms 内再点「相机2」的，两路就合成一批（`batch_size` 显示 2）。
只点一路不管它，窗口到期也会照常执行（`batch_size` 显示 1）。

等待窗口是延迟与吞吐的交换：设成 0 就是各跑各的，设大了单路延迟变高。一般取"最慢那路相机相对最快那路的到达时间差"再留一点余量。

另外要注意，合批加速的是推理本身。实测 8 路 640×640 检测，推理从 17.8 毫秒降到 7.1 毫秒，但端到端只从 55.8 毫秒降到 41.3 毫秒，因为预处理和后处理不受合批影响。节点耗时里推理占多大比例，可以用 `infer_ms` 和节点总耗时相比得出。

`sync`（默认）下一个节点的多路输入是一次提交进去的，直接成批，等待窗口按 0 处理，不会有任何额外延迟。
只有 `async` 下窗口才真的起作用。

**模型在选好文件时就加载，不是等到运行。** 在参数面板里选中模型文件（或改了推理后端、显卡编号）之后，界面会在后台线程立刻把权重加载进内存/显存，日志里会写 `模型 xxx.onnx 已就绪（加载耗时 nnn ms）`；打开方案时也会把方案里所有深度学习节点的模型预加载一遍。这样第一次运行不必再等几秒的加载时间，界面全程不卡。预加载失败不弹窗，只记日志，真正运行时还会再试一次并照常报错。

**禁用节点会把权重卸载掉。** 取消参数面板里的"启用"勾选（或在画布上右键禁用）之后，该节点占的权重会从显存/内存里放掉；重新启用时再自动加载回来。权重是跨节点共享的，只有没有别的节点还在用时才真正释放。实测一个 384 MiB 的模型：加载后本进程占 942 MiB 显存，禁用后降到 420 MiB，释放了 522 MiB——剩下的那部分是 CUDA 上下文与 cuDNN/cuBLAS 句柄，属于进程级开销，只有退出程序才会归还。

模型的批次维必须是动态的才能真正合批。导出时批次维固定为 1 的模型会自动退回逐张推理，`batch_size` 输出会显示实际的批大小，可以据此确认。

**输入尺寸会按模型自动适配。** 模型把输入形状写死时，比如只接受 512×512，节点的宽度、高度、张量布局、颜色通道会按模型自动调整，日志里会说明改了什么。形状写死的情况下别的取值本来就会被推理引擎拒绝，所以这个修正不存在歧义；形状是动态的则完全听节点参数。实在无法自动修正时，报错会同时给出模型期望的形状和实际送入的形状，并指明该调哪个参数。

注意共享省下的是权重，激活内存（中间张量）仍然随同时推理的张数增长。

## 写自己的算法节点

把一个 `.py` 放进插件目录（方案里的 `plugin_dirs`，或菜单 Plugins → Load plugin folder），定义 `Node` 子类即可，无需改平台代码：

```python
from cvflow.core import Node, Port, Param, DataType, Overlay, Rect

class MyDefectNet(Node):
    type_id = "mycompany.defect_net"      # 全局唯一
    category = "Deep Learning"
    label = "My Defect Net"
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("score", DataType.FLOAT), Port("boxes", DataType.LIST)]
    params = [Param("weights", "", "file"), Param("threshold", 0.5, "float", min=0, max=1)]

    def setup(self, ctx=None):            # 流程启动时执行一次：加载模型
        import torch; self.model = torch.load(self.get("weights"))

    def process(self, ctx, inputs):       # 每张图执行
        img = inputs["image"]             # cvflow.core.Image，.data 是 numpy BGR/灰度
        score, boxes = self.infer(img.data)
        for x, y, w, h in boxes:
            ctx.add_overlay(Overlay.rect(Rect(x, y, w, h), "#ff0000"))
        ctx.judge(score < self.get("threshold"), "defect")   # 参与整次运行的 OK/NG
        return {"score": score, "boxes": boxes}
```

要点：

* `self.state` 是会随方案保存的节点状态，适合放学习统计、计数器（见 bandit 插件）。
* `ctx.publish(name, value)` 发布的值可以在通信模板里用 `{out.name}` 引用。
* 快速试验可直接用内置的 **Python Script** 节点，在参数面板里写 `process(ctx, inputs, params, state)`。
* 内置 `ONNX Inference / Classifier / Detector(YOLO)` 节点用 onnxruntime 运行导出的模型，推理后端可选 CPU/CUDA/TensorRT。
  装了 GPU 版 onnxruntime 不等于 GPU 可用：缺少对应版本的 CUDA 或 cuDNN 运行库时后端会加载失败并回退到 CPU，日志里会写明实际使用的是哪个后端。

## 通信模型

| 概念 | 说明 |
|---|---|
| **设备** | 一条通往外部一方的链路。有稳定的 **id**，改名不破坏流程与规则里的引用；可启用/禁用、复制、随方案保存；多设备同时运行，同一设备可被多条流程引用。连接由管理器统一持有，**节点不自己建连接**。类型：`tcp_client` `tcp_server` `udp` `serial` `modbus_tcp_client`(我们是主站) `modbus_tcp_server`(我们是从站) `modbus_rtu_client` `mc` `s7` |
| **分帧** | `delimiter` / `fixed` / `length_prefix` / `raw`，另有单条上限、缓冲上限、不完整报文超时。每条连接一个独立分帧器 |
| **解析规则** | 报文 → 命名字段：`delimited`(按分隔符下标) / `regex`(分组) / `json`(点号路径) / `fixed_fields`(定长切片) / `whole`。字段可声明类型、缩放、是否必填、默认值 |
| **格式化规则** | 结果 → 报文：`text` / `json` / `binary`。字段绑定来源、小数位、缩放、宽度对齐、前后缀、结束符 |
| **数据点** | 给 Modbus/MC/S7 的地址起名字：数据区、协议地址、类型、数量、方向、轮询周期、缩放、字节序与字序、绑定变量 |
| **触发规则** | 报文：`any / startswith / equals / contains / regex`；数据点：`rising / falling / changed / equals / not_equals / greater / less / in_range`。动作：触发某流程（可带解析出来的参数与请求编号），或写入全局变量。忙时策略：拒绝 / 丢弃 / 有界排队 |
| **发送规则** | 流程结束时按条件(always/ok/ng/error)发送。三种载荷可并存，顺序是：写数据点 → 写寄存器（裸地址，老方案）→ 发报文（格式化规则或文本模板）。目标可选**回复请求来源** / 设备默认 / 广播 |
| **检测握手** | 把设备与流程绑成闭环：Ready/Busy/Done/OK-NG/ErrorCode/请求编号/完成编号/确认位，见上面的「PLC 联动」 |
| **通信节点** | `Comm Receive`（从收件箱取报文）、`Comm Send`、`Parse Message`、`Format Result`、`Read Data Point`、`Write Data Point`、`Comm Status`；采集分类里的 `Trigger Data` 被扩展成也输出请求编号与本次请求冻结的参数 |

文本模板的取值范围是固定的几个命名空间，**不做表达式求值**（模板里写不出可执行代码）：
`{status} {ok} {ng} {run_id} {flow} {duration_ms:.1f} {error} {request_id}`、
`{out.名字}`（发布结果节点）、`{var.名字}`（全局变量）、`{field.名字}`（本次请求解析出的字段）、
`{node[节点名].端口}`。格式化规则的字段来源同样只认一张固定的取值表，外加 `literal:常量`。

**后台监听是唯一读套接字的一方**：收到的报文复制进各个收件箱，`Comm Receive` 节点从自己的
收件箱取，所以节点与监听线程不会抢读同一个连接。

Modbus 两个角色都是自带实现：从站支持功能码 01/02/03/04/05/06/15/16，四个数据区互相独立，
对端的每次写入立即被观察到、不需要轮询；主站自己掌握传输层，轮询与主动读写排在同一条队列上。
pymodbus 现在只作为**测试对端**（第三方独立实现）使用，不是运行时依赖。

## 目录

```
cvflow/core        类型、节点模型、注册表/插件加载、图、执行引擎、运行时、事件、变量、叠加层渲染
cvflow/operators   内置节点（source/preprocess/analysis/dl/logic/output/comm）
cvflow/camera      相机抽象：folder 模拟、OpenCV、GenICam(harvesters)
cvflow/comm        通信：设备(base/tcp/serial)、分帧(framing)、解析与格式化(parsing)、
                   Modbus(modbus_client/modbus_map/modbus)、MC、S7、检测握手(handshake)、管理器(manager)
cvflow/ui          PySide6 界面（通信面板 comm_panel + 配置对话框 comm_dialogs）
cvflow/cli.py      命令行：gui / run / serve / nodes / validate / new / gpu / shortcut
examples/          示例图像生成器、示例方案、示例插件
tests/             pytest（核心、算子、通信、界面 offscreen）
tools/             辅助脚本：测试报告生成、虚拟环境清理、权重共享实测（bench_sharing.py）
packaging/         Windows 安装包：PyInstaller 配置 + Inno Setup 脚本 + 一键构建
install.ps1        Windows 一键安装（建环境 → 装对推理运行时 → 实测 CUDA）
install.sh         Linux / macOS 一键安装
docs/install.md    安装环境详解：步骤、自检、常见错误对照表
docs/comm-manual.md    通信使用说明书：从零配通的逐步操作、分帧与字节序、握手、故障对照表
docs/comm-examples.md  三个通信示例：报文格式、寄存器映射、联调步骤
```

## 节点黑盒测试报告

`docs/node-test-report.md` 是按节点归类的黑盒测试报告：契约测试自动覆盖注册表里的全部节点（元数据、输出端口与声明一致、错误输入被隔离），功能测试用真值可解析计算的合成图像逐个验证各模式与参数。重新生成：

```bash
python tools/node_test_report.py
```

`docs/ui-test-report.md` 是界面的黑盒测试报告：只通过用户可见的操作（菜单、工具栏、鼠标拖拽、键盘、对话框、表格编辑）驱动界面并检查可见结果，覆盖启动布局、文件与流程管理、节点库、节点编辑器、参数面板、运行控制、图像窗口、结果/变量/日志/通信面板、相机管理、插件与帮助。重新生成：

```bash
python tools/ui_test_report.py
```

`docs/comm-test-report.md` 是通信功能的黑盒测试报告：对端用真实套接字、Linux 伪终端串口、第三方 pymodbus 与 snap7、按协议手册手工拼出的报文做字节级核对，覆盖报文分帧（半包/粘包/多帧/畸形/超限/超时）、报文解析与结果格式化、Modbus 数据点映射（地址换算与四种字节序按**实际字节**核对）、TCP/UDP/串口/Modbus 主从站/三菱 MC/西门子 S7、触发规则与发送条件、检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离）、通信节点、统一设备管理、持久化、连接测试、三个示例方案端到端和无界面生产模式。重新生成：

```bash
python tools/comm_test_report.py
```

## 目前的边界（诚实版）

* 传统算子仅覆盖 OpenCV 能直接提供的部分：Blob、轮廓、NCC 模板匹配（仅平移）、亚像素边缘卡尺、圆查找、强度统计、二维码。形状匹配（旋转/缩放）、高精度标定、OCR 还没有。
* 执行是单线程同步 DAG，Output 类节点总是排在其它就绪节点之后执行（存图/渲染能看到最终判定）；流程间并行，流程内不并行。节拍 < 20 ms 的高速线需要把重算子放到多进程或 C++ 扩展。
* 海康 MVS SDK 封装和 GenICam 取图都还没有在真机上验证（本机没有相机）；GigE 发现、强制 IP、连接测试用模拟相机做了自动化测试。
* **通信**：TCP（客户端/服务端）、UDP、Modbus TCP 主站与从站都做了双向互通测试（对端用第三方 pymodbus）。
  Modbus RTU 只实现了**主站**：CRC 已按手册的已知值核对，读/写单个/写多个/字符串在 Linux 伪终端上
  与一个手工拼帧的 RTU 从站互通验证过，但**没有在真串口和真 PLC 上跑过**（本机只有一个 `/dev/ttyS0`）；
  RTU 从站、Modbus ASCII 还没有。
  三菱 MC 和西门子 S7 只在模拟器/snap7 服务器上验证过。欧姆龙 FINS、罗克韦尔 EtherNet/IP、OPC UA、
  PROFINET 都没有实现——它们不会出现在设备类型列表里，普通 TCP 也不会被标成工业协议。
* 通信还没有 TLS 与鉴权；断线期间的发送直接失败并计数，**没有重发队列**，恢复后不会补发。
* 还没有权限管理、运行日志落库、方案版本管理、安装包。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。

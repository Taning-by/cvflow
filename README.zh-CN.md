# CVFlow — 开放的流程式工业视觉平台

[English README](README.md)

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

需要 Python 3.10 以上，Linux / Windows / macOS 均可。

```bash
git clone https://github.com/Taning-by/cvflow.git
cd cvflow
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,cpu]"        # 核心 + 界面 + 测试依赖 + CPU 推理
```

**推理运行时要在 `cpu` 和 `gpu` 里二选一**，不能同时装：`onnxruntime` 与 `onnxruntime-gpu` 装的是同一个 Python 模块，共存时会互相覆盖，典型症状就是"装了 GPU 版却只有 CPU 后端"。用显卡就把上面那行换成：

```bash
pip install -e ".[dev,gpu]"        # onnxruntime-gpu + CUDA 12 / cuDNN 9（都从 pip 装，约 2 GB）
```

GPU 版连同 CUDA 12 与 cuDNN 9 的 pip 包一起装好，**系统不需要另外装 CUDA，也不用配 `LD_LIBRARY_PATH` / `PATH`**：这些库不在系统搜索路径里，软件会在建立推理会话前自动预加载它们。

已经装错了的话，先清干净再装：`pip uninstall -y onnxruntime onnxruntime-gpu`，然后只装需要的那一个。

其它可选依赖：`.[genicam]`（harvesters，GigE/USB3 Vision 相机）、`.[plc]`（python-snap7，西门子 S7）、`.[dl]`（PyTorch，给插件模板用）。

## 运行

```bash
cvflow nodes                                          # 列出 40 个内置节点
cvflow run examples/solutions/demo_holes.json -n 10   # 终端跑示例流程
cvflow gui examples/solutions/demo_holes.json         # 打开桌面程序
cvflow serve examples/solutions/demo_holes.json       # 无界面生产模式
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

* **搜索相机**：用 GigE Vision 发现协议在本机每个网卡上广播，列出网络上所有 GigE 相机的 IP、MAC、厂商、型号、序列号、用户名，并标出"不在本机网段"的相机。搜索不依赖任何 SDK；装了海康 MVS SDK 时会同时用 SDK 枚举（含 USB3 相机），填了 GenTL 驱动时也会用 GenICam 枚举。
* **按 IP 添加**：向指定 IP 单播发现请求，没响应也可以先手动加入。
* **连接测试**：读相机的 GigE Vision 版本寄存器，报告在线与否和耗时；海康相机还会尝试用 SDK 打开。
* **强制 IP**：相机和网卡不在同一网段时，按 MAC 给相机下发临时 IP。
* **添加到流程**：生成一个"相机"节点，取图方式自动选择海康 MVS（`hik`）或 GenICam（`genicam`）。

相机节点参数：触发模式（沿用 / 自由采集 / 软触发 / 硬件触发）、曝光（微秒）、增益、超时，以及任意 GenICam 特性的 JSON。海康 MVS SDK 的 Python 模块通过环境变量 `MVCAM_SDK_PATH` 指向 `MVS/Development/Samples/Python/MvImport` 目录；GenICam 方式需要 `pip install harvesters` 和厂商的 `.cti` 驱动。

## PLC 联动

通信面板里每个设备都有 **测试连接**（TCP 探测连接耗时、Modbus/MC/S7 读一个寄存器并报告值与耗时、串口试开）。支持的设备：

| 类型 | 说明 |
|---|---|
| TCP 客户端 / 服务端、UDP、串口 | 文本报文；可配置心跳报文与间隔 |
| Modbus TCP 主站 / 从站 | 主站轮询 PLC 寄存器；从站自带实现，PLC 直接读写本机寄存器 |
| 三菱 MC 协议（3E 二进制） | 三菱 Q/L/iQ-R/FX5 及汇川、基恩士等兼容 PLC，支持 D/W/R 字与 M/X/Y 位软元件，地址写 `D100`、`M20`；附带 `McSimulatorServer` 模拟器 |
| 西门子 S7 | 基于 python-snap7，以 DB 块为交换区，地址为字节偏移；S7-1200/1500 需开启 PUT/GET 并关闭 DB 优化访问 |

寄存器类设备都支持同一套握手：

* **触发**：接收规则 `register_rising` 监视触发字的 0→非 0 上升沿。
* **忙标志** `busy_address`：触发后写 1，流程结束并写完结果后写 0，PLC 据此等待结果。
* **触发自动复位** `trigger_reset`：触发后由视觉把触发字清零，下一次 PLC 写 1 即可再次产生上升沿。
* **心跳** `heartbeat_address` + `heartbeat_s`：按间隔自增，PLC 看到数值变化就知道视觉在线；文本设备则按间隔发送心跳报文。
* **结果**：发送规则里的寄存器写入列表，`kind` 支持 int16/uint16/int32/uint32/float32/bool。

* 结果面板里的文字可以选中复制：Ctrl+C 复制选中行，右键可复制单个单元格、整行或全部结果，内容过长时显示会截断但悬停和复制都是完整的。报错信息就显示在"值"一列。

## 没有相机和 PLC 时怎么测试

仓库自带两个模拟器，每条链路都是"打开方案 → 进入运行模式 → 运行模拟器"三步：

| 想测什么 | 第一步 | 第二步 |
|---|---|---|
| 相机搜索 / 连接测试 / 强制 IP | `python examples/sim_camera.py` | 相机 → 相机管理 → 搜索相机，或"按 IP 添加"填 127.0.0.1 |
| 相机取图 | 相机节点类型选 `folder`，来源填 `examples/images` | 运行一次 |
| TCP 报文触发与回复 | `cvflow gui examples/solutions/demo_holes.json`，进入运行模式 | `python examples/sim_plc.py tcp` |
| Modbus TCP（PLC 做主站） | `cvflow gui examples/solutions/demo_modbus.json`，进入运行模式 | `python examples/sim_plc.py modbus` |
| 三菱 MC 协议 | `python examples/sim_plc.py mc` | `cvflow gui examples/solutions/demo_mc.json`，进入运行模式 |
| 西门子 S7 | `python examples/sim_plc.py s7` | `cvflow gui examples/solutions/demo_s7.json`，进入运行模式 |

PLC 模拟器会周期性地触发一次检测，并打印触发字是否被复位、忙标志是否出现、结果寄存器的值和心跳计数，对应软件里通信面板的"监视"页和结果页。模拟相机只实现发现协议，不出图，所以取图用文件夹模拟。

## 深度学习节点的多输入合批

三个 ONNX 节点都支持多个图像输入。把输入个数设为 N，节点会多出 `image2` 到 `imageN` 这些输入端口，以及每一路各自的输出端口，比如 `count`、`count2`、`count3`。一次运行里，所有已连接输入的图像会被拼成一个批次做一次推理，再按来源拆回各自的端口，所以结果能分清来自哪个输入。

| 参数 | 作用 | 默认 |
|---|---|---|
| 输入个数 | 图像输入端口的数量，最多 8 | 1 |
| 批次上限 | 一个批次最多几张图 | 8 |
| 合批等待（毫秒） | 等同组其它节点的图像一起合批的时间，**只有填了合批分组才生效** | 0 |
| 合批分组 | 留空则本节点独立推理、与其它节点互不排队；填相同名字的节点共用批处理器，可跨节点跨流程合批但会排队 | 空 |
| 推理后端 | `auto` 按 TensorRT → CUDA → CPU 挑第一个能用的，也可以强制指定 | auto |
| 显卡编号 | 多卡时用哪一块，CPU 后端忽略 | 0 |
| 推理会话（高级） | 权重加载几份：`shared` 全软件一份；`exclusive` 按分组/节点各一份；`auto` 在 CPU 上共享、GPU 上各一份 | auto |
| 叠加层来自 | 图像窗口上画哪一路的结果 | 1 |

选中多路输入的节点时，图像窗口上方会出现一个选择框，可以切换查看哪一路输入的图像，以及该节点的输出图像。每一路的检测框、文字等叠加层都按来源标记，切到哪一路就只显示那一路的结果。

### 在 GPU 上跑

按[安装](#安装)里的 `.[dev,gpu]` 装好（`.[gpu]` 与 `.[cpu]` 二选一），默认的 `auto` 就会优先用 GPU，不需要改任何参数。先确认包里有哪些后端：

```bash
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
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

典型场景：几台相机各自被 PLC 触发，图像不是同一时刻来的。做法是**每路一个流程**（接收规则 → 触发流程），每个流程里的深度学习节点填**相同的合批分组**，再把**合批等待**设成能接受的节拍余量：

* 先到的那一路成为"领队"，在窗口内等后到的；
* 凑够**批次上限**就立刻执行，不必等满窗口；
* 窗口到期就按当前已有的数量执行，**没赶上的留给下一批，谁都不会被永远卡住**；
* 每一路各自拿回自己的结果，`batch_size` 显示这一批实际有几张。

等待窗口是延迟与吞吐的交换：设成 0 就是各跑各的，设大了单路延迟变高。一般取"最慢那路相机相对最快那路的到达时间差"再留一点余量。

另外要注意，合批加速的是推理本身。实测 8 路 640×640 检测，推理从 17.8 毫秒降到 7.1 毫秒，但端到端只从 55.8 毫秒降到 41.3 毫秒，因为预处理和后处理不受合批影响。节点耗时里推理占多大比例，可以用 `infer_ms` 和节点总耗时相比得出。

同一个节点的多路输入本来就同时到达，直接成批，不受等待窗口影响。等待窗口只在跨线程时起作用：多个流程各自推理时，先到的那一路在窗口内等后到的一起合批，窗口到期就按当前已有的数量执行，不会一直等。所以单流程场景把等待时间保持为 0，不会有任何额外延迟。

**模型在选好文件时就加载，不是等到运行。** 在参数面板里选中模型文件（或改了推理后端、显卡编号）之后，界面会在后台线程立刻把权重加载进内存/显存，日志里会写 `模型 xxx.onnx 已就绪（加载耗时 nnn ms）`；打开方案时也会把方案里所有深度学习节点的模型预加载一遍。这样第一次运行不必再等几秒的加载时间，界面全程不卡。预加载失败不弹窗，只记日志，真正运行时还会再试一次并照常报错。

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
| 设备 | `tcp_client` `tcp_server` `udp` `serial` `modbus_tcp_client`(我们是主站) `modbus_tcp_server`(我们是从站) |
| 接收规则 | 文本设备：`startswith / equals / contains / regex / any`；Modbus：`register_rising`(0→非0) / `register_change` / `register_equals`。动作：触发某流程，或把报文写入全局变量 |
| 发送规则 | 流程结束时按条件(always/ok/ng/error)发送。文本模板：`{status} {ok} {ng} {run_id} {duration_ms:.1f} {out.名字} {var.名字} {node[节点名].端口}`；Modbus：寄存器写入列表 `[{"address":10,"expr":"{out.count}","kind":"int16"}]`，kind 支持 int16/uint16/int32/uint32/float32/bool |
| 节点内发送 | `Send Message`、`Modbus Write` 节点可在流程中间主动发数据 |

Modbus TCP 从站是自带实现（功能码 1/2/3/4/5/6/15/16），PLC 的每次写入都立即被观察到，不需要轮询；主站基于 pymodbus，按 `poll_ms` 轮询监视寄存器段并检测变化。

## 目录

```
cvflow/core        类型、节点模型、注册表/插件加载、图、执行引擎、运行时、事件、变量、叠加层渲染
cvflow/operators   内置节点（source/preprocess/analysis/dl/logic/output）
cvflow/camera      相机抽象：folder 模拟、OpenCV、GenICam(harvesters)
cvflow/comm        通信设备与规则管理
cvflow/ui          PySide6 界面
cvflow/cli.py      命令行：gui / run / serve / nodes / validate / new
examples/          示例图像生成器、示例方案、示例插件
tests/             pytest（核心、算子、通信、界面 offscreen）
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

`docs/comm-test-report.md` 是通信功能的黑盒测试报告：对端用真实套接字、Linux 伪终端串口、第三方 pymodbus 与 snap7、按协议手册手工拼出的报文做字节级核对，覆盖 TCP/UDP/串口/Modbus 主从站/三菱 MC/西门子 S7、接收与发送规则、模板、握手、心跳、事件、持久化、连接测试和无界面生产模式。重新生成：

```bash
python tools/comm_test_report.py
```

## 目前的边界（诚实版）

* 传统算子仅覆盖 OpenCV 能直接提供的部分：Blob、轮廓、NCC 模板匹配（仅平移）、亚像素边缘卡尺、圆查找、强度统计、二维码。形状匹配（旋转/缩放）、高精度标定、OCR 还没有。
* 执行是单线程同步 DAG，Output 类节点总是排在其它就绪节点之后执行（存图/渲染能看到最终判定）；流程间并行，流程内不并行。节拍 < 20 ms 的高速线需要把重算子放到多进程或 C++ 扩展。
* 海康 MVS SDK 封装和 GenICam 取图都还没有在真机上验证（本机没有相机）；GigE 发现、强制 IP、连接测试用模拟相机做了自动化测试。
* 三菱 MC 和西门子 S7 只在模拟器/snap7 服务器上验证过，欧姆龙 FINS、罗克韦尔 EtherNet/IP 还没有。
* 还没有权限管理、运行日志落库、方案版本管理、安装包。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。

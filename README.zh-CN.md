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
pip install -e ".[dev]"            # 核心 + 界面 + 测试依赖
```

可选依赖：`.[gpu]`（onnxruntime-gpu）、`.[genicam]`（harvesters，GigE/USB3 Vision 相机）、`.[plc]`（python-snap7，西门子 S7）、`.[dl]`（PyTorch，给插件模板用）。

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

## 界面操作

* 左侧节点库拖到画布，或双击加到视图中心；从输出端口拖到输入端口连线，Delete 删除，F 适配视图，中键/Alt+左键平移。
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
* 内置 `ONNX Inference / Classifier / Detector(YOLO)` 节点用 onnxruntime 运行导出的模型，`provider` 可选 CUDA/TensorRT。

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

## 目前的边界（诚实版）

* 传统算子仅覆盖 OpenCV 能直接提供的部分：Blob、轮廓、NCC 模板匹配（仅平移）、亚像素边缘卡尺、圆查找、强度统计、二维码。形状匹配（旋转/缩放）、高精度标定、OCR 还没有。
* 执行是单线程同步 DAG，Output 类节点总是排在其它就绪节点之后执行（存图/渲染能看到最终判定）；流程间并行，流程内不并行。节拍 < 20 ms 的高速线需要把重算子放到多进程或 C++ 扩展。
* 海康 MVS SDK 封装和 GenICam 取图都还没有在真机上验证（本机没有相机）；GigE 发现、强制 IP、连接测试用模拟相机做了自动化测试。
* 三菱 MC 和西门子 S7 只在模拟器/snap7 服务器上验证过，欧姆龙 FINS、罗克韦尔 EtherNet/IP 还没有。
* 还没有权限管理、运行日志落库、方案版本管理、安装包。

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。

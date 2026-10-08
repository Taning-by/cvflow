# 通信示例：三个完整业务闭环

三个示例共用**同一条视觉流程**（取图 → 灰度 → 阈值 → 开运算 → 数孔；两把卡尺量宽度与偏转角；
圆查找给定位坐标），发布 `holes` `x` `y` `angle` `width` 五个结果。区别只在通信这一侧：

| 示例 | 方案文件 | cvflow 的角色 | 对端 | 外部脚本 |
|---|---|---|---|---|
| A | `examples/solutions/demo_tcp_handshake.json` | TCP 服务端 | 上位机 / 机器人 | `sim_handshake.py tcp` |
| B | `examples/solutions/demo_modbus_master.json` | Modbus TCP **主站** | 模拟 PLC（从站） | `sim_handshake.py slave` |
| C | `examples/solutions/demo_modbus_slave.json` | Modbus TCP **从站** | 模拟 PLC（主站） | `sim_handshake.py master` |

重新生成方案：`python examples/build_handshake_demos.py`
自动化验收：`pytest tests/test_comm_examples.py`（示例 A 里还包含一条用 `cvflow serve` 跑的子进程用例）

> 不熟悉通信面板怎么用，先看 [comm-manual.md](comm-manual.md)：从零配通的逐步操作与故障对照表。

> **这三个示例的协议细节是 cvflow 的设计要求**，不是对任何一款商业软件的兼容声明——
> 本仓库里没有 VisionMaster 的手册或截图可以逐项对照。

---

## 示例 A：TCP 文本协议

### 报文格式

对端 → cvflow（分隔符 `\n`，UTF-8）：

```
TRIGGER,<请求编号>\n
```

cvflow → 对端（**只发给发起这次请求的那个客户端**）：

| 场景 | 报文 | 说明 |
|---|---|---|
| 正常 | `RESULT,<请求编号>,<OK\|NG\|ERROR>,<x>,<y>,<角度>\n` | 坐标与角度固定 3 位小数 |
| 忙 | `BUSY,<请求编号>\n` | 上一次检测还没跑完，本次**没有**被接受 |
| 参数错误 | `ERROR,<请求编号>,2\n` | 请求编号不是数字等；**不会**触发流程 |
| 流程未运行 | `ERROR,<请求编号>,5\n` | 还没进入运行模式 |
| 超时 | `ERROR,<请求编号>,4\n` | 流程超过 5 秒没跑完 |

错误码：`0` 正常、`1` 忙、`2` 参数错误、`3` 流程异常、`4` 超时、`5` 流程未运行、`6` 重复请求、`7` 内部错误。
流程里有节点报错时，状态字段是 `ERROR`（而不是单独发一条错误报文），错误码记在握手状态里。

方案里还预置了一条 `result_json` 格式化规则，把同样的内容发成 JSON
（`{"request_id":1001,"status":"OK","holes":3,"x":425.400,...}`）——
在「通信 → 握手」里把结果回复换成它即可。

### 配置要点（都在「通信」面板里）

| 位置 | 配置 |
|---|---|
| 设备「上位机」 | TCP 服务端，`0.0.0.0:6000`，分帧＝分隔符 `\n`，单条上限 4096 字节，不完整报文超时 2 秒 |
| 解析规则「request」 | 按 `,` 切分：第 0 段 `cmd`，第 1 段 `request_id`（整数） |
| 格式化规则「result」 | 文本，字段 `literal:RESULT` / `request_id` / `status` / `out.x` / `out.y` / `out.angle` |
| 触发规则「触发检测」 | 报文前缀匹配 `TRIGGER` → 触发流程 `main`，解析规则 `request`，忙时**拒绝** |
| 检测握手「检测握手」 | 文本模式，结果/忙/错误各绑一条格式化规则，去重窗口 3 秒（重复编号重发上次结果），流程超时 5 秒 |

### 联调步骤

```bash
# 1) 开 cvflow，打开方案，点“进入运行模式”
cvflow gui examples/solutions/demo_tcp_handshake.json
#    或者无界面：
cvflow serve examples/solutions/demo_tcp_handshake.json

# 2) 另开一个终端，跑上位机脚本（正常、参数错误、忙、重复、多客户端五段都会走一遍）
python examples/sim_handshake.py tcp
```

实际输出（本机实测）：

```
【1】正常请求：期望 RESULT,<编号>,OK,<x>,<y>,<角度>
  → TRIGGER,1001    ← RESULT,1001,OK,425.400,243.000,0.002    86 ms
  → TRIGGER,1002    ← RESULT,1002,OK,322.200,246.600,0.014    34 ms

【2】参数错误：请求编号不是数字，期望 ERROR,<编号>,2（参数错误），且不触发流程
  → TRIGGER,abc    ← ERROR,0,2    1 ms    错误码含义：参数错误

【4】重复编号：同一个编号再发一次，期望重发上次结果且不重跑流程
  ← 第一次 RESULT,1005,OK,324.600,233.400,-0.004
  ← 第二次 RESULT,1005,OK,324.600,233.400,-0.004    （一致 ✓）

【5】多客户端路由：两个客户端各自请求，结果必须回到各自那一个
  客户端1 请求 1006  ← RESULT,1006,OK,214.200,237.000,-0.006    客户端2 是否没收到：是 ✓
  客户端2 请求 1007  ← RESULT,1007,OK,313.800,237.000,-0.001    客户端1 是否没收到：是 ✓
```

**看不到 BUSY 是正常的**：示例流程只要 30~90 ms，连发两条时第一条往往已经跑完。想复现忙时分支，
把流程里的「上卡尺」换成更慢的算子，或在流程里插一个 `Delay` 节点（200 ms 以上）。

---

## 示例 B 与 C：Modbus 寄存器映射

**两个示例的寄存器映射完全一样**，区别只在谁是主站。换角色不用改流程，也不用改映射——
这正是数据点表的用处：流程与握手都按**名字**引用，地址与角色是设备侧的事。

内部一律使用**协议地址**（报文里那个数，从 0 开始）；下表同时给出 40001 那一套**参考地址**。
32 位数据占两个寄存器，布局是 **CDAB（寄存器间字交换）**——这是国内 PLC 最常见的默认，
在界面上改成 ABCD / BADC / DCBA 都只是一个下拉框。

| 数据点 | 协议地址 | 参考地址 | 类型 | 字节序 | 方向 | 轮询 | 含义 |
|---|---|---|---|---|---|---|---|
| `trigger` | 0 | 40001 | int16 | — | 读写 | 20 ms | 对端写 1 触发一次检测（上升沿） |
| `request_id` | 1–2 | 40002 | int32 | CDAB | 读写 | 20 ms | 本次请求编号，由对端写入 |
| `ready` | 3 | 40004 | int16 | — | 写 | — | 1＝可以接受新任务 |
| `busy` | 4 | 40005 | int16 | — | 写 | — | 1＝正在执行 |
| `done` | 5 | 40006 | int16 | — | 写 | — | 1＝结果已写完，可以读了 |
| `result` | 6 | 40007 | int16 | — | 写 | — | 1＝OK，2＝NG |
| `error_code` | 7 | 40008 | int16 | — | 写 | — | 见上面的错误码表 |
| `done_id` | 8–9 | 40009 | int32 | CDAB | 写 | — | 本次完成对应的请求编号 |
| `ack` | 10 | 40011 | int16 | — | 读写 | 20 ms | 对端写 1 表示已取走结果 |
| `holes` | 20 | 40021 | int16 | — | 写 | — | 孔数 |
| `x` | 21–22 | 40022 | float32 | CDAB | 写 | — | 定位 X（像素） |
| `y` | 23–24 | 40024 | float32 | CDAB | 写 | — | 定位 Y（像素） |
| `angle` | 25–26 | 40026 | float32 | CDAB | 写 | — | 偏转角（度） |
| `width` | 27–28 | 40028 | float32 | CDAB | 写 | — | 宽度（像素） |
| `heartbeat` | 30 | 40031 | int16 | — | 写 | — | 视觉在线心跳，每秒自增 |

### 握手时序

```
PLC: 等 ready=1
PLC: 写 request_id = N
PLC: 写 trigger = 1                     ← 上升沿
          │
视觉:     ├─ ready=0、busy=1、done=0、done_id=0、result=0、error_code=0
          ├─ 取图 → 算 → 判定
          ├─ 写 holes / x / y / angle / width      ← 先写结果
          └─ 写 result、error_code、done_id=N、done=1   ← 后写完成
          │
PLC: 看到 done=1 → 读结果（这时读到的一定是本次的）
PLC: 校验 done_id == N
PLC: 写 trigger = 0、ack = 1
          │
视觉:     └─ done=0、done_id=0、result=0、ready=1   ← 回到就绪
```

三条硬保证：

1. **先结果、后 Done**。发送规则写完测量值，握手才写 `done`/`done_id`。
2. **结果绑定请求**。接受任务的瞬间就冻结请求编号；`done_id` 让 PLC 能确认结果属于哪一次。
3. **断线不重放**。接受任务时记下连接代号，断线重连后那次任务的结果会被丢弃而不是写给新连接。

没确认之前 `ready` 一直是 0，这期间再来的触发会被拒绝并把 `error_code` 写成 1（忙），流程只跑一次。
`ack` 迟迟不来时，握手按 `auto_reset_s`（示例里 10 秒）自动复位，不会永久卡住。

### 示例 B 联调步骤（cvflow 做主站）

PLC 是从站，所以**先起模拟 PLC**，再让 cvflow 连过去：

```bash
# 1) 先跑模拟 PLC（Modbus 从站，监听 5020）
python examples/sim_handshake.py slave

# 2) 再开 cvflow 并进入运行模式
cvflow serve examples/solutions/demo_modbus_master.json
```

模拟 PLC 侧的实测输出：

```
→ 请求 2001：写 trigger=1，request_id=2001
  ← done=1  完成编号=2001（与请求编号一致 ✓）  判定=OK  错误码=0（正常）
     孔数=3  x=425.400  y=243.000  角度=0.002  宽度=401.00
     忙标志出现过：是 ✓  心跳=1  耗时 101 ms
  ← 确认后 done 已复位，ready=1（可以接受下一次任务）
```

### 示例 C 联调步骤（cvflow 做从站）

cvflow 是从站，所以**先开 cvflow**，再让模拟 PLC 连进来：

```bash
# 1) 先开 cvflow 并进入运行模式（监听 5020）
cvflow serve examples/solutions/demo_modbus_slave.json

# 2) 再跑模拟 PLC（Modbus 主站）
python examples/sim_handshake.py master
```

实测输出与示例 B 一致（编号从 3001 起）：

```
→ 请求 3001：写 trigger=1，request_id=3001
  ← done=1  完成编号=3001（一致 ✓）  判定=OK  错误码=0（正常）
     孔数=3  x=425.400  y=243.000  角度=0.002  宽度=401.00
     忙标志出现过：是 ✓  心跳=4  耗时 84 ms
  ← 确认后 done 已复位，ready=1（可以接受下一次任务）
```

示例 C 也可以用任意第三方主站验证（`tests/test_comm_examples.py` 里有一条用 pymodbus 当主站的用例）：

```bash
python -c "
from pymodbus.client import ModbusTcpClient
c = ModbusTcpClient('127.0.0.1', port=5020, timeout=2); c.connect()
c.write_registers(1, [0x0BB9, 0], device_id=1)   # request_id = 3001，CDAB 低字在前
c.write_register(0, 1, device_id=1)              # 触发
import time; time.sleep(0.5)
print(c.read_holding_registers(0, count=32, device_id=1).registers)
"
```

---

## 常见问题

**连不上 / 脚本一直在等。** 检查 cvflow 是否已经**进入运行模式**——编辑模式下设备不连接、流程也不接受触发。
界面上「通信 → 设备」页的状态列会显示「● 已连接」。

**Linux 上 502 端口起不来。** 502 是特权端口，需要 root。示例统一用 5020。

**看到 `error_code=2`（参数错误）。** 对端发来的报文不符合解析规则。「调试」页能看到原始报文与
错误说明；也可以在「解析/格式」页改规则。

**数值不对、差一个字节序。** 「数据点」页每一行都显示当前的字节序布局，编辑对话框里有**实际字节示例**
（例如 `CDAB` → 大端值 `0x12345678` 发出去是 `56 78 12 34`）。拿对端读到的字节和示例比一下就知道该选哪个。

**done 一直不翻转。** 看「握手」页的状态列与错误码：`流程未运行` 说明没进运行模式，
`忙` 说明上一次还没确认（`ack` 没写），`超时` 说明流程超过 `run_timeout_s` 没跑完。

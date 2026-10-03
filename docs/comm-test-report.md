# 通信功能黑盒测试报告

生成时间：2026-10-03 09:44　环境：Python 3.10.12, pymodbus 3.15.0, python-snap7 3.2.0, Linux 5.15.0-117-generic

测试方式：把通信层当作整体，只从对端观察。对端使用真实 TCP/UDP 套接字、Linux 伪终端串口、第三方的 pymodbus 主站/从站与 snap7 服务器，以及按协议手册手工拼出的报文（三菱 MC 3E 帧、Modbus 异常响应）做字节级核对；观察点是事件总线上的事件、被触发的流程与统计、写回到对端的数据、持久化文件，以及无界面生产模式的子进程。

**合计 33 个用例：通过 33，失败 0，跳过 0。**

| 功能区 | 场景 | 结果 | 耗时(s) | 说明 |
|---|---|---|---|---|
| TCP 服务端（对端：真实套接字客户端） | broadcast framing and multiple clients | ✅ 通过 | 1.1 |  |
| TCP 服务端（对端：真实套接字客户端） | terminator variants encoding and raw | ✅ 通过 | 1.2 |  |
| TCP 客户端（对端：真实套接字服务端） | connect receive send reconnect | ✅ 通过 | 0.2 |  |
| TCP 客户端（对端：真实套接字服务端） | no server and no auto reconnect | ✅ 通过 | 0.6 |  |
| UDP | datagrams both ways | ✅ 通过 | 0.5 |  |
| 串口（Linux 伪终端回环） | pty loopback | ✅ 通过 | 0.1 |  |
| Modbus TCP 从站（对端：第三方 pymodbus 主站） | function codes and exceptions | ✅ 通过 | 0.5 |  |
| Modbus TCP 从站（对端：第三方 pymodbus 主站） | unit id filter | ✅ 通过 | 1.5 |  |
| Modbus TCP 主站（对端：第三方 pymodbus 从站） | against third party server | ✅ 通过 | 0.4 |  |
| 三菱 MC 协议（字节级核对 + 模拟 PLC） | request frames match specification | ✅ 通过 | 0.0 |  |
| 三菱 MC 协议（字节级核对 + 模拟 PLC） | end code error is reported | ✅ 通过 | 0.0 |  |
| 三菱 MC 协议（字节级核对 + 模拟 PLC） | simulator end to end | ✅ 通过 | 0.5 |  |
| 西门子 S7（对端：第三方 snap7 服务器） | third party server roundtrip and trigger | ✅ 通过 | 0.1 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[startswith-TRIG-TRIGGER\n-1] | ✅ 通过 | 1.0 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[startswith-TRIG-XTRIG\n-0] | ✅ 通过 | 0.5 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[equals-GO-GO\n-1] | ✅ 通过 | 1.0 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[equals-GO-GO1\n-0] | ✅ 通过 | 0.5 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[contains-AB-xxABxx\n-1] | ✅ 通过 | 1.0 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[regex-^T\\d{2}$-T42\n-1] | ✅ 通过 | 0.5 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[regex-^T\\d{2}$-T4\n-0] | ✅ 通过 | 0.5 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[any--whatever\n-1] | ✅ 通过 | 1.0 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | text match kinds[regex-(-x\n-0] | ✅ 通过 | 0.5 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | device filter disabled and flow not running | ✅ 通过 | 1.0 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | register match kinds | ✅ 通过 | 0.5 |  |
| 接收规则（文本匹配、寄存器匹配、设备过滤、流程状态）与发送条件 | send conditions and flow filter | ✅ 通过 | 1.0 |  |
| 发送模板字段、格式与寄存器类型 | fields specs and errors | ✅ 通过 | 0.5 |  |
| 发送模板字段、格式与寄存器类型 | register kinds | ✅ 通过 | 0.5 |  |
| PLC 握手（忙标志、触发复位） | busy flag visible to plc | ✅ 通过 | 1.0 |  |
| 心跳（寄存器自增、文本报文、断开后停止） | register and text and stop | ✅ 通过 | 0.9 |  |
| 事件总线（连接/收/发/断开的事件与载荷） | sequence and payloads | ✅ 通过 | 0.6 |  |
| 配置持久化与非法配置 | roundtrip and invalid kind | ✅ 通过 | 0.0 |  |
| 连接测试（各设备类型的正反用例） | all kinds | ✅ 通过 | 0.5 |  |
| 无界面生产模式端到端（cvflow serve） | serve modbus end to end | ✅ 通过 | 0.7 |  |

## 说明

- 串口用例依赖 Linux 伪终端，在 Windows 上跳过；西门子用例依赖 python-snap7，缺少时跳过。
- 三菱 MC 的字节级核对用原始套接字扮演 PLC，直接比对客户端发出的请求帧与手册示例，不依赖项目自带的模拟器。
- Modbus 从站对单元号不符的请求按规范不应答，主站表现为超时。

重新生成：`python tools/comm_test_report.py`

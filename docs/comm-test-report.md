# 通信功能黑盒测试报告

生成时间：2026-10-08 01:54　环境：Python 3.10.12, pymodbus 3.15.0, python-snap7 3.2.0, Linux 5.15.0-117-generic

测试方式：把通信层当作整体，只从对端观察。对端使用真实 TCP/UDP 套接字、Linux 伪终端串口、第三方的 pymodbus 主站/从站与 snap7 服务器，以及按协议手册手工拼出的报文（三菱 MC 3E 帧、Modbus 异常响应）做字节级核对；观察点是事件总线上的事件、被触发的流程与统计、写回到对端的数据、持久化文件，以及无界面生产模式的子进程。分帧与字节序这类纯逻辑按**实际字节**核对，不只看往返一致（往返一致掩盖得了字节序写错）。

**合计 137 个用例：通过 137，失败 0，跳过 0。**

| 功能区 | 场景 | 结果 | 耗时(s) | 说明 |
|---|---|---|---|---|
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing delimiter handles partial glued and multiple | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing delimiter crlf and custom bytes | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing fixed length | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing length prefix semantics[payload-0] | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing length prefix semantics[total-0] | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing length prefix semantics[after field-0] | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing length prefix semantics[payload-6] | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing length prefix little endian and keep header | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing length prefix bad length resyncs and reports | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing raw mode is one recv one frame | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing max frame and buffer limits | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing incomplete timeout | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing for send adds terminator or prefix | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | framing config from legacy terminator | ✅ 通过 | 0.0 |  |
| 报文分帧（半包、粘包、多帧、畸形、超限、超时） | encoding helpers | ✅ 通过 | 0.0 |  |
| 报文解析（分隔符、正则、JSON、定长切片） | parse delimited with types and missing field | ✅ 通过 | 0.0 |  |
| 报文解析（分隔符、正则、JSON、定长切片） | parse regex named and numbered groups | ✅ 通过 | 0.0 |  |
| 报文解析（分隔符、正则、JSON、定长切片） | parse json paths and fixed fields | ✅ 通过 | 0.0 |  |
| 报文解析（分隔符、正则、JSON、定长切片） | parse bool and whole | ✅ 通过 | 0.0 |  |
| 报文解析（分隔符、正则、JSON、定长切片） | parse rule roundtrip | ✅ 通过 | 0.0 |  |
| 结果格式化（文本、JSON、二进制、取值来源受限） | format text fields with width decimals and missing | ✅ 通过 | 0.0 |  |
| 结果格式化（文本、JSON、二进制、取值来源受限） | format text template mode and json and binary | ✅ 通过 | 0.0 |  |
| 结果格式化（文本、JSON、二进制、取值来源受限） | format resolve source is restricted | ✅ 通过 | 0.0 |  |
| 结果格式化（文本、JSON、二进制、取值来源受限） | pack unpack scalars clamp | ✅ 通过 | 0.0 |  |
| 结果格式化（文本、JSON、二进制、取值来源受限） | format template | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | reference address conversion both ways | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout actual bytes for 32bit[ABCD-12 34 56 78] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout actual bytes for 32bit[BADC-34 12 78 56] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout actual bytes for 32bit[CDAB-56 78 12 34] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout actual bytes for 32bit[DCBA-78 56 34 12] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout 16bit only byte swap applies[ABCD-12 34] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout 16bit only byte swap applies[CDAB-12 34] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout 16bit only byte swap applies[BADC-34 12] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | layout 16bit only byte swap applies[DCBA-34 12] | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | float32 and string and words for | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | datapoint validation rejects bad configurations | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | datapoint scale applies both ways | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | datapoint table dedups and merges read blocks | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | read blocks respect protocol limits | ✅ 通过 | 0.0 |  |
| Modbus 数据点映射（地址换算、数据类型、字节序与字序） | register conversions | ✅ 通过 | 0.0 |  |
| TCP 服务端（对端：真实套接字客户端） | routing framing and multiple clients | ✅ 通过 | 2.2 |  |
| TCP 服务端（对端：真实套接字客户端） | terminator variants encoding and raw | ✅ 通过 | 1.2 |  |
| TCP 服务端（对端：真实套接字客户端） | tcp server trigger and reply | ✅ 通过 | 0.5 |  |
| TCP 客户端（对端：真实套接字服务端） | connect receive send reconnect | ✅ 通过 | 0.2 |  |
| TCP 客户端（对端：真实套接字服务端） | no server and no auto reconnect | ✅ 通过 | 0.6 |  |
| TCP 客户端（对端：真实套接字服务端） | tcp client reconnects and receives | ✅ 通过 | 0.1 |  |
| UDP | datagrams both ways | ✅ 通过 | 0.5 |  |
| UDP | udp loopback | ✅ 通过 | 0.5 |  |
| 串口（Linux 伪终端回环） | pty loopback | ✅ 通过 | 0.1 |  |
| Modbus TCP 从站（对端：第三方 pymodbus 主站） | function codes and exceptions | ✅ 通过 | 0.5 |  |
| Modbus TCP 从站（对端：第三方 pymodbus 主站） | unit id filter | ✅ 通过 | 1.5 |  |
| Modbus TCP 从站（对端：第三方 pymodbus 主站） | modbus server rising edge and result | ✅ 通过 | 0.5 |  |
| Modbus TCP 主站（对端：第三方 pymodbus 从站） | against third party server | ✅ 通过 | 0.4 |  |
| Modbus TCP 主站（对端：第三方 pymodbus 从站） | modbus client polls server | ✅ 通过 | 0.5 |  |
| 三菱 MC 协议（字节级核对 + 模拟 PLC） | request frames match specification | ✅ 通过 | 0.0 |  |
| 三菱 MC 协议（字节级核对 + 模拟 PLC） | end code error is reported | ✅ 通过 | 0.0 |  |
| 三菱 MC 协议（字节级核对 + 模拟 PLC） | simulator end to end | ✅ 通过 | 0.5 |  |
| 西门子 S7（对端：第三方 snap7 服务器） | third party server roundtrip and trigger | ✅ 通过 | 0.1 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[startswith-TRIG-TRIGGER\n-1] | ✅ 通过 | 1.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[startswith-TRIG-XTRIG\n-0] | ✅ 通过 | 0.5 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[equals-GO-GO\n-1] | ✅ 通过 | 1.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[equals-GO-GO1\n-0] | ✅ 通过 | 0.5 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[contains-AB-xxABxx\n-1] | ✅ 通过 | 1.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[regex-^T\\d{2}$-T42\n-1] | ✅ 通过 | 0.5 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[regex-^T\\d{2}$-T4\n-0] | ✅ 通过 | 0.5 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[any--whatever\n-1] | ✅ 通过 | 1.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | text match kinds[regex-(-x\n-0] | ✅ 通过 | 0.5 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | device filter disabled and flow not running | ✅ 通过 | 1.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | register match kinds | ✅ 通过 | 0.5 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | send conditions and flow filter | ✅ 通过 | 1.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu crc matches known values | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu expected frame length[3-\x01\x03\x04-9] | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu expected frame length[1-\x01\x01\x01-6] | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu expected frame length[6-\x01\x06-8] | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu expected frame length[16-\x01\x10-8] | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu expected frame length[131-\x01\x83-5] | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu expected frame length[3-\x01\x03--1] | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu expected frame length[64-\x01@--2] | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu pdu limits are enforced | ✅ 通过 | 0.0 |  |
| 触发规则（文本匹配、数据点条件、设备过滤、流程状态）与发送条件 | rtu client roundtrip over pty | ✅ 通过 | 4.0 |  |
| 发送模板字段、格式与寄存器类型 | fields specs and errors | ✅ 通过 | 0.5 |  |
| 发送模板字段、格式与寄存器类型 | register kinds | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | busy flag visible to plc | ✅ 通过 | 1.0 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake replies to origin client | ✅ 通过 | 1.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake bad params and flow error | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake reports flow exception | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake busy is rejected not queued | ✅ 通过 | 1.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake duplicate request resends last result | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake not running flow is reported | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake timeout reports once without retrigger | ✅ 通过 | 2.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | text handshake discards result of disconnected request | ✅ 通过 | 1.8 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | register handshake full cycle with ack | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | register handshake writes results before done | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | register trigger held high does not refire | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | register handshake rejects while busy | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | master handshake against third party slave | ✅ 通过 | 0.4 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | modbus handshake busy and trigger reset | ✅ 通过 | 1.0 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | mc protocol against simulator | ✅ 通过 | 0.5 |  |
| 检测握手（Ready/Busy/Done、请求编号、确认、忙时、超时、断线隔离） | s7 against snap7 server | ✅ 通过 | 0.2 |  |
| 心跳（寄存器自增、文本报文、断开后停止） | register and text and stop | ✅ 通过 | 0.9 |  |
| 心跳（寄存器自增、文本报文、断开后停止） | modbus and tcp heartbeats | ✅ 通过 | 0.7 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node receive takes from inbox not the socket | ✅ 通过 | 0.5 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node receive empty modes | ✅ 通过 | 0.5 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node send replies to origin peer | ✅ 通过 | 1.0 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node send hex mode and dead peer | ✅ 通过 | 0.5 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node parse uses frozen trigger fields | ✅ 通过 | 0.0 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node parse error modes | ✅ 通过 | 0.0 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node format renders current run | ✅ 通过 | 0.0 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node data point read write | ✅ 通过 | 0.5 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node read point marks stale after disconnect | ✅ 通过 | 0.5 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node trigger data exposes frozen request | ✅ 通过 | 0.0 |  |
| 通信节点（收发、解析、格式化、数据点读写、触发数据、状态） | node comm status | ✅ 通过 | 0.5 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | device ids are stable across rename | ✅ 通过 | 0.0 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | rename rewrites legacy name references | ✅ 通过 | 0.0 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | device duplicate enable disable and remove | ✅ 通过 | 0.5 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | device auto connect can be turned off | ✅ 通过 | 0.5 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | manual disconnect stops auto reconnect | ✅ 通过 | 1.0 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | references point at concrete places | ✅ 通过 | 0.0 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | validate reports dangling references | ✅ 通过 | 0.0 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | bad point config is skipped not fatal | ✅ 通过 | 0.0 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | shutdown releases ports and threads | ✅ 通过 | 0.5 |  |
| 统一设备管理（稳定 id、改名、复制、启用禁用、引用查找、资源释放） | comm log capacity pause and export | ✅ 通过 | 1.0 |  |
| 事件总线（连接/收/发/断开的事件与载荷） | sequence and payloads | ✅ 通过 | 0.6 |  |
| 配置持久化与非法配置 | roundtrip and invalid kind | ✅ 通过 | 0.0 |  |
| 配置持久化与非法配置 | persistence roundtrip with points rules and handshake | ✅ 通过 | 0.0 |  |
| 配置持久化与非法配置 | solution comm config roundtrip | ✅ 通过 | 0.0 |  |
| 连接测试（各设备类型的正反用例） | all kinds | ✅ 通过 | 0.5 |  |
| 连接测试（各设备类型的正反用例） | connection tests for text and modbus devices | ✅ 通过 | 0.5 |  |
| 示例方案端到端（TCP 握手、Modbus 主站、Modbus 从站） | example a tcp request response cycle | ✅ 通过 | 0.5 |  |
| 示例方案端到端（TCP 握手、Modbus 主站、Modbus 从站） | example a bad params and duplicate and routing | ✅ 通过 | 1.5 |  |
| 示例方案端到端（TCP 握手、Modbus 主站、Modbus 从站） | example a flow error is reported as error status | ✅ 通过 | 0.5 |  |
| 示例方案端到端（TCP 握手、Modbus 主站、Modbus 从站） | example b master polls simulated plc | ✅ 通过 | 0.5 |  |
| 示例方案端到端（TCP 握手、Modbus 主站、Modbus 从站） | example c interoperates with third party master | ✅ 通过 | 0.5 |  |
| 示例方案端到端（TCP 握手、Modbus 主站、Modbus 从站） | example c rejects second request while busy or unacked | ✅ 通过 | 0.5 |  |
| 无界面生产模式端到端（cvflow serve） | serve modbus end to end | ✅ 通过 | 0.8 |  |
| 无界面生产模式端到端（cvflow serve） | example a runs under cvflow serve | ✅ 通过 | 0.8 |  |
| 无界面生产模式端到端（cvflow serve） | example c slave serves external master | ✅ 通过 | 0.5 |  |

## 说明

- 串口用例依赖 Linux 伪终端，在 Windows 上跳过；西门子用例依赖 python-snap7，缺少时跳过。
- 三菱 MC 的字节级核对用原始套接字扮演 PLC，直接比对客户端发出的请求帧与手册示例，不依赖项目自带的模拟器。
- Modbus 从站对单元号不符的请求按规范不应答，主站表现为超时。
- 示例方案用例直接加载 `examples/solutions` 里的文件，所以它们同时验证了示例本身可用。
- Modbus RTU 主站在 Linux 伪终端上与一个手工拼帧的从站做了互通验证（读、写单个、写多个、字符串、CRC、站号不符不应答）；**真串口与真 PLC 还没有验证过**，见 README 的“目前的边界”。

重新生成：`python tools/comm_test_report.py`

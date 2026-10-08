"""检测握手的外部联调脚本：没有真实 PLC 也能把三个示例跑通。

    python examples/sim_handshake.py tcp      [--host 127.0.0.1 --port 6000]
    python examples/sim_handshake.py slave    [--port 5020]
    python examples/sim_handshake.py master   [--host 127.0.0.1 --port 5020]

| 模式 | 它扮演谁 | 配套方案 | 启动顺序 |
|---|---|---|---|
| ``tcp``    | 上位机 / 机器人（TCP 客户端） | ``demo_tcp_handshake.json`` | 先开 cvflow 进运行模式，再跑脚本 |
| ``slave``  | PLC（Modbus **从站**）        | ``demo_modbus_master.json`` | **先跑脚本**，再开 cvflow |
| ``master`` | PLC（Modbus **主站**）        | ``demo_modbus_slave.json``  | 先开 cvflow 进运行模式，再跑脚本 |

``tcp`` 模式默认把异常分支也走一遍：正常请求、参数错误、忙时请求、重复编号、两个客户端
各自请求（验证结果回到正确的那一个）。

寄存器映射与报文格式见 ``docs/comm-examples.md``。
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)

# 寄存器映射（与 build_handshake_demos.register_map 一致）
TRIGGER, REQUEST_ID, READY, BUSY, DONE, RESULT, ERROR_CODE, DONE_ID = 0, 1, 3, 4, 5, 6, 7, 8
ACK, HOLES, X, Y, ANGLE, WIDTH, HEARTBEAT = 10, 20, 21, 23, 25, 27, 30

ERROR_TEXT = {0: "正常", 1: "忙", 2: "参数错误", 3: "流程异常", 4: "超时",
              5: "流程未运行", 6: "重复请求", 7: "内部错误", 8: "判定 NG"}


def f32(words, lo_first=True) -> float:
    """CDAB 布局：低字在前。"""
    a, b = (words[1], words[0]) if lo_first else (words[0], words[1])
    return struct.unpack(">f", struct.pack(">HH", a, b))[0]


def i32(words, lo_first=True) -> int:
    a, b = (words[1], words[0]) if lo_first else (words[0], words[1])
    return struct.unpack(">i", struct.pack(">HH", a, b))[0]


def to_i32(value: int) -> list[int]:
    hi, lo = struct.unpack(">HH", struct.pack(">i", int(value)))
    return [lo, hi]      # CDAB


# =================================================================== TCP 上位机
class Upper:
    """一个 TCP 上位机连接，按行收发。"""

    def __init__(self, host: str, port: int, name: str = "上位机"):
        self.name = name
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(8)
        self._buf = b""

    def request(self, text: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        self.sock.sendall(text.encode())
        return self.line(), (time.perf_counter() - t0) * 1000

    def line(self, timeout: float = 8.0) -> str:
        self.sock.settimeout(timeout)
        try:
            while b"\n" not in self._buf:
                chunk = self.sock.recv(4096)
                if not chunk:
                    return "(对端关闭)"
                self._buf += chunk
        except socket.timeout:
            return "(无回复/超时)"
        line, _, rest = self._buf.partition(b"\n")
        self._buf = rest
        return line.decode(errors="replace")

    def quiet(self, timeout: float = 0.5) -> bool:
        self.sock.settimeout(timeout)
        try:
            return not self.sock.recv(4096)
        except socket.timeout:
            return True

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def run_tcp(a) -> int:
    print(f"连接 cvflow TCP 服务端 {a.host}:{a.port}（方案 demo_tcp_handshake.json，需先进入运行模式）")
    try:
        up = Upper(a.host, a.port)
    except OSError as e:
        print(f"连接失败：{e}\n  —— cvflow 是否已经打开 demo_tcp_handshake.json 并点了“进入运行模式”？")
        return 1
    rid = a.start_id
    print("\n【1】正常请求：期望 RESULT,<编号>,OK,<x>,<y>,<角度>")
    for _ in range(a.count):
        reply, ms = up.request(f"TRIGGER,{rid}\n")
        print(f"  → TRIGGER,{rid}    ← {reply}    {ms:.0f} ms")
        rid += 1
        time.sleep(a.interval)

    print("\n【2】参数错误：请求编号不是数字，期望 ERROR,<编号>,2（参数错误），且不触发流程")
    reply, ms = up.request("TRIGGER,abc\n")
    code = reply.split(",")[-1] if reply.startswith("ERROR") else "?"
    print(f"  → TRIGGER,abc    ← {reply}    {ms:.0f} ms"
          f"    错误码含义：{ERROR_TEXT.get(int(code), '?') if code.isdigit() else '—'}")

    print("\n【3】忙时请求：连续两条不等回复，期望第二条拿到 BUSY")
    up.sock.sendall(f"TRIGGER,{rid}\n".encode())
    up.sock.sendall(f"TRIGGER,{rid + 1}\n".encode())
    first, second = up.line(), up.line()
    print(f"  → 连发两条（{rid}、{rid + 1}）    ← {first}    ← {second}")
    print("   （流程很快时两条都可能成功——把示例流程里的“上卡尺”换成更慢的算子更容易看到 BUSY）")
    rid += 2

    print("\n【4】重复编号：同一个编号再发一次，期望重发上次结果且**不重跑流程**")
    reply1, _ = up.request(f"TRIGGER,{rid}\n")
    reply2, _ = up.request(f"TRIGGER,{rid}\n")
    same = "一致 ✓" if reply1 == reply2 else "不一致 ✗"
    print(f"  ← 第一次 {reply1}\n  ← 第二次 {reply2}    （{same}）")
    rid += 1

    print("\n【5】多客户端路由：两个客户端各自请求，结果必须回到各自那一个")
    other = Upper(a.host, a.port, "上位机2")
    time.sleep(0.2)
    r1, _ = up.request(f"TRIGGER,{rid}\n")
    quiet = other.quiet(0.6)
    r2, _ = other.request(f"TRIGGER,{rid + 1}\n")
    quiet2 = up.quiet(0.6)
    print(f"  客户端1 请求 {rid}  ← {r1}    客户端2 是否没收到：{'是 ✓' if quiet else '否 ✗'}")
    print(f"  客户端2 请求 {rid + 1}  ← {r2}    客户端1 是否没收到：{'是 ✓' if quiet2 else '否 ✗'}")
    other.close()
    up.close()
    print("\n完成。对应界面上的「调试」页可以看到每一条收发记录。")
    return 0


# =================================================================== Modbus 从站（给示例 B）
def run_slave(a) -> int:
    """扮演 PLC（Modbus 从站）：cvflow 做主站轮询我的触发位。"""
    from cvflow.comm.modbus import ModbusTcpServerDevice
    plc = ModbusTcpServerDevice("模拟PLC", {"host": a.host, "port": a.port, "unit_id": a.unit,
                                            "register_count": 8192})
    plc.connect()
    print(f"模拟 PLC（Modbus 从站）已监听 {a.host}:{plc.bound_port}，站号 {a.unit}")
    print("请打开 demo_modbus_master.json 并进入运行模式（cvflow 会主动连过来轮询）\n")
    rid = a.start_id
    try:
        while True:
            if not plc.peers:
                print("等待 cvflow 连接…")
                time.sleep(1.0)
                continue
            ready = plc.read_area("holding", READY, 1)[0]
            if ready != 1:
                time.sleep(0.05)
                continue
            # 下发一次任务
            plc.write_area("holding", REQUEST_ID, to_i32(rid))
            plc.write_area("holding", TRIGGER, [1])
            t0 = time.perf_counter()
            print(f"→ 请求 {rid}：写 trigger=1，request_id={rid}")
            busy_seen = False
            while time.perf_counter() - t0 < 6.0:
                if plc.read_area("holding", BUSY, 1)[0]:
                    busy_seen = True
                if plc.read_area("holding", DONE, 1)[0] == 1:
                    break
                time.sleep(0.01)
            regs = plc.read_area("holding", 0, 32)
            done, result, err = regs[DONE], regs[RESULT], regs[ERROR_CODE]
            done_id = i32(regs[DONE_ID:DONE_ID + 2])
            if done != 1:
                print(f"  ✗ {6.0:.0f} 秒内没有看到 done=1（错误码 {err} {ERROR_TEXT.get(err, '')}）")
            else:
                print(f"  ← done=1  完成编号={done_id}"
                      f"{'（与请求编号一致 ✓）' if done_id == rid else '（与请求编号不一致 ✗）'}"
                      f"  判定={'OK' if result == 1 else 'NG' if result == 2 else result}"
                      f"  错误码={err}（{ERROR_TEXT.get(err, '?')}）")
                print(f"     孔数={regs[HOLES]}  x={f32(regs[X:X + 2]):.3f}  y={f32(regs[Y:Y + 2]):.3f}"
                      f"  角度={f32(regs[ANGLE:ANGLE + 2]):.3f}  宽度={f32(regs[WIDTH:WIDTH + 2]):.2f}")
                print(f"     忙标志出现过：{'是 ✓' if busy_seen else '否（流程太快，一个轮询周期内就跑完了）'}"
                      f"  心跳={regs[HEARTBEAT]}  耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")
            plc.write_area("holding", TRIGGER, [0])     # 触发位复位，为下一次上升沿做准备
            plc.write_area("holding", ACK, [1])        # 确认取走结果
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < 3.0:
                # Ready 由视觉在下一个维护周期写回，所以等它一起到位再打印，免得读到中间态
                if (plc.read_area("holding", DONE, 1)[0] == 0
                        and plc.read_area("holding", READY, 1)[0] == 1):
                    print("  ← 确认后 done 已复位，ready=1（可以接受下一次任务）")
                    break
                time.sleep(0.01)
            else:
                print(f"  ✗ 确认后没有回到就绪：done={plc.read_area('holding', DONE, 1)[0]}"
                      f" ready={plc.read_area('holding', READY, 1)[0]}")
            rid += 1
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        plc.disconnect()
    return 0


# =================================================================== Modbus 主站（给示例 C）
def run_master(a) -> int:
    """扮演 PLC（Modbus 主站）：cvflow 做从站，我来写触发、读结果、写确认。"""
    from cvflow.comm.modbus_client import ModbusClient, TcpTransport
    client = ModbusClient(TcpTransport(a.host, a.port, 3.0), unit_id=a.unit, timeout=2.0)
    print(f"模拟 PLC（Modbus 主站）连接 cvflow 从站 {a.host}:{a.port}，站号 {a.unit}")
    print("请先打开 demo_modbus_slave.json 并进入运行模式\n")
    rid = a.start_id
    try:
        while True:
            try:
                ready = client.read_words("holding", READY, 1)[0]
            except Exception as e:
                print(f"读取失败：{e}（cvflow 是否已进入运行模式？）")
                time.sleep(1.5)
                continue
            if ready != 1:
                time.sleep(0.05)
                continue
            client.write_registers(REQUEST_ID, to_i32(rid))
            client.write_register(TRIGGER, 1)
            t0 = time.perf_counter()
            print(f"→ 请求 {rid}：写 trigger=1，request_id={rid}")
            busy_seen, done = False, 0
            while time.perf_counter() - t0 < 6.0:
                regs = client.read_words("holding", 0, 12)
                busy_seen = busy_seen or bool(regs[BUSY])
                if regs[DONE] == 1:
                    done = 1
                    break
                time.sleep(0.01)
            regs = client.read_words("holding", 0, 32)
            if not done:
                print(f"  ✗ 超时没看到 done=1（错误码 {regs[ERROR_CODE]} "
                      f"{ERROR_TEXT.get(regs[ERROR_CODE], '')}）")
            else:
                done_id = i32(regs[DONE_ID:DONE_ID + 2])
                print(f"  ← done=1  完成编号={done_id}"
                      f"{'（一致 ✓）' if done_id == rid else '（不一致 ✗）'}"
                      f"  判定={'OK' if regs[RESULT] == 1 else 'NG' if regs[RESULT] == 2 else regs[RESULT]}"
                      f"  错误码={regs[ERROR_CODE]}（{ERROR_TEXT.get(regs[ERROR_CODE], '?')}）")
                print(f"     孔数={regs[HOLES]}  x={f32(regs[X:X + 2]):.3f}  y={f32(regs[Y:Y + 2]):.3f}"
                      f"  角度={f32(regs[ANGLE:ANGLE + 2]):.3f}  宽度={f32(regs[WIDTH:WIDTH + 2]):.2f}")
                print(f"     忙标志出现过：{'是 ✓' if busy_seen else '否（流程太快）'}"
                      f"  心跳={regs[HEARTBEAT]}  耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")
            client.write_register(TRIGGER, 0)
            client.write_register(ACK, 1)
            t1 = time.perf_counter()
            while time.perf_counter() - t1 < 3.0:
                if (client.read_words("holding", DONE, 1)[0] == 0
                        and client.read_words("holding", READY, 1)[0] == 1):
                    print("  ← 确认后 done 已复位，ready=1（可以接受下一次任务）")
                    break
                time.sleep(0.01)
            else:
                print(f"  ✗ 确认后没有回到就绪：done={client.read_words('holding', DONE, 1)[0]}"
                      f" ready={client.read_words('holding', READY, 1)[0]}")
            rid += 1
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        client.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tcp", help="扮演 TCP 上位机（示例 A）")
    t.add_argument("--host", default="127.0.0.1")
    t.add_argument("--port", type=int, default=6000)
    t.add_argument("--count", type=int, default=3, help="正常请求发几次")
    t.add_argument("--interval", type=float, default=0.3)
    t.add_argument("--start-id", type=int, default=1001)
    t.set_defaults(fn=run_tcp)

    s = sub.add_parser("slave", help="扮演 PLC（Modbus 从站，示例 B）")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5020)
    s.add_argument("--unit", type=int, default=1)
    s.add_argument("--interval", type=float, default=1.0)
    s.add_argument("--start-id", type=int, default=2001)
    s.set_defaults(fn=run_slave)

    m = sub.add_parser("master", help="扮演 PLC（Modbus 主站，示例 C）")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=5020)
    m.add_argument("--unit", type=int, default=1)
    m.add_argument("--interval", type=float, default=1.0)
    m.add_argument("--start-id", type=int, default=3001)
    m.set_defaults(fn=run_master)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

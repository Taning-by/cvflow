"""PLC 模拟器：没有 PLC 时测试触发、握手、心跳和结果回写。

    python examples/sim_plc.py tcp     [--port 6000]   模拟用 TCP 报文触发的 PLC，周期发送 TRIG 并打印回复
    python examples/sim_plc.py modbus  [--port 5020]   模拟 Modbus 主站 PLC：写触发寄存器，读结果寄存器
    python examples/sim_plc.py mc      [--port 5000]   启动三菱 MC 协议的模拟 PLC（从站），周期置位 D0
    python examples/sim_plc.py s7      [--port 1102]   启动西门子 S7 模拟 PLC（snap7 服务器），周期置位 DB1.DBW0

配套方案（examples/solutions/）：demo_holes.json（tcp）、demo_modbus.json、demo_mc.json、demo_s7.json。
用法：先 `cvflow gui examples/solutions/demo_xxx.json`，点“进入运行模式”，再运行对应的模拟器。
mc / s7 两种要先启动模拟器再进入运行模式（它们是 PLC 侧的服务器）。
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # 实时打印每条记录


def run_tcp(a) -> int:
    print(f"连接视觉 TCP 服务端 {a.host}:{a.port}，每 {a.interval}s 发送 TRIG，Ctrl-C 退出")
    while True:
        try:
            s = socket.create_connection((a.host, a.port), timeout=5)
        except OSError as e:
            print(f"连接失败：{e}，2 秒后重试（视觉是否已进入运行模式？）")
            time.sleep(2)
            continue
        s.settimeout(5)
        f = s.makefile("r", encoding="utf-8", errors="replace", newline="\n")
        try:
            while True:
                s.sendall(b"TRIG\n")
                t0 = time.perf_counter()
                reply = f.readline().strip()
                print(f"→ TRIG   ← {reply or '(无回复)'}   {1000 * (time.perf_counter() - t0):.0f} ms")
                time.sleep(a.interval)
        except (OSError, socket.timeout) as e:
            print(f"连接断开：{e}")
        finally:
            s.close()


def run_modbus(a) -> int:
    from pymodbus.client import ModbusTcpClient
    print(f"作为 Modbus 主站连接视觉从站 {a.host}:{a.port}，每 {a.interval}s 写触发寄存器 {a.trigger}=1，"
          f"读结果寄存器 {a.result}..{a.result + a.count - 1}，Ctrl-C 退出")
    c = ModbusTcpClient(a.host, port=a.port, timeout=3)
    while True:
        if not c.connected and not c.connect():
            print("连接失败，2 秒后重试（视觉是否已进入运行模式？）")
            time.sleep(2)
            continue
        c.write_register(a.trigger, 1, device_id=1)
        t0 = time.perf_counter()
        busy_seen = False
        for _ in range(100):  # 最多等 2 秒
            rr = c.read_holding_registers(a.trigger, count=2, device_id=1)
            if not rr.isError():
                trig, busy = rr.registers[0], rr.registers[1]
                busy_seen = busy_seen or busy == 1
                if trig == 0 and busy == 0:
                    break
            time.sleep(0.02)
        rr = c.read_holding_registers(a.result, count=a.count, device_id=1)
        hb = c.read_holding_registers(a.heartbeat, count=1, device_id=1)
        regs = rr.registers if not rr.isError() else []
        print(f"触发 → 触发字已复位且忙标志清零，耗时 {1000 * (time.perf_counter() - t0):.0f} ms，"
              f"忙标志曾为1: {busy_seen}，结果寄存器 {regs}，心跳 {hb.registers[0] if not hb.isError() else '?'}")
        time.sleep(a.interval)


def run_mc(a) -> int:
    from cvflow.comm.mc import McSimulatorServer
    sim = McSimulatorServer("0.0.0.0", a.port).start()
    print(f"三菱 MC 模拟 PLC 监听 {a.port}，每 {a.interval}s 把 D0 置 1（视觉应清零 D0、翻转 D1、写 D10..），Ctrl-C 退出")
    try:
        while True:
            sim.set_word("D", 0, 1)
            t0 = time.perf_counter()
            for _ in range(100):
                if sim.get_word("D", 0) == 0 and sim.get_word("D", 1) == 0:
                    break
                time.sleep(0.02)
            print(f"D0 置 1 → 复位耗时 {1000 * (time.perf_counter() - t0):.0f} ms，"
                  f"D10..D13 = {[sim.get_word('D', 10 + i) for i in range(4)]}，心跳 D20 = {sim.get_word('D', 20)}")
            time.sleep(a.interval)
    finally:
        sim.stop()


def run_s7(a) -> int:
    import snap7
    from snap7.type import SrvArea
    db = bytearray(64)
    server = snap7.server.Server()
    server.register_area(SrvArea.DB, 1, db)
    server.start(tcp_port=a.port)
    print(f"西门子 S7 模拟 PLC 监听 {a.port}（机架 0 插槽 1，DB1 64 字节），每 {a.interval}s 把 DBW0 置 1，Ctrl-C 退出")
    try:
        while True:
            db[0:2] = b"\x00\x01"
            t0 = time.perf_counter()
            for _ in range(100):
                if db[0:2] == b"\x00\x00" and db[2:4] == b"\x00\x00":
                    break
                time.sleep(0.02)
            res = struct.unpack(">hh", bytes(db[10:14]))
            hb = struct.unpack(">H", bytes(db[20:22]))[0]
            print(f"DBW0 置 1 → 复位耗时 {1000 * (time.perf_counter() - t0):.0f} ms，DBW10/12 = {res}，心跳 DBW20 = {hb}")
            time.sleep(a.interval)
    finally:
        server.stop()
        server.destroy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tcp", "modbus", "mc", "s7"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--interval", type=float, default=2.0, help="两次触发之间的秒数")
    ap.add_argument("--trigger", type=int, default=0, help="modbus：触发寄存器地址")
    ap.add_argument("--result", type=int, default=10, help="modbus：结果寄存器起始地址")
    ap.add_argument("--count", type=int, default=4, help="modbus：读取的结果寄存器数量")
    ap.add_argument("--heartbeat", type=int, default=20, help="modbus：心跳寄存器地址")
    a = ap.parse_args()
    if a.port is None:
        a.port = {"tcp": 6000, "modbus": 5020, "mc": 5000, "s7": 1102}[a.mode]
    try:
        return {"tcp": run_tcp, "modbus": run_modbus, "mc": run_mc, "s7": run_s7}[a.mode](a)
    except KeyboardInterrupt:
        print("已退出")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

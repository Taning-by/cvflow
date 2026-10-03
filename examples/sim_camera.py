"""模拟一台 GigE Vision 相机（只实现发现协议），用于在没有相机时测试“相机管理”。

    python examples/sim_camera.py                       # 默认：IP 192.168.1.20，序列号 SIM0001
    python examples/sim_camera.py --ip 10.0.0.8 --serial ABC123 --model MV-CA050-10GM

启动后在软件里：相机 → 相机管理 → 搜索相机（广播）或“按 IP 添加”填 127.0.0.1，
再点“连接测试”“强制 IP”都会有响应。模拟相机不出图：把生成的相机节点类型改成 folder，
来源填一个图片目录即可模拟取图。

注意：本脚本要监听 UDP 3956 端口，如果本机装了相机驱动（如 MVS）占用了该端口，请先退出驱动软件。
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(line_buffering=True)  # 实时打印每条记录

from cvflow.camera.discovery import GVCP_PORT, GigEDevice, build_discovery_ack  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.20", help="模拟相机上报的 IP")
    ap.add_argument("--serial", default="SIM0001")
    ap.add_argument("--model", default="MV-CA050-10GM")
    ap.add_argument("--name", default="sim-cam", help="用户自定义名")
    ap.add_argument("--bind", default="0.0.0.0", help="监听地址，默认全部网卡")
    args = ap.parse_args()
    dev = GigEDevice(ip=args.ip, mac="C4:2F:90:12:34:56", subnet="255.255.255.0", manufacturer="Hikrobot",
                     model=args.model, version="V1.6.2", serial=args.serial, user_name=args.name)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((args.bind, GVCP_PORT))
    except OSError as e:
        print(f"无法监听 UDP {GVCP_PORT} 端口：{e}（是否有相机驱动占用？）")
        return 1
    print(f"模拟 GigE 相机运行中：{dev.model} 序列号 {dev.serial} IP {dev.ip}，监听 {args.bind}:{GVCP_PORT}，Ctrl-C 退出")
    try:
        while True:
            data, addr = s.recvfrom(1024)
            if len(data) < 8 or data[0] != 0x42:
                continue
            cmd, length, rid = struct.unpack(">HHH", data[2:8])
            if cmd == 0x0002:
                s.sendto(build_discovery_ack(dev, rid), addr)
                print(f"[{addr[0]}] 发现请求 → 已应答")
            elif cmd == 0x0080:
                (address,) = struct.unpack(">I", data[8:12])
                value = 0x00010002 if address == 0 else 0
                s.sendto(struct.pack(">HHHH", 0, 0x0081, 4, rid) + struct.pack(">I", value), addr)
                print(f"[{addr[0]}] 读寄存器 0x{address:04X} → 0x{value:08X}（连接测试）")
            elif cmd == 0x0004:
                new_ip = socket.inet_ntoa(data[8 + 20:8 + 24])
                dev.ip = new_ip
                s.sendto(struct.pack(">HHHH", 0, 0x0005, 0, rid), addr)
                print(f"[{addr[0]}] 强制 IP → 相机 IP 改为 {new_ip}")
    except KeyboardInterrupt:
        print("已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""GigE Vision 相机发现与基础控制（GVCP，UDP 3956 端口），纯 Python 实现。

与 VisionMaster 的"相机管理"一致：在本机每个网卡上广播 DISCOVERY_CMD，
列出网络上所有 GigE Vision 相机的 IP、MAC、厂商、型号、序列号、用户名；
可以对指定 IP 做连接测试（读版本寄存器），也可以"强制 IP"（FORCEIP_CMD）
把子网不匹配的相机临时改到本机网段。

取图本身交给厂商 SDK（海康 MVS）或 GenICam（harvesters）。
"""
from __future__ import annotations

import socket
import struct
import time
from dataclasses import asdict, dataclass, field

GVCP_PORT = 3956
_MAGIC = 0x42
_CMD_DISCOVERY = 0x0002
_ACK_DISCOVERY = 0x0003
_CMD_FORCEIP = 0x0004
_ACK_FORCEIP = 0x0005
_CMD_READREG = 0x0080
_ACK_READREG = 0x0081
_CMD_WRITEREG = 0x0082
_FLAG_ACK = 0x01
_FLAG_BROADCAST_ACK = 0x10


@dataclass
class GigEDevice:
    ip: str
    mac: str
    subnet: str = ""
    gateway: str = ""
    manufacturer: str = ""
    model: str = ""
    version: str = ""
    serial: str = ""
    user_name: str = ""
    interface_ip: str = ""       # 本机收到应答的网卡
    reachable: bool = True       # 相机 IP 是否与本机网卡同网段（否则需要强制 IP）
    source: str = "gvcp"         # gvcp | hik | genicam
    extra: dict = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.user_name or f"{self.model} {self.serial}".strip() or self.ip

    def to_dict(self) -> dict:
        return asdict(self)


def _cstr(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def _ip(n: int) -> str:
    return socket.inet_ntoa(struct.pack(">I", n))


def parse_discovery_ack(payload: bytes) -> GigEDevice | None:
    """解析 DISCOVERY_ACK 的 248 字节负载。"""
    if len(payload) < 248:
        return None
    mac_hi, mac_lo = struct.unpack(">HI", payload[10:16])
    mac = ":".join(f"{b:02X}" for b in struct.pack(">HI", mac_hi, mac_lo))
    ip = _ip(struct.unpack(">I", payload[36:40])[0])
    subnet = _ip(struct.unpack(">I", payload[52:56])[0])
    gateway = _ip(struct.unpack(">I", payload[68:72])[0])
    return GigEDevice(
        ip=ip, mac=mac, subnet=subnet, gateway=gateway,
        manufacturer=_cstr(payload[72:104]), model=_cstr(payload[104:136]),
        version=_cstr(payload[136:168]), serial=_cstr(payload[216:232]),
        user_name=_cstr(payload[232:248]),
        extra={"spec": f"{struct.unpack('>H', payload[0:2])[0]}.{struct.unpack('>H', payload[2:4])[0]}",
               "info": _cstr(payload[168:216])},
    )


def build_discovery_ack(dev: GigEDevice, req_id: int) -> bytes:
    """构造 DISCOVERY_ACK（用于测试/模拟相机）。"""
    p = bytearray(248)
    struct.pack_into(">HH", p, 0, 1, 2)
    mac = bytes(int(x, 16) for x in dev.mac.split(":"))
    p[10:16] = mac
    struct.pack_into(">I", p, 16, 0x80000007)
    struct.pack_into(">I", p, 20, 0x00000004)
    p[36:40] = socket.inet_aton(dev.ip)
    p[52:56] = socket.inet_aton(dev.subnet or "255.255.255.0")
    p[68:72] = socket.inet_aton(dev.gateway or "0.0.0.0")
    for off, size, text in ((72, 32, dev.manufacturer), (104, 32, dev.model), (136, 32, dev.version),
                            (168, 48, dev.extra.get("info", "")), (216, 16, dev.serial), (232, 16, dev.user_name)):
        b = text.encode("utf-8")[: size - 1]
        p[off:off + len(b)] = b
    return struct.pack(">HHHH", 0, _ACK_DISCOVERY, len(p), req_id) + bytes(p)


def local_interfaces() -> list[str]:
    """本机所有 IPv4 地址（不含回环）。"""
    ips: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    try:  # 通过"连接"一个外部地址得到默认出口网卡
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        import psutil  # 可选：更完整的网卡枚举
        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family == socket.AF_INET:
                    ips.add(a.address)
    except Exception:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def _same_subnet(a: str, b: str, mask: str) -> bool:
    try:
        ai, bi, mi = (struct.unpack(">I", socket.inet_aton(x))[0] for x in (a, b, mask))
        return (ai & mi) == (bi & mi)
    except OSError:
        return False


def discover(timeout: float = 1.0, targets: list[str] | None = None,
             interfaces: list[str] | None = None) -> list[GigEDevice]:
    """搜索相机。默认在每个网卡上广播；``targets`` 给定时改为向这些 IP 单播（用于手动按 IP 添加）。"""
    found: dict[str, GigEDevice] = {}
    req_id = int(time.time() * 1000) & 0xFFFE or 1
    cmd = struct.pack(">BBHHH", _MAGIC, _FLAG_ACK | _FLAG_BROADCAST_ACK, _CMD_DISCOVERY, 0, req_id)
    ifaces = interfaces if interfaces is not None else (local_interfaces() or ["0.0.0.0"])
    if targets:
        ifaces = ["0.0.0.0"]
    for iface in ifaces:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((iface if iface != "0.0.0.0" else "", 0))
            s.settimeout(0.2)
            dests = [(t, GVCP_PORT) for t in targets] if targets else [("255.255.255.255", GVCP_PORT)]
            for d in dests:
                try:
                    s.sendto(cmd, d)
                except OSError:
                    continue
            end = time.time() + timeout
            while time.time() < end:
                try:
                    data, addr = s.recvfrom(1024)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if len(data) < 8:
                    continue
                status, ack, length, rid = struct.unpack(">HHHH", data[:8])
                if ack != _ACK_DISCOVERY:
                    continue
                dev = parse_discovery_ack(data[8:8 + length])
                if dev is None:
                    continue
                dev.interface_ip = iface if iface != "0.0.0.0" else addr[0]
                dev.reachable = (iface == "0.0.0.0") or _same_subnet(dev.ip, iface, dev.subnet or "255.255.255.0")
                if dev.mac not in found or dev.reachable:
                    found[dev.mac] = dev
        finally:
            s.close()
    return sorted(found.values(), key=lambda d: d.ip)


def read_register(ip: str, address: int = 0x0000, timeout: float = 1.0) -> int | None:
    """GVCP READREG：读取一个 32 位寄存器，失败返回 None。地址 0 是 GigE Vision 版本寄存器。"""
    req_id = 0x1234
    cmd = struct.pack(">BBHHH", _MAGIC, _FLAG_ACK, _CMD_READREG, 4, req_id) + struct.pack(">I", address)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(timeout)
        s.sendto(cmd, (ip, GVCP_PORT))
        end = time.time() + timeout
        while time.time() < end:
            data, _ = s.recvfrom(512)
            if len(data) >= 12:
                status, ack, length, rid = struct.unpack(">HHHH", data[:8])
                if ack == _ACK_READREG and rid == req_id and status == 0:
                    return struct.unpack(">I", data[8:12])[0]
        return None
    except (socket.timeout, OSError):
        return None
    finally:
        s.close()


def test_connection(ip: str, timeout: float = 1.0) -> tuple[bool, str]:
    """连接测试：能读到版本寄存器即认为相机在线且 GVCP 可达。"""
    t0 = time.perf_counter()
    v = read_register(ip, 0x0000, timeout)
    ms = (time.perf_counter() - t0) * 1000
    if v is None:
        return False, f"{ip}：无应答（检查网线、IP 网段或相机是否被其它软件占用）"
    return True, f"{ip}：在线，GigE Vision {v >> 16}.{v & 0xFFFF}，耗时 {ms:.0f} ms"


def force_ip(mac: str, ip: str, subnet: str = "255.255.255.0", gateway: str = "0.0.0.0",
             timeout: float = 1.0) -> bool:
    """FORCEIP_CMD：按 MAC 给相机设置临时 IP（相机重启后失效，等同 VisionMaster 的“强制 IP”）。"""
    mac_b = bytes(int(x, 16) for x in mac.split(":"))
    payload = bytearray(56)
    payload[2:8] = mac_b
    payload[20:24] = socket.inet_aton(ip)
    payload[36:40] = socket.inet_aton(subnet)
    payload[52:56] = socket.inet_aton(gateway)
    req_id = 0x4321
    cmd = struct.pack(">BBHHH", _MAGIC, _FLAG_ACK | _FLAG_BROADCAST_ACK, _CMD_FORCEIP, len(payload), req_id) + bytes(payload)
    ok = False
    for iface in local_interfaces() or ["0.0.0.0"]:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.bind((iface if iface != "0.0.0.0" else "", 0))
            s.settimeout(timeout)
            s.sendto(cmd, ("255.255.255.255", GVCP_PORT))
            try:
                data, _ = s.recvfrom(512)
                if len(data) >= 8 and struct.unpack(">HHHH", data[:8])[1] == _ACK_FORCEIP:
                    ok = True
            except socket.timeout:
                pass
        except OSError:
            pass
        finally:
            s.close()
    return ok

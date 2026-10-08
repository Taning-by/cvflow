"""RS-232 / RS-485 串口设备（基于 pyserial）。

文本与二进制都走同一条路径：收到的字节先交给 ``Framer``（分隔符 / 定长 / 长度前缀 / 原始），
切出完整报文后再按需要解码文本。所以二进制协议不用另配一种设备，把分帧方式设成定长或
长度前缀即可。

两类现场故障单独处理：
* **端口被占用**：打开时报 "另一个程序正在使用"，不反复重试刷屏；
* **设备被拔掉**：读写抛异常或端口消失，标记断开，按自动重连设置决定是否重试（USB 转串口
  重新插上后会自己恢复）。
"""
from __future__ import annotations

import threading

from .base import CommDevice, CommError, Peer


def list_ports() -> list[dict[str, str]]:
    """枚举本机串口。pyserial 缺失或平台不支持时返回空列表，不抛异常。"""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    out = []
    try:
        for p in lp.comports():
            out.append({"device": p.device, "description": p.description or "",
                        "hwid": p.hwid or "", "manufacturer": getattr(p, "manufacturer", "") or ""})
    except Exception:
        return out
    return sorted(out, key=lambda d: d["device"])


def _port_hint(port: str, error: Exception) -> str:
    """打开失败时的补充提示：占用、权限，或者列出本机到底有哪些串口。"""
    text = str(error).lower()
    if "denied" in text or "busy" in text or "access is denied" in text:
        return "（端口已被另一个程序占用，或当前用户没有权限：Linux 下把用户加入 dialout 组）"
    names = [p["device"] for p in list_ports()]
    if names and port not in names:
        return f"（本机枚举到的串口是：{', '.join(names)}）"
    if not names:
        return "（本机没有枚举到任何串口）"
    return ""


class SerialDevice(CommDevice):
    kind = "serial"
    role = "peer"
    config_schema = [
        ("port", "string", "串口号", "/dev/ttyUSB0", "Windows 形如 COM3，Linux 形如 /dev/ttyUSB0"),
        ("baudrate", "int", "波特率", 9600, "常用 9600 / 19200 / 38400 / 57600 / 115200"),
        ("bytesize", "enum:5,6,7,8", "数据位", "8", ""),
        ("parity", "enum:N,E,O,M,S", "校验位", "N", "N 无校验，E 偶校验，O 奇校验"),
        ("stopbits", "enum:1,1.5,2", "停止位", "1", ""),
        ("rtscts", "bool", "RTS/CTS 硬件流控", False, ""),
        ("dsrdtr", "bool", "DSR/DTR 硬件流控", False, ""),
        ("heartbeat_text", "string", "心跳报文", "", "留空关闭"),
    ]

    def __init__(self, name, config=None, bus=None, device_id=None, enabled=True):
        super().__init__(name, config, bus, device_id, enabled)
        self._ser = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- 参数换算 ----
    def _open_kwargs(self) -> dict:
        return {
            "port": str(self.config["port"]),
            "baudrate": self.cfg_int("baudrate", 9600),
            "bytesize": int(float(self.config.get("bytesize", 8) or 8)),
            "parity": str(self.config.get("parity", "N") or "N")[:1].upper(),
            "stopbits": float(self.config.get("stopbits", 1) or 1),
            "rtscts": bool(self.config.get("rtscts", False)),
            "dsrdtr": bool(self.config.get("dsrdtr", False)),
            "timeout": 0.2,
            "write_timeout": self.connect_timeout,
        }

    def peer_text(self) -> str:
        cfg = self.config
        return (f"{cfg.get('port')} {cfg.get('baudrate')},{cfg.get('bytesize')}"
                f"{cfg.get('parity')},{cfg.get('stopbits')}")

    # ---- 生命周期 ----
    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._open()
        self._thread = threading.Thread(target=self._loop, name=f"serial-{self.name}", daemon=True)
        self._thread.start()

    def _open(self) -> None:
        if self._ser is not None:
            return
        try:
            import serial
        except ImportError as e:
            raise CommError("串口支持需要 pyserial：pip install pyserial") from e
        try:
            self._ser = serial.Serial(**self._open_kwargs())
        except Exception as e:
            msg = str(e)
            if "Permission" in msg or "denied" in msg.lower():
                msg = f"{msg}（权限不足：Linux 下把用户加入 dialout 组，或改用 sudo 运行）"
            elif "could not open" in msg.lower() or "FileNotFound" in type(e).__name__:
                msg = f"{msg}（端口不存在或已被拔掉）"
            elif "Access is denied" in msg or "busy" in msg.lower():
                msg = f"{msg}（端口已被另一个程序占用）"
            self._error(f"打开 {self.config.get('port')} 失败：{msg}")
            raise CommError(msg) from e
        self.reset_framer()
        self._add_peer(str(self.config["port"]), self.peer_text())
        self._set_connected(True)

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2.0)
            self._thread = None
        self._close("已关闭")

    def _close(self, reason: str) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
        self._set_connected(False, reason)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._ser is None:
                if not self.auto_reconnect:
                    return
                try:
                    self._open()
                except CommError:
                    self._stop.wait(self.reconnect_interval)
                    continue
            peer = self.peers.get(str(self.config["port"]))
            try:
                n = self._ser.in_waiting
                data = self._ser.read(n if n else 1)
            except Exception as e:
                # 设备被拔掉 / 驱动报错：关掉句柄，按自动重连设置决定是否再开
                self._error(f"读取失败：{e}（串口是否被拔掉？）")
                self._close(str(e))
                if not self.auto_reconnect:
                    return
                self._stop.wait(self.reconnect_interval)
                continue
            if data:
                self._feed(data, peer)
            else:
                self.check_frame_timeout()

    def _send_bytes(self, data: bytes, peer: Peer | None = None) -> None:
        if not self._ser:
            raise CommError("串口未打开")
        self._ser.write(data)

    def test_connection(self) -> tuple[bool, str]:
        """真打开一次。枚举结果只作为失败时的提示——伪终端、部分虚拟串口和 pyserial 的
        ``socket://`` 这类 URL 都枚举不到，但确实能打开，不能因为不在列表里就判失败。"""
        port = str(self.config.get("port", ""))
        if self._ser is not None:
            return True, f"串口 {self.peer_text()} 已打开"
        try:
            import serial
        except ImportError:
            return False, "串口支持需要 pyserial：pip install pyserial"
        try:
            s = serial.Serial(**self._open_kwargs())
            s.close()
        except Exception as e:
            return False, f"串口 {port} 打开失败：{e}{_port_hint(port, e)}"
        return True, f"串口 {self.peer_text()} 可以打开"

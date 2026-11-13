"""提交 bug：收集软件/系统信息与最近的日志，通过 SMTP 发到开发者邮箱。

发信用的邮箱账号**不在代码仓库里**。它写在 ``bug_report.json`` 里，按下面的顺序找：

1. 环境变量 ``CVFLOW_BUG_REPORT_CONFIG`` 指向的文件
2. 打包后的程序目录（PyInstaller 的数据目录，或 exe 所在目录）
3. 源码运行时 ``cvflow/`` 包目录（已在 .gitignore 里）

格式见 ``packaging/bug_report.example.json``。找不到配置时界面上只能"另存为文件"。
"""
from __future__ import annotations

import collections
import json
import logging
import os
import platform
import smtplib
import socket
import ssl
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from . import __version__

DEFAULT_RECIPIENT = "callme@tanxinji.com"
CONFIG_NAME = "bug_report.json"
LOG_CAPACITY = 1000

log = logging.getLogger("cvflow.bug_report")


# ---------------------------------------------------------------- 最近的日志
class _RingHandler(logging.Handler):
    """把最近的日志记录留在内存里（含报错堆栈），提交 bug 时附上。"""

    def __init__(self, capacity: int = LOG_CAPACITY) -> None:
        super().__init__(logging.INFO)
        self.records: collections.deque[str] = collections.deque(maxlen=capacity)
        self._lock_ = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record)
        except Exception:                      # pragma: no cover - 格式化失败不能影响业务
            return
        with self._lock_:
            self.records.append(text)

    def text(self) -> str:
        with self._lock_:
            return "\n".join(self.records)


_ring: _RingHandler | None = None


def install_log_buffer() -> None:
    """开始记录最近的日志（重复调用无副作用）。界面启动时调用一次。"""
    global _ring
    if _ring is not None:
        return
    _ring = _RingHandler()
    logger = logging.getLogger("cvflow")
    logger.addHandler(_ring)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)


def recent_log() -> str:
    return _ring.text() if _ring is not None else ""


# ---------------------------------------------------------------- 软件和系统信息
def system_info() -> str:
    lines = [
        f"CVFlow {__version__}" + ("（安装包）" if getattr(sys, "frozen", False) else "（源码运行）"),
        f"系统：{platform.platform()}",
        f"Python：{sys.version.split()[0]}（{platform.architecture()[0]}）",
        f"时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}",
    ]
    for mod, attr in (("numpy", "__version__"), ("cv2", "__version__"), ("onnxruntime", "__version__"),
                      ("PySide6", "__version__")):
        try:
            m = __import__(mod)
            lines.append(f"{mod}：{getattr(m, attr, '?')}")
        except Exception:
            lines.append(f"{mod}：未安装")
    try:
        import onnxruntime as ort
        lines.append("推理后端：" + ", ".join(ort.get_available_providers()))
    except Exception:
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------- 发信配置
@dataclass
class SmtpConfig:
    host: str
    port: int = 465
    security: str = "ssl"                 # ssl（465）/ starttls（587）/ none
    username: str = ""
    password: str = ""                    # 邮箱的 SMTP 授权码
    sender: str = ""                      # 发件人，默认等于 username
    to: str = DEFAULT_RECIPIENT
    timeout_s: float = 20.0

    @classmethod
    def from_dict(cls, d: dict) -> "SmtpConfig":
        cfg = cls(host=str(d["host"]), port=int(d.get("port", 465)),
                  security=str(d.get("security", "ssl")).lower(),
                  username=str(d.get("username", "")), password=str(d.get("password", "")),
                  sender=str(d.get("sender", "") or d.get("username", "")),
                  to=str(d.get("to", "") or DEFAULT_RECIPIENT),
                  timeout_s=float(d.get("timeout_s", 20.0)))
        if cfg.security not in ("ssl", "starttls", "none"):
            raise ValueError(f"security 只能是 ssl / starttls / none，不是 {cfg.security!r}")
        return cfg


def config_candidates() -> list[Path]:
    out = []
    env = os.environ.get("CVFLOW_BUG_REPORT_CONFIG")
    if env:
        out.append(Path(env))
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            out.append(Path(meipass) / CONFIG_NAME)
        out.append(Path(sys.executable).resolve().parent / CONFIG_NAME)
    out.append(Path(__file__).resolve().parent / CONFIG_NAME)
    return out


def load_config() -> SmtpConfig | None:
    """读发信配置；没有配置返回 None，配置写错了抛 ValueError（界面上要让人看见）。"""
    for path in config_candidates():
        if path.is_file():
            try:
                return SmtpConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"发信配置 {path} 写得不对：{e}") from e
    return None


# ---------------------------------------------------------------- 报告本身
@dataclass
class BugReport:
    description: str
    contact: str = ""
    include_system: bool = True
    include_log: bool = True
    system: str = ""
    log_text: str = ""

    @classmethod
    def collect(cls, description: str, contact: str = "", include_system: bool = True,
                include_log: bool = True) -> "BugReport":
        return cls(description=description.strip(), contact=contact.strip(),
                   include_system=include_system, include_log=include_log,
                   system=system_info() if include_system else "",
                   log_text=recent_log() if include_log else "")

    @property
    def subject(self) -> str:
        first = next((ln.strip() for ln in self.description.splitlines() if ln.strip()), "（无描述）")
        if len(first) > 40:
            first = first[:40] + "…"
        return f"[CVFlow {__version__} bug] {first}"

    def body(self) -> str:
        parts = ["问题描述：", self.description or "（无）", ""]
        parts += ["联系方式：" + (self.contact or "（未填）"), ""]
        if self.include_system:
            parts += ["—— 软件和系统信息 ——", self.system, ""]
        if self.include_log:
            parts += ["最近的日志见附件 log.txt。"]
        return "\n".join(parts)

    def attachments(self) -> dict[str, str]:
        files = {}
        if self.include_log:
            files["log.txt"] = self.log_text or "（没有日志）"
        return files

    def to_message(self, cfg: SmtpConfig) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = self.subject
        msg["From"] = cfg.sender
        msg["To"] = cfg.to
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="cvflow")
        if "@" in self.contact:
            msg["Reply-To"] = self.contact
        msg.set_content(self.body())
        for name, data in self.attachments().items():
            msg.add_attachment(data, subtype="plain", charset="utf-8", filename=name)
        return msg

    def save_zip(self, path: str | os.PathLike) -> Path:
        """存成一个 zip（发不出去时让用户自己转交）。"""
        path = Path(path)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("report.txt", self.subject + "\n\n" + self.body())
            for name, data in self.attachments().items():
                z.writestr(name, data)
        return path


# ---------------------------------------------------------------- 连到邮件服务器
#: 系统 DNS 查不到时改问这几个公共 DNS（阿里、腾讯、114），直接用 IP 发 UDP，不依赖本机 DNS 设置
PUBLIC_DNS = ("223.5.5.5", "119.29.29.29", "114.114.114.114")
CONNECT_TIMEOUT_S = 10.0


class ConnectError(OSError):
    """几条路都连不上。消息里逐条写了每条路是怎么失败的，给人看、也给开发者排查。"""


def _system_proxy(host: str) -> str | None:
    """浏览器用的那个代理（Windows 读注册表里的"代理服务器"设置，其它系统读 *_proxy 环境变量）。

    客户电脑"能上网"却查不到邮件服务器的域名，多半就是浏览器走了公司代理、本机自己并不能直连外网。
    只认 http 代理；自动配置脚本（PAC）读不到，这种情况只能退回直连。
    """
    import ipaddress
    import urllib.request
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private or ip.is_link_local:
            return None
    except ValueError:
        if host.lower() == "localhost":
            return None
    try:
        if urllib.request.proxy_bypass(host):
            return None
        proxies = urllib.request.getproxies()
    except Exception:  # noqa: BLE001 - 注册表读不出来就当没有代理
        return None
    url = proxies.get("https") or proxies.get("http")
    if not url:
        return None
    if "://" not in url:
        url = "http://" + url
    return url if url.lower().startswith("http://") else None


def _connect_via_proxy(proxy_url: str, host: str, port: int, timeout: float) -> socket.socket:
    """经 http 代理的 CONNECT 隧道连到 host:port，返回的套接字和直连的一样用。"""
    import base64
    from urllib.parse import unquote, urlsplit
    u = urlsplit(proxy_url)
    sock = socket.create_connection((u.hostname, u.port or 8080), timeout)
    try:
        req = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n"
        if u.username:
            cred = f"{unquote(u.username)}:{unquote(u.password or '')}".encode()
            req += "Proxy-Authorization: Basic " + base64.b64encode(cred).decode() + "\r\n"
        sock.sendall((req + "\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:      # 逐字节读到头结束，不多读一个字节——后面就是邮件服务器的数据了
            ch = sock.recv(1)
            if not ch:
                raise OSError("代理没回应就断开了")
            head += ch
            if len(head) > 16384:
                raise OSError("代理的回应太长")
        status = head.split(b"\r\n", 1)[0].decode("latin-1").strip()
        parts = status.split(None, 2)
        if len(parts) < 2 or parts[1] != "200":
            hint = "（代理要求登录）" if len(parts) > 1 and parts[1] == "407" else ""
            raise OSError(f"代理拒绝转发到 {host}:{port}：{status}{hint}")
        return sock
    except BaseException:
        sock.close()
        raise


def _dns_query(host: str, server: str, timeout: float) -> list[str]:
    """向指定 DNS 服务器查 host 的 A 记录（UDP 53），返回 IP 列表。"""
    qid = os.urandom(2)
    qname = b"".join(bytes([len(p)]) + p.encode("idna") for p in host.rstrip(".").split(".")) + b"\0"
    packet = qid + b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00" + qname + b"\x00\x01\x00\x01"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        s.sendto(packet, (server, 53))
        while True:
            data, _ = s.recvfrom(4096)
            if data[:2] == qid:
                return _parse_dns_a(data)


def _parse_dns_a(data: bytes) -> list[str]:
    """从 DNS 回应里取出所有 A 记录（CNAME 之类的跳过）。"""
    def skip_name(i: int) -> int:
        while True:
            n = data[i]
            if n == 0:
                return i + 1
            if n & 0xC0 == 0xC0:              # 压缩指针，两个字节
                return i + 2
            i += 1 + n

    if len(data) < 12 or data[3] & 0x0F:      # rcode 非 0：查无此名之类
        return []
    qd, an = int.from_bytes(data[4:6], "big"), int.from_bytes(data[6:8], "big")
    i = 12
    for _ in range(qd):
        i = skip_name(i) + 4
    ips = []
    for _ in range(an):
        i = skip_name(i)
        rtype, rlen = int.from_bytes(data[i:i + 2], "big"), int.from_bytes(data[i + 8:i + 10], "big")
        i += 10
        if rtype == 1 and rlen == 4:
            ips.append(".".join(str(b) for b in data[i:i + 4]))
        i += rlen
    return ips


def _resolve_public(host: str, timeout: float = 3.0) -> list[str]:
    errors = []
    for server in PUBLIC_DNS:
        try:
            ips = _dns_query(host, server, timeout)
            if ips:
                return ips
            errors.append(f"{server} 查无结果")
        except (OSError, IndexError) as e:
            errors.append(f"{server} {e or type(e).__name__}")
    raise OSError("；".join(errors))


def open_socket(host: str, port: int, timeout: float = CONNECT_TIMEOUT_S) -> socket.socket:
    """连到邮件服务器。依次试：直连 → 系统代理 → （本机 DNS 查不到时）用公共 DNS 查到的 IP 直连。

    直连排第一：原来能发的电脑走的还是原来那条路。都不行就抛 ``ConnectError``，消息里写清每条路的
    失败原因（"直连：本机查不到地址"、"经系统代理：拒绝转发"……），一看就知道是没外网、DNS 坏了还是代理拦了。
    """
    tried = []
    dns_failed = False
    try:
        return socket.create_connection((host, port), timeout)
    except socket.gaierror as e:
        dns_failed = True
        tried.append(f"直连：本机查不到 {host} 的地址（{e}）")
    except OSError as e:
        tried.append(f"直连：{e}")
    proxy = _system_proxy(host)
    if proxy:
        try:
            sock = _connect_via_proxy(proxy, host, port, timeout)
            log.info("直连 %s 不通，经系统代理 %s 连上了", host, proxy)
            return sock
        except OSError as e:
            tried.append(f"经系统代理 {proxy}：{e}")
    else:
        tried.append("系统没有设置代理")
    if dns_failed:
        try:
            ips = _resolve_public(host)
        except OSError as e:
            tried.append(f"公共 DNS 也查不到：{e}")
            ips = []
        for ip in ips:
            try:
                sock = socket.create_connection((ip, port), timeout)
                log.info("本机 DNS 查不到 %s，改用公共 DNS 查到的 %s 连上了", host, ip)
                return sock
            except OSError as e:
                tried.append(f"公共 DNS 查到 {ip}，连接失败：{e}")
    raise ConnectError("；".join(tried))


def _open_for_smtp(host, port, timeout) -> socket.socket:
    t = float(timeout) if isinstance(timeout, (int, float)) else CONNECT_TIMEOUT_S
    sock = open_socket(host, port, min(t, CONNECT_TIMEOUT_S))
    sock.settimeout(t)
    return sock


class _Smtp(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):
        return _open_for_smtp(host, port, timeout)


class _SmtpSsl(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):
        sock = _open_for_smtp(host, port, timeout)
        # 不管走的代理还是 IP，证书都按域名校验
        return self.context.wrap_socket(sock, server_hostname=self._host)


def send(report: BugReport, cfg: SmtpConfig) -> None:
    """发出去；失败抛 OSError / smtplib.SMTPException，消息给人看。"""
    msg = report.to_message(cfg)
    ctx = ssl.create_default_context()
    if cfg.security == "ssl":
        server: smtplib.SMTP = _SmtpSsl(cfg.host, cfg.port, timeout=cfg.timeout_s, context=ctx)
    else:
        server = _Smtp(cfg.host, cfg.port, timeout=cfg.timeout_s)
    try:
        if cfg.security == "starttls":
            server.starttls(context=ctx)
        if cfg.username:
            server.login(cfg.username, cfg.password)
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:
            server.close()
    log.info("bug 报告已发送到 %s", cfg.to)

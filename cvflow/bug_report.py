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


def send(report: BugReport, cfg: SmtpConfig) -> None:
    """发出去；失败抛 OSError / smtplib.SMTPException，消息给人看。"""
    msg = report.to_message(cfg)
    ctx = ssl.create_default_context()
    if cfg.security == "ssl":
        server: smtplib.SMTP = smtplib.SMTP_SSL(cfg.host, cfg.port, timeout=cfg.timeout_s, context=ctx)
    else:
        server = smtplib.SMTP(cfg.host, cfg.port, timeout=cfg.timeout_s)
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

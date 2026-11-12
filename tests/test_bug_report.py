"""提交 bug：报告内容、发信配置、真的走一遍 SMTP。"""
from __future__ import annotations

import email
import json
import logging
import socket
import threading
import zipfile
from email import policy

import pytest

from cvflow import bug_report as br


class FakeSmtp:
    """最小的 SMTP 服务器：只够 smtplib 发一封信，收到的原文放在 ``received`` 里。"""

    def __init__(self, auth_ok: bool = True) -> None:
        self.received: list[bytes] = []
        self.auth_ok = auth_ok
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        conn, _ = self.sock.accept()
        f = conn.makefile("rwb")

        def w(line: str) -> None:
            f.write(line.encode() + b"\r\n")
            f.flush()

        w("220 fake ESMTP")
        data, buf = False, []
        for line in iter(f.readline, b""):
            if data:
                if line == b".\r\n":
                    self.received.append(b"".join(buf))
                    data, buf = False, []
                    w("250 OK")
                else:
                    buf.append(line[1:] if line.startswith(b"..") else line)
                continue
            cmd = line.decode().strip().upper()
            if cmd.startswith("EHLO"):
                w("250-fake")
                w("250 AUTH PLAIN LOGIN")
            elif cmd.startswith("AUTH"):
                w("235 ok" if self.auth_ok else "535 authentication failed")
            elif cmd.startswith("DATA"):
                data = True
                w("354 go")
            elif cmd.startswith("QUIT"):
                w("221 bye")
                break
            else:
                w("250 OK")
        conn.close()

    def config(self) -> br.SmtpConfig:
        return br.SmtpConfig(host="127.0.0.1", port=self.port, security="none",
                             username="u", password="p", sender="sender@test", timeout_s=5)


def test_report_sends_description_system_info_and_log_attachment():
    br.install_log_buffer()
    logging.getLogger("cvflow.engine").error("[脚本] 第 2 行 helper()：KeyError: 'a'")
    server = FakeSmtp()
    report = br.BugReport.collect("相机打不开\n换了网线也不行", contact="me@example.com")
    br.send(report, server.config())
    server.thread.join(5)
    msg = email.message_from_bytes(server.received[0], policy=policy.default)
    assert msg["To"] == br.DEFAULT_RECIPIENT == "callme@tanxinji.com"
    assert msg["Reply-To"] == "me@example.com"
    assert msg["Subject"].endswith("相机打不开")
    body = msg.get_body(("plain",)).get_content()
    assert "换了网线也不行" in body and "CVFlow" in body and "Python" in body
    (att,) = list(msg.iter_attachments())
    assert att.get_filename() == "log.txt" and "第 2 行 helper()" in att.get_content()


def test_unchecked_parts_are_left_out():
    report = br.BugReport.collect("只有描述", include_system=False, include_log=False)
    assert "软件和系统信息" not in report.body() and report.attachments() == {}
    assert report.system == "" and report.log_text == ""


def test_subject_uses_the_first_line_and_is_shortened():
    report = br.BugReport.collect("\n\n" + "很长" * 40 + "\n第二行")
    assert report.subject.endswith("…") and "第二行" not in report.subject


def test_wrong_password_raises_authentication_error():
    import smtplib
    server = FakeSmtp(auth_ok=False)
    with pytest.raises(smtplib.SMTPAuthenticationError):
        br.send(br.BugReport.collect("x"), server.config())


def test_config_comes_from_the_env_var_and_bad_config_is_reported(tmp_path, monkeypatch):
    path = tmp_path / "bug_report.json"
    path.write_text(json.dumps({"host": "smtp.163.com", "username": "a@163.com", "password": "code"}),
                    encoding="utf-8")
    monkeypatch.setenv("CVFLOW_BUG_REPORT_CONFIG", str(path))
    cfg = br.load_config()
    assert (cfg.host, cfg.port, cfg.security, cfg.sender, cfg.to) == \
        ("smtp.163.com", 465, "ssl", "a@163.com", "callme@tanxinji.com")
    path.write_text(json.dumps({"port": 465}), encoding="utf-8")       # 少了 host
    with pytest.raises(ValueError, match="bug_report.json"):
        br.load_config()


def test_report_can_be_saved_as_zip(tmp_path):
    report = br.BugReport.collect("存下来再转交", include_system=False)
    path = report.save_zip(tmp_path / "r.zip")
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["log.txt", "report.txt"]
        assert "存下来再转交" in z.read("report.txt").decode("utf-8")


def test_example_config_is_valid_and_real_config_is_ignored_by_git():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    example = json.loads((root / "packaging" / "bug_report.example.json").read_text(encoding="utf-8"))
    assert br.SmtpConfig.from_dict(example).to == "callme@tanxinji.com"
    assert "bug_report.json" in (root / ".gitignore").read_text(encoding="utf-8").split()

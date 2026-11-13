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


# ---------------------------------------------------------------- 连不上时的几条退路
@pytest.fixture
def no_proxy_env(monkeypatch):
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY",
              "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


class FakeProxy:
    """http 代理：收下 CONNECT 请求记在 ``requests`` 里；``reply`` 是 200 就把隧道接到 ``upstream_port``。"""

    def __init__(self, upstream_port: int = 0, reply: str = "HTTP/1.1 200 Connection established") -> None:
        self.requests: list[str] = []
        self.upstream_port, self.reply = upstream_port, reply
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.url = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        conn, _ = self.sock.accept()
        head = b""
        while b"\r\n\r\n" not in head:
            head += conn.recv(1)
        self.requests.append(head.decode())
        conn.sendall(self.reply.encode() + b"\r\n\r\n")
        if not self.reply.split()[1] == "200":
            conn.close()
            return
        up = socket.create_connection(("127.0.0.1", self.upstream_port))

        def pipe(a, b):
            for chunk in iter(lambda: a.recv(4096), b""):
                b.sendall(chunk)
            b.close()
        threading.Thread(target=pipe, args=(up, conn), daemon=True).start()
        pipe(conn, up)


def _unresolvable(monkeypatch, *hosts):
    """让这几个域名在"本机 DNS"里查不到（真去查的话，开了代理软件的机器什么域名都能解析出假 IP）。"""
    real = socket.create_connection

    def create_connection(addr, *a, **k):
        if addr[0] in hosts:
            raise socket.gaierror(11001, "getaddrinfo failed")
        return real(addr, *a, **k)
    monkeypatch.setattr(socket, "create_connection", create_connection)


def test_send_goes_through_the_system_proxy_when_one_is_set(no_proxy_env):
    """浏览器走代理上网的电脑：本机查不到邮件服务器的域名，照样经代理发出去。"""
    server = FakeSmtp()
    proxy = FakeProxy(server.port)
    no_proxy_env.setenv("https_proxy", proxy.url)
    _unresolvable(no_proxy_env, "smtp.behind-proxy.test")
    no_proxy_env.setattr(br, "_resolve_public", lambda host: pytest.fail("经代理能通就不该再查公共 DNS"))
    cfg = server.config()
    cfg.host = "smtp.behind-proxy.test"
    br.send(br.BugReport.collect("经代理发"), cfg)
    assert proxy.requests[0].startswith(f"CONNECT smtp.behind-proxy.test:{server.port} HTTP/1.1")
    assert len(server.received) == 1


def test_direct_connection_comes_first_even_with_a_proxy_set(no_proxy_env):
    """原来能直连的电脑不受影响：设了代理也先直连，代理一次都不碰。"""
    server = FakeSmtp()
    proxy = FakeProxy(server.port)
    no_proxy_env.setenv("https_proxy", proxy.url)
    br.send(br.BugReport.collect("直连"), server.config())
    assert len(server.received) == 1 and proxy.requests == []


def test_proxy_refusal_and_dns_failure_are_both_spelled_out(no_proxy_env, monkeypatch):
    proxy = FakeProxy(reply="HTTP/1.1 407 Proxy Authentication Required")
    no_proxy_env.setenv("https_proxy", proxy.url)
    monkeypatch.setattr(br, "_resolve_public", lambda host: (_ for _ in ()).throw(OSError("223.5.5.5 timed out")))
    _unresolvable(monkeypatch, "smtp.163.com")
    with pytest.raises(br.ConnectError) as ei:
        br.send(br.BugReport.collect("x"), br.SmtpConfig(host="smtp.163.com", timeout_s=5))
    msg = str(ei.value)
    assert "代理要求登录" in msg and "本机查不到 smtp.163.com" in msg and "公共 DNS 也查不到" in msg

    from cvflow.ui.bug_report_dialog import describe_send_error
    assert describe_send_error(ei.value, br.SmtpConfig(host="smtp.163.com")).startswith("连不上邮件服务器 smtp.163.com:465。")


def test_when_local_dns_fails_the_ip_from_public_dns_is_used(no_proxy_env, monkeypatch):
    """能上网但网卡 DNS 配坏了：改问公共 DNS，拿到 IP 直连。"""
    server = FakeSmtp()
    _unresolvable(monkeypatch, "smtp.broken-dns.test")
    asked = []
    monkeypatch.setattr(br, "_resolve_public", lambda host: asked.append(host) or ["127.0.0.1"])
    cfg = server.config()
    cfg.host = "smtp.broken-dns.test"
    br.send(br.BugReport.collect("公共 DNS"), cfg)
    assert asked == ["smtp.broken-dns.test"] and len(server.received) == 1


def test_dns_answer_with_cname_and_compressed_names_is_parsed():
    q = b"\x04smtp\x03163\x03com\x00"
    resp = (b"\x12\x34\x81\x80\x00\x01\x00\x02\x00\x00\x00\x00" + q + b"\x00\x01\x00\x01"
            + b"\xc0\x0c\x00\x05\x00\x01\x00\x00\x00\x3c\x00\x06\x03abc\xc0\x11"           # CNAME，跳过
            + b"\xc0\x2d\x00\x01\x00\x01\x00\x00\x00\x3c\x00\x04" + bytes([220, 181, 12, 16]))
    assert br._parse_dns_a(resp) == ["220.181.12.16"]
    nx = b"\x12\x34\x81\x83\x00\x01\x00\x00\x00\x00\x00\x00" + q + b"\x00\x01\x00\x01"
    assert br._parse_dns_a(nx) == []


def test_certificate_failure_is_not_reported_as_no_network():
    import ssl
    from cvflow.ui.bug_report_dialog import describe_send_error
    e = ssl.SSLCertVerificationError(1, "certificate verify failed")
    e.verify_message = "unable to get local issuer certificate"
    msg = describe_send_error(e, br.SmtpConfig(host="smtp.163.com"))
    assert "证书" in msg and "杀毒软件" in msg and "没有外网" not in msg

"""TCP 客户端、TCP 服务端与 UDP。

三者都在自己的后台线程里收发，不占用界面线程。字节流分帧由 ``Framer`` 负责，**每条连接一个
分帧器**——两个客户端的字节流绝不能混在一个缓冲里。

TCP 服务端支持按客户端发送：``send(data, peer="192.168.0.20:51234")`` 只发给那一个客户端，
``send(data)`` 按 ``default_target`` 决定广播还是只发给最近通信的那个。检测握手回复**必须**
指定对端，否则多台上位机同时连进来时结果会发错地方。

UDP 记住每个来源地址，``send(data, peer=...)`` 回到来源；``reply_to_source`` 打开时，
不指定对端的发送也会回到最近一次的来源，而不是配置里的固定目标。
"""
from __future__ import annotations

import socket
import threading
import time

from .base import CommDevice, CommError, Peer
from .framing import Framer


class TcpClientDevice(CommDevice):
    kind = "tcp_client"
    role = "client"
    config_schema = [
        ("host", "string", "目标地址", "127.0.0.1", "对端 IP 或主机名"),
        ("port", "int", "目标端口", 5000, ""),
        ("heartbeat_text", "string", "心跳报文", "HB\\n", "按心跳间隔发送，让对端知道视觉在线"),
    ]

    def __init__(self, name, config=None, bus=None, device_id=None, enabled=True):
        super().__init__(name, config, bus, device_id, enabled)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def connect(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"tcpc-{self.name}", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop.set()
        self._close("用户关闭")
        if self._thread:
            self._thread.join(2.0)
            self._thread = None

    def _close(self, reason: str) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
        self.reset_framer()
        self._set_connected(False, reason)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.config["host"], self.cfg_int("port", 5000)),
                                             timeout=self.connect_timeout)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(0.5)
                with self._lock:
                    self._sock = s
                self.reset_framer()
                peer = self._add_peer(f"{self.config['host']}:{self.config.get('port')}")
                self._set_connected(True)
                while not self._stop.is_set():
                    try:
                        data = s.recv(4096)
                    except socket.timeout:
                        self.check_frame_timeout()
                        continue
                    if not data:
                        raise ConnectionError("对端关闭了连接")
                    self._feed(data, peer)
            except (OSError, ConnectionError) as e:
                if not self._stop.is_set():
                    if self.connected or not self.last_error:
                        self._error(f"{e}")
                    self._close(str(e))
                    if not self.auto_reconnect:
                        return
                    self._stop.wait(self.reconnect_interval)

    def _send_bytes(self, data: bytes, peer: Peer | None = None) -> None:
        if self._sock is None:
            raise CommError("未连接")
        self._sock.sendall(data)

    def test_connection(self) -> tuple[bool, str]:
        host, port = self.config["host"], self.cfg_int("port", 5000)
        t0 = time.perf_counter()
        try:
            s = socket.create_connection((host, port), timeout=self.connect_timeout)
            s.close()
            return True, f"{host}:{port} 可连接，耗时 {(time.perf_counter() - t0) * 1000:.0f} ms"
        except OSError as e:
            return False, f"{host}:{port} 连接失败：{e}"


class _Client:
    """服务端侧的一个客户端连接：套接字 + 对端信息 + 它自己的分帧器。"""

    __slots__ = ("sock", "peer", "framer")

    def __init__(self, sock: socket.socket, peer: Peer, framer: Framer) -> None:
        self.sock, self.peer, self.framer = sock, peer, framer


class TcpServerDevice(CommDevice):
    kind = "tcp_server"
    role = "server"
    supports_peers = True
    config_schema = [
        ("host", "string", "监听地址", "0.0.0.0", "0.0.0.0 表示所有网卡"),
        ("port", "int", "监听端口", 6000, ""),
        ("max_clients", "int", "最大客户端数", 8, "超过时拒绝新连接"),
        ("default_target", "enum:broadcast,last,none", "不指定对端时发给谁", "broadcast",
         "broadcast 广播给所有客户端；last 只发给最近通信的那个；none 不发并计一次丢弃"),
        ("heartbeat_text", "string", "心跳报文", "HB\\n", "按心跳间隔广播"),
    ]

    def __init__(self, name, config=None, bus=None, device_id=None, enabled=True):
        super().__init__(name, config, bus, device_id, enabled)
        self._server: socket.socket | None = None
        self._clients: dict[str, _Client] = {}
        self._last_peer: str = ""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- 状态 ----
    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def bound_port(self) -> int:
        return self._server.getsockname()[1] if self._server else self.cfg_int("port", 6000)

    def client_ids(self) -> list[str]:
        return list(self._clients)

    @property
    def last_peer(self) -> str:
        return self._last_peer

    def peer_text(self) -> str:
        if not self._clients:
            return f"监听 {self.config.get('host')}:{self.bound_port}，无客户端"
        if len(self._clients) == 1:
            return next(iter(self._clients.values())).peer.address
        return f"{len(self._clients)} 个客户端"

    def test_connection(self) -> tuple[bool, str]:
        if self._server is not None:
            return True, (f"正在监听 {self.config['host']}:{self.bound_port}，"
                          f"当前 {self.client_count} 个客户端"
                          + (f"：{', '.join(self.client_ids())}" if self._clients else ""))
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.config["host"], self.cfg_int("port", 6000)))
            s.close()
            return True, f"端口 {self.config['port']} 可用（设备尚未启动监听）"
        except OSError as e:
            return False, f"无法监听 {self.config['host']}:{self.config['port']}：{e}"

    # ---- 生命周期 ----
    def connect(self) -> None:
        if self._server:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.config["host"], self.cfg_int("port", 6000)))
        srv.listen(max(1, self.cfg_int("max_clients", 8)))
        srv.settimeout(0.5)
        self._server = srv
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name=f"tcps-{self.name}", daemon=True)
        self._thread.start()
        self._set_connected(True)       # 服务端的"已连接"＝正在监听

    def disconnect(self) -> None:
        self._stop.set()
        with self._lock:
            for c in list(self._clients.values()):
                try:
                    c.sock.close()
                except OSError:
                    pass
            self._clients.clear()
            if self._server:
                try:
                    self._server.close()
                except OSError:
                    pass
                self._server = None
        if self._thread:
            self._thread.join(2.0)
            self._thread = None
        self._set_connected(False, "已关闭")

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._server.accept()
            except socket.timeout:
                for c in list(self._clients.values()):      # 顺带检查不完整报文超时
                    for err in c.framer.check_timeout():
                        self._error(f"{c.peer.id}：{err}")
                continue
            except OSError:
                break
            if len(self._clients) >= max(1, self.cfg_int("max_clients", 8)):
                self._error(f"客户端数已达上限 {self.cfg_int('max_clients', 8)}，拒绝 {addr[0]}:{addr[1]}")
                try:
                    conn.close()
                except OSError:
                    pass
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(0.5)
            pid = f"{addr[0]}:{addr[1]}"
            peer = self._add_peer(pid, pid)
            client = _Client(conn, peer, Framer(self.framing))
            with self._lock:
                self._clients[pid] = client
                self._last_peer = pid
            threading.Thread(target=self._client_loop, args=(client,), daemon=True,
                             name=f"tcps-{self.name}-{pid}").start()

    def _client_loop(self, client: _Client) -> None:
        try:
            while not self._stop.is_set():
                try:
                    data = client.sock.recv(4096)
                except socket.timeout:
                    for err in client.framer.check_timeout():
                        self._error(f"{client.peer.id}：{err}")
                    continue
                except OSError:
                    break
                if not data:
                    break
                with self._lock:
                    self._last_peer = client.peer.id
                if not self.byte_stream:
                    self._dispatch_rx(data, client.peer)
                    continue
                frames = client.framer.feed(data)
                for err in client.framer.take_errors():
                    self._error(f"{client.peer.id}：{err}")
                for frame in frames:
                    self._dispatch_rx(frame, client.peer)
        finally:
            with self._lock:
                self._clients.pop(client.peer.id, None)
            self._drop_peer(client.peer.id)
            try:
                client.sock.close()
            except OSError:
                pass

    def _send_bytes(self, data: bytes, peer: Peer | None = None) -> None:
        if peer is not None:
            client = self._clients.get(peer.id)
            if client is None:
                raise CommError(f"客户端 {peer.id} 已断开")
            client.sock.sendall(data)
            return
        target = str(self.config.get("default_target", "broadcast"))
        if target == "none" or not self._clients:
            self.stats["dropped"] += 1
            return                       # 没人连着不算错误：服务端本来就可能空闲
        if target == "last":
            client = self._clients.get(self._last_peer)
            if client is None:
                self.stats["dropped"] += 1
                return
            client.sock.sendall(data)
            return
        dead = []
        for pid, client in list(self._clients.items()):
            try:
                client.sock.sendall(data)
                client.peer.tx += 1
            except OSError:
                dead.append(pid)
        for pid in dead:
            self._clients.pop(pid, None)
            self._drop_peer(pid)

    def info(self):
        d = super().info()
        d["clients"] = self.client_count
        return d


class UdpDevice(CommDevice):
    kind = "udp"
    role = "peer"
    byte_stream = False         # 一个数据报就是一条报文，边界由协议保证
    supports_peers = True
    config_schema = [
        ("host", "string", "目标地址", "127.0.0.1", ""),
        ("port", "int", "目标端口", 7000, ""),
        ("local_host", "string", "本地绑定地址", "0.0.0.0", ""),
        ("local_port", "int", "本地绑定端口", 7001, "0 表示由系统分配"),
        ("reply_to_source", "bool", "回复发给来源", True,
         "打开时，不指定对端的发送回到最近一次来源地址，而不是上面的固定目标"),
        ("encoding", "string", "文本编码", "utf-8", ""),
        ("max_frame", "int", "单个数据报上限（字节）", 65535, ""),
        ("heartbeat_text", "string", "心跳报文", "", "留空关闭"),
    ]

    def __init__(self, name, config=None, bus=None, device_id=None, enabled=True):
        super().__init__(name, config, bus, device_id, enabled)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_peer: str = ""

    @property
    def bound_port(self) -> int:
        return self._sock.getsockname()[1] if self._sock else self.cfg_int("local_port", 0)

    @property
    def last_peer(self) -> str:
        return self._last_peer

    def peer_text(self) -> str:
        fixed = f"{self.config.get('host')}:{self.config.get('port')}"
        if self._last_peer:
            return f"来源 {self._last_peer} / 目标 {fixed}"
        return f"目标 {fixed}"

    def connect(self) -> None:
        if self._sock:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((str(self.config.get("local_host", "0.0.0.0") or "0.0.0.0"), self.cfg_int("local_port", 0)))
        s.settimeout(0.5)
        self._sock = s
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"udp-{self.name}", daemon=True)
        self._thread.start()
        self._set_connected(True)

    def disconnect(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread:
            self._thread.join(2.0)
            self._thread = None
        self._set_connected(False, "已关闭")

    def _loop(self) -> None:
        limit = max(1, self.cfg_int("max_frame", 65535))
        while not self._stop.is_set() and self._sock:
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            pid = f"{addr[0]}:{addr[1]}"
            peer = self._add_peer(pid, pid)
            self._last_peer = pid
            if len(data) > limit:
                self._error(f"数据报 {len(data)} 字节超过上限 {limit}，已截断")
                data = data[:limit]
            self._dispatch_rx(data, peer)

    def _send_bytes(self, data: bytes, peer: Peer | None = None) -> None:
        if not self._sock:
            raise CommError("未打开")
        if peer is not None:
            host, _, port = peer.address.rpartition(":")
            self._sock.sendto(data, (host, int(port)))
            return
        if self.config.get("reply_to_source") and self._last_peer:
            host, _, port = self._last_peer.rpartition(":")
            self._sock.sendto(data, (host, int(port)))
            return
        self._sock.sendto(data, (self.config["host"], self.cfg_int("port", 7000)))

    def test_connection(self) -> tuple[bool, str]:
        if self._sock is not None:
            return True, (f"UDP 已绑定 {self.config.get('local_host', '0.0.0.0')}:{self.bound_port}，"
                          f"目标 {self.config['host']}:{self.config['port']}"
                          + (f"，最近来源 {self._last_peer}" if self._last_peer else ""))
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((str(self.config.get("local_host", "0.0.0.0") or "0.0.0.0"), self.cfg_int("local_port", 0)))
            s.sendto(b"", (self.config["host"], self.cfg_int("port", 7000)))
            s.close()
            return True, (f"UDP 无连接概念；本地端口 {self.config.get('local_port', 0)} 可绑定，"
                          f"远端 {self.config['host']}:{self.config['port']} 可寻址")
        except OSError as e:
            return False, f"UDP 端口检查失败：{e}"

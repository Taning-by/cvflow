"""TCP client, TCP server and UDP devices (threaded, auto-reconnecting)."""
from __future__ import annotations

import socket
import threading
import time

from .base import CommDevice, CommError


class TcpClientDevice(CommDevice):
    kind = "tcp_client"
    config_schema = [
        ("host", "string", "Host", "127.0.0.1", "Remote IP"),
        ("port", "int", "Port", 5000, "Remote port"),
        ("terminator", "string", "Terminator", "\\n", "Frame terminator (escapes ok, empty = raw)"),
        ("encoding", "string", "Encoding", "utf-8", ""),
        ("auto_reconnect", "bool", "Auto reconnect", True, ""),
        ("reconnect_s", "float", "Reconnect interval (s)", 2.0, ""),
    ]

    def __init__(self, name, config=None, bus=None):
        super().__init__(name, config, bus)
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
        self._close("closed by user")
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
        self._buffer = b""
        self._set_connected(False, reason)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                s = socket.create_connection((self.config["host"], int(self.config["port"])), timeout=3.0)
                s.settimeout(0.5)
                with self._lock:
                    self._sock = s
                self._set_connected(True)
                while not self._stop.is_set():
                    try:
                        data = s.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        raise ConnectionError("peer closed")
                    self._feed(data)
            except (OSError, ConnectionError) as e:
                if not self._stop.is_set():
                    if self.connected or not self.last_error:
                        self._error(f"{e}")
                    self._close(str(e))
                    if not self.config.get("auto_reconnect", True):
                        return
                    self._stop.wait(float(self.config.get("reconnect_s", 2.0)))

    def _send_bytes(self, data: bytes) -> None:
        if self._sock is None:
            raise CommError("not connected")
        self._sock.sendall(data)


class TcpServerDevice(CommDevice):
    """Listens for PLC/HMI clients; ``send`` broadcasts to every connected client."""

    kind = "tcp_server"
    config_schema = [
        ("host", "string", "Bind address", "0.0.0.0", ""),
        ("port", "int", "Port", 6000, ""),
        ("terminator", "string", "Terminator", "\\n", "Frame terminator (escapes ok, empty = raw)"),
        ("encoding", "string", "Encoding", "utf-8", ""),
    ]

    def __init__(self, name, config=None, bus=None):
        super().__init__(name, config, bus)
        self._server: socket.socket | None = None
        self._clients: dict[socket.socket, bytes] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def bound_port(self) -> int:
        return self._server.getsockname()[1] if self._server else int(self.config["port"])

    def connect(self) -> None:
        if self._server:
            return
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.config["host"], int(self.config["port"])))
        srv.listen(8)
        srv.settimeout(0.5)
        self._server = srv
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name=f"tcps-{self.name}", daemon=True)
        self._thread.start()
        self._set_connected(True)  # for a server "connected" means listening

    def disconnect(self) -> None:
        self._stop.set()
        with self._lock:
            for c in list(self._clients):
                try:
                    c.close()
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
        self._set_connected(False, "closed")

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(0.5)
            with self._lock:
                self._clients[conn] = b""
            threading.Thread(target=self._client_loop, args=(conn, addr), daemon=True).start()

    def _client_loop(self, conn: socket.socket, addr) -> None:
        term = self.terminator
        buf = b""
        try:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                if not term:
                    self._dispatch_rx(data)
                    continue
                buf += data
                while True:
                    i = buf.find(term)
                    if i < 0:
                        break
                    frame, buf = buf[:i], buf[i + len(term):]
                    if frame:
                        self._dispatch_rx(frame)
        except OSError:
            pass
        finally:
            with self._lock:
                self._clients.pop(conn, None)
            try:
                conn.close()
            except OSError:
                pass

    def _send_bytes(self, data: bytes) -> None:
        if not self._clients:
            self.stats["dropped"] = self.stats.get("dropped", 0) + 1
            return  # nobody listening: not an error for a server
        dead = []
        for c in list(self._clients):
            try:
                c.sendall(data)
            except OSError:
                dead.append(c)
        for c in dead:
            self._clients.pop(c, None)


class UdpDevice(CommDevice):
    kind = "udp"
    config_schema = [
        ("host", "string", "Remote host", "127.0.0.1", ""),
        ("port", "int", "Remote port", 7000, ""),
        ("local_port", "int", "Local port", 7001, "0 = any"),
        ("terminator", "string", "Terminator", "", "Usually empty: one datagram = one frame"),
        ("encoding", "string", "Encoding", "utf-8", ""),
    ]

    def __init__(self, name, config=None, bus=None):
        super().__init__(name, config, bus)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def connect(self) -> None:
        if self._sock:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", int(self.config.get("local_port", 0))))
        s.settimeout(0.5)
        self._sock = s
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"udp-{self.name}", daemon=True)
        self._thread.start()
        self._set_connected(True)

    def disconnect(self) -> None:
        self._stop.set()
        if self._sock:
            self._sock.close()
            self._sock = None
        if self._thread:
            self._thread.join(2.0)
            self._thread = None
        self._set_connected(False, "closed")

    def _loop(self) -> None:
        while not self._stop.is_set() and self._sock:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self._feed(data)

    def _send_bytes(self, data: bytes) -> None:
        if not self._sock:
            raise CommError("not open")
        self._sock.sendto(data, (self.config["host"], int(self.config["port"])))

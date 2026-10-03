import socket
import struct
import threading
import time

import pytest

from cvflow.camera import discovery, enumerate_cameras
from cvflow.camera import test_camera as camera_test
from cvflow.camera.discovery import GigEDevice, build_discovery_ack, parse_discovery_ack

FAKE = GigEDevice(ip="192.168.1.20", mac="C4:2F:90:11:22:33", subnet="255.255.255.0", manufacturer="Hikrobot",
                  model="MV-CA050-10GM", version="V1.6.2", serial="DA1234567", user_name="cam1")


class FakeGigECamera(threading.Thread):
    """Answers GVCP discovery / readreg / forceip on 127.0.0.1:3956."""

    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", discovery.GVCP_PORT))
        self.sock.settimeout(0.2)
        self.stop = threading.Event()
        self.forced_ip = None

    def run(self):
        while not self.stop.is_set():
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            if len(data) < 8 or data[0] != 0x42:
                continue
            cmd, length, rid = struct.unpack(">HHH", data[2:8])
            if cmd == 0x0002:
                self.sock.sendto(build_discovery_ack(FAKE, rid), addr)
            elif cmd == 0x0080:
                (address,) = struct.unpack(">I", data[8:12])
                value = 0x00010002 if address == 0 else 0
                self.sock.sendto(struct.pack(">HHHH", 0, 0x0081, 4, rid) + struct.pack(">I", value), addr)
            elif cmd == 0x0004:
                self.forced_ip = socket.inet_ntoa(data[8 + 20:8 + 24])
                self.sock.sendto(struct.pack(">HHHH", 0, 0x0005, 0, rid), addr)

    def close(self):
        self.stop.set()
        self.join(1.0)
        self.sock.close()


@pytest.fixture
def fake_cam():
    try:
        cam = FakeGigECamera()
    except OSError as e:
        pytest.skip(f"cannot bind GVCP port: {e}")
    cam.start()
    yield cam
    cam.close()


def test_discovery_packet_roundtrip():
    pkt = build_discovery_ack(FAKE, 9)
    assert struct.unpack(">HHHH", pkt[:8]) == (0, 0x0003, 248, 9)
    d = parse_discovery_ack(pkt[8:])
    assert (d.ip, d.mac, d.manufacturer, d.model, d.serial, d.user_name) == \
        ("192.168.1.20", "C4:2F:90:11:22:33", "Hikrobot", "MV-CA050-10GM", "DA1234567", "cam1")
    assert d.subnet == "255.255.255.0" and d.display_name == "cam1"
    assert parse_discovery_ack(b"\x00" * 10) is None


def test_discover_by_ip_and_connection_test(fake_cam):
    devs = discovery.discover(timeout=0.5, targets=["127.0.0.1"])
    assert len(devs) == 1 and devs[0].serial == "DA1234567" and devs[0].model == "MV-CA050-10GM"
    ok, msg = discovery.test_connection("127.0.0.1", timeout=1.0)
    assert ok and "1.2" in msg
    assert discovery.read_register("127.0.0.1", 0x0000) == 0x00010002
    ok, msg = discovery.test_connection("127.0.0.1" if False else "192.0.2.1", timeout=0.3)
    assert not ok and "无应答" in msg


def test_enumerate_cameras_includes_gvcp(fake_cam):
    devs = enumerate_cameras(timeout=0.5, targets=["127.0.0.1"])
    assert any(d.serial == "DA1234567" and "gvcp" in d.extra.get("sources", []) for d in devs)


def test_test_camera_helper(images_dir):
    ok, msg = camera_test("folder", str(images_dir))
    assert ok
    ok, msg = camera_test("hik", "192.0.2.1")
    assert not ok and "无应答" in msg
    ok, msg = camera_test("folder", "/nonexistent/dir")
    assert not ok


def test_camera_node_software_trigger_and_settings(reg, images_dir):
    from cvflow.core import Graph
    from cvflow.core.engine import Engine
    g = Graph()
    n = g.add_node(reg.create("source.camera", values={"camera": "trig_cam", "kind": "folder", "source": str(images_dir),
                                                       "trigger_mode": "software", "exposure_us": 5000, "gain": 2.0}))
    eng = Engine(g)
    r = eng.run()
    assert r.ok and r.node_results[n.id].outputs["frame_id"] == 1
    eng.teardown_nodes()

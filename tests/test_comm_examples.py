"""三个示例方案的验收测试：直接加载 ``examples/solutions`` 里的文件，从对端观察。

这些用例既验证示例本身可用，也验证"方案文件 → 通信配置 → 业务闭环"整条链路：
方案里写的分帧、解析/格式化规则、数据点映射、检测握手，读进来之后真的按预期工作。

* 示例 A（``demo_tcp_handshake``）：上位机 ``TRIGGER,1001`` → ``RESULT,1001,OK,x,y,角度``，
  并覆盖参数错误、重复编号、多客户端路由；另外用 ``cvflow serve`` 子进程跑一遍无界面模式。
* 示例 B（``demo_modbus_master``）：cvflow 做主站，轮询模拟 PLC（从站）的触发点。
* 示例 C（``demo_modbus_slave``）：cvflow 做从站，外部主站写触发、读结果、写确认。
"""
from __future__ import annotations

import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cvflow.comm import CommManager, set_manager
from cvflow.comm.modbus import ModbusTcpServerDevice
from cvflow.comm.modbus_client import ModbusClient, TcpTransport
from cvflow.core import FlowRunner, Solution, registry

ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = ROOT / "examples" / "solutions"
registry.load_builtins()

# 寄存器映射（与 examples/build_handshake_demos.py 一致）
TRIGGER, REQUEST_ID, READY, BUSY, DONE, RESULT, ERROR_CODE, DONE_ID = 0, 1, 3, 4, 5, 6, 7, 8
ACK, HOLES, X, Y, ANGLE, WIDTH, HEARTBEAT = 10, 20, 21, 23, 25, 27, 30

RESULT_RE = re.compile(r"^RESULT,(\d+),(OK|NG|ERROR),(-?\d+\.\d{3}),(-?\d+\.\d{3}),(-?\d+\.\d{3})$")


def _wait(cond, timeout=10.0, step=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def f32(words) -> float:
    """CDAB：低字在前。"""
    return struct.unpack(">f", struct.pack(">HH", words[1], words[0]))[0]


def i32(words) -> int:
    return struct.unpack(">i", struct.pack(">HH", words[1], words[0]))[0]


def to_i32(value: int) -> list[int]:
    hi, lo = struct.unpack(">HH", struct.pack(">i", int(value)))
    return [lo, hi]


class Example:
    """加载一个示例方案并按运行模式启动（设备 + 流程 + 握手）。"""

    def __init__(self, name: str, patch_device: dict | None = None):
        path = SOLUTIONS / f"{name}.json"
        assert path.is_file(), f"{path} 不存在，请先运行 python examples/build_handshake_demos.py"
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in (patch_device or {}).items():
            data["comm"]["devices"][0]["config"][key] = value
        self.sol = Solution.from_dict(data, base_dir=path.parent)
        self.sol.path = path
        from cvflow.core import paths
        paths.set_base_dir(path.parent)       # 方案里的相对路径（图像目录）按方案所在目录解析
        self.mgr = CommManager(self.sol.bus, self.sol.variables)
        self.errors = self.mgr.load_dict(self.sol.comm_config)
        self.runners = {n: FlowRunner(g, self.sol.variables, self.sol.bus)
                        for n, g in self.sol.flows.items()}
        self.mgr.set_runners(self.runners)
        set_manager(self.mgr)

    @property
    def device(self):
        return next(iter(self.mgr.devices.values()))

    def start(self) -> list[str]:
        errors = self.mgr.connect_all()
        for r in self.runners.values():
            errors += r.start()
        return errors

    def stop(self):
        for r in self.runners.values():
            r.stop()
        self.mgr.shutdown()
        set_manager(None)


class LineClient:
    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self._buf = b""

    def send(self, text: str):
        self.sock.sendall(text.encode())

    def line(self, timeout=10.0) -> str:
        self.sock.settimeout(timeout)
        while b"\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            self._buf += chunk
        line, _, rest = self._buf.partition(b"\n")
        self._buf = rest
        return line.decode(errors="replace")

    def quiet(self, timeout=0.5) -> bool:
        self.sock.settimeout(timeout)
        try:
            return not self.sock.recv(4096)
        except socket.timeout:
            return True

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ============================================================ 示例 A
@pytest.fixture
def example_a():
    ex = Example("demo_tcp_handshake", {"port": 0})
    assert ex.errors == [], ex.errors
    assert ex.mgr.validate() == [], ex.mgr.validate()
    yield ex
    ex.stop()


def test_example_a_tcp_request_response_cycle(example_a):
    ex = example_a
    assert ex.start() == []
    c = LineClient(ex.device.bound_port)
    time.sleep(0.25)
    c.send("TRIGGER,1001\n")
    reply = c.line()
    m = RESULT_RE.match(reply)
    assert m, f"回复不符合约定格式：{reply!r}"
    assert m.group(1) == "1001" and m.group(2) == "OK"
    # 第二次请求用新编号，结果应该重新测量
    c.send("TRIGGER,1002\n")
    assert RESULT_RE.match(c.line()).group(1) == "1002"
    assert ex.runners["main"].stats.count == 2
    c.close()


def test_example_a_bad_params_and_duplicate_and_routing(example_a):
    ex = example_a
    assert ex.start() == []
    a = LineClient(ex.device.bound_port)
    b = LineClient(ex.device.bound_port)
    time.sleep(0.25)
    # 参数错误：不触发流程，回 ERROR,<编号>,2
    a.send("TRIGGER,abc\n")
    assert a.line() == "ERROR,0,2"
    assert ex.runners["main"].stats.count == 0
    # 正常请求
    a.send("TRIGGER,2001\n")
    first = a.line()
    assert RESULT_RE.match(first)
    assert b.quiet(0.5)                     # 结果只回给发起请求的客户端
    # 重复编号：重发上次结果，不重跑流程
    a.send("TRIGGER,2001\n")
    assert a.line() == first
    assert ex.runners["main"].stats.count == 1
    # 换客户端请求，结果回到它那里
    b.send("TRIGGER,2002\n")
    assert RESULT_RE.match(b.line()).group(1) == "2002"
    assert a.quiet(0.5)
    hs = ex.mgr.handshakes[0]
    assert hs.counters["duplicate"] == 1 and hs.counters["completed"] == 2
    a.close()
    b.close()


def test_example_a_flow_error_is_reported_as_error_status(example_a):
    """流程里出异常时，回复的状态字段是 ERROR，错误码是“流程异常”。"""
    ex = example_a
    bad = registry.create("logic.expression", name="故障注入", values={"expression": "missing_name + 1"})
    ex.sol.flows["main"].add_node(bad)
    assert ex.start() == []
    c = LineClient(ex.device.bound_port)
    time.sleep(0.25)
    c.send("TRIGGER,3001\n")
    reply = c.line()
    assert reply.startswith("RESULT,3001,ERROR"), reply
    from cvflow.comm.handshake import ERROR_CODES
    assert ex.mgr.handshakes[0].last_error_code == ERROR_CODES["flow_error"]
    c.close()


def test_example_a_runs_under_cvflow_serve(tmp_path):
    """无界面生产模式：cvflow serve 跑同一个方案，对端行为一致。"""
    src = SOLUTIONS / "demo_tcp_handshake.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    data["comm"]["devices"][0]["config"]["port"] = 0
    tmp = src.parent / "_ex_tcp_serve.json"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    log_path = tmp_path / "serve.log"
    log = open(log_path, "w", encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "cvflow", "serve", str(tmp), "--stats-every", "100"],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        def bound_port():
            log.flush()
            m = re.search(r"监听 [^:\s]+:(\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
            return int(m.group(1)) if m else None
        _wait(lambda: proc.poll() is not None or bound_port(), 60)
        port = bound_port()
        if proc.poll() is not None or not port:
            raise AssertionError(f"serve 退出码 {proc.poll()}，输出：\n"
                                 f"{log_path.read_text(encoding='utf-8', errors='replace')[-2000:]}")
        c = LineClient(port)
        time.sleep(0.3)
        c.send("TRIGGER,4001\n")
        reply = c.line(20)
        assert RESULT_RE.match(reply), reply
        assert reply.startswith("RESULT,4001,OK")
        c.close()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)
        log.close()
        tmp.unlink(missing_ok=True)


# ============================================================ 示例 B：cvflow 做主站
def test_example_b_master_polls_simulated_plc():
    port = free_port()
    plc = ModbusTcpServerDevice("模拟PLC", {"host": "127.0.0.1", "port": port, "unit_id": 1,
                                            "register_count": 64})
    plc.connect()
    ex = Example("demo_modbus_master", {"port": port})
    assert ex.errors == [], ex.errors
    assert ex.mgr.validate() == [], ex.mgr.validate()
    try:
        assert ex.start() == []
        assert _wait(lambda: ex.device.connected, 10)
        assert _wait(lambda: plc.read_area("holding", READY, 1)[0] == 1, 10)   # Ready 由视觉写上来
        plc.write_area("holding", REQUEST_ID, to_i32(2001))
        plc.write_area("holding", TRIGGER, [1])
        assert _wait(lambda: plc.read_area("holding", DONE, 1)[0] == 1, 15)
        regs = plc.read_area("holding", 0, 32)
        assert regs[BUSY] == 0 and regs[RESULT] in (1, 2) and regs[ERROR_CODE] == 0
        assert i32(regs[DONE_ID:DONE_ID + 2]) == 2001          # 完成编号＝请求编号
        assert regs[HOLES] == 3
        assert 0 <= f32(regs[X:X + 2]) < 2000 and 0 <= f32(regs[Y:Y + 2]) < 2000
        assert 300 < f32(regs[WIDTH:WIDTH + 2]) < 500
        assert ex.runners["main"].stats.count == 1
        # 确认 → 复位
        plc.write_area("holding", TRIGGER, [0])
        plc.write_area("holding", ACK, [1])
        assert _wait(lambda: plc.read_area("holding", DONE, 1)[0] == 0, 10)
        assert _wait(lambda: plc.read_area("holding", READY, 1)[0] == 1, 10)
        assert _wait(lambda: plc.read_area("holding", HEARTBEAT, 1)[0] >= 1, 10)   # 心跳
        # 第二次请求
        plc.write_area("holding", REQUEST_ID, to_i32(2002))
        plc.write_area("holding", TRIGGER, [1])
        assert _wait(lambda: i32(plc.read_area("holding", DONE_ID, 2)) == 2002, 15)
        assert ex.runners["main"].stats.count == 2
    finally:
        ex.stop()
        plc.disconnect()


# ============================================================ 示例 C：cvflow 做从站
def test_example_c_slave_serves_external_master():
    ex = Example("demo_modbus_slave", {"port": 0})
    assert ex.errors == [], ex.errors
    assert ex.mgr.validate() == [], ex.mgr.validate()
    try:
        assert ex.start() == []
        port = ex.device.bound_port
        plc = ModbusClient(TcpTransport("127.0.0.1", port, 3.0), unit_id=1, timeout=2.0)
        assert _wait(lambda: plc.read_words("holding", READY, 1)[0] == 1, 10)
        plc.write_registers(REQUEST_ID, to_i32(3001))
        plc.write_register(TRIGGER, 1)
        assert _wait(lambda: plc.read_words("holding", DONE, 1)[0] == 1, 15)
        regs = plc.read_words("holding", 0, 32)
        assert regs[BUSY] == 0 and regs[ERROR_CODE] == 0
        assert i32(regs[DONE_ID:DONE_ID + 2]) == 3001
        assert regs[HOLES] == 3 and 300 < f32(regs[WIDTH:WIDTH + 2]) < 500
        plc.write_register(TRIGGER, 0)
        plc.write_register(ACK, 1)
        assert _wait(lambda: plc.read_words("holding", DONE, 1)[0] == 0, 10)
        assert _wait(lambda: plc.read_words("holding", READY, 1)[0] == 1, 10)
        plc.close()
    finally:
        ex.stop()


def test_example_c_interoperates_with_third_party_master():
    """同一个示例，用第三方实现（pymodbus）当主站，行为一致。"""
    pytest.importorskip("pymodbus")
    from pymodbus.client import ModbusTcpClient
    ex = Example("demo_modbus_slave", {"port": 0})
    try:
        assert ex.start() == []
        c = ModbusTcpClient("127.0.0.1", port=ex.device.bound_port, timeout=2)
        assert c.connect()
        assert _wait(lambda: c.read_holding_registers(READY, count=1, device_id=1).registers[0] == 1, 10)
        assert not c.write_registers(REQUEST_ID, to_i32(4001), device_id=1).isError()
        assert not c.write_register(TRIGGER, 1, device_id=1).isError()
        assert _wait(lambda: c.read_holding_registers(DONE, count=1, device_id=1).registers[0] == 1, 15)
        regs = c.read_holding_registers(0, count=32, device_id=1).registers
        assert i32(regs[DONE_ID:DONE_ID + 2]) == 4001 and regs[HOLES] == 3
        c.close()
    finally:
        ex.stop()


def test_example_c_rejects_second_request_while_busy_or_unacked():
    """没确认之前 Ready 为 0，再来的触发被拒绝并写忙错误码，流程只跑一次。"""
    from cvflow.comm.handshake import ERROR_CODES
    ex = Example("demo_modbus_slave", {"port": 0})
    try:
        assert ex.start() == []
        plc = ModbusClient(TcpTransport("127.0.0.1", ex.device.bound_port, 3.0), unit_id=1, timeout=2.0)
        assert _wait(lambda: plc.read_words("holding", READY, 1)[0] == 1, 10)
        plc.write_registers(REQUEST_ID, to_i32(5001))
        plc.write_register(TRIGGER, 1)
        assert _wait(lambda: plc.read_words("holding", DONE, 1)[0] == 1, 15)
        assert plc.read_words("holding", READY, 1)[0] == 0        # 等确认，不接新任务
        plc.write_register(TRIGGER, 0)
        plc.write_registers(REQUEST_ID, to_i32(5002))
        plc.write_register(TRIGGER, 1)
        assert _wait(lambda: plc.read_words("holding", ERROR_CODE, 1)[0] == ERROR_CODES["busy"], 10)
        assert ex.runners["main"].stats.count == 1
        plc.close()
    finally:
        ex.stop()

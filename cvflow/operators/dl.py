"""Deep-learning and scripting nodes.

* OnnxInference / OnnxClassifier / OnnxDetector run any exported model with onnxruntime.
* PythonScript executes user code (PyTorch, an RL policy, anything importable) inside the flow.

For a reusable algorithm write a plugin file instead (see examples/plugins).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ..core import paths
from ..core.batching import BatchExecutor
from ..core.node import Node, NodeError, Param, Port
from ..core.registry import register
from ..core.types import DataType, Image, Overlay, Rect
from ._util import as_bgr, as_gray, parse_floats, require_image

_COLOR = "#ad1457"
log = logging.getLogger("cvflow.dl")


def _load_labels(path: str) -> list[str]:
    path = paths.resolve(path)
    if not path or not os.path.isfile(path):
        return []
    return [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]


# --------------------------------------------------------------------------- 模型输入形状
def read_input_spec(session) -> dict:
    """读出模型第一个输入的形状，推断布局、通道数与固定的高宽。

    返回的 ``height`` / ``width`` / ``layout`` / ``channels`` 只在模型把它们写死时才有值，
    动态维度一律是 None，表示由节点参数决定。
    """
    inp = session.get_inputs()[0]
    shape = list(inp.shape or [])
    spec = {"name": inp.name, "shape": shape, "dynamic_batch": True,
            "layout": None, "channels": None, "height": None, "width": None}
    if not shape:
        return spec
    spec["dynamic_batch"] = not isinstance(shape[0], int)
    if len(shape) == 4:
        def at(i):
            return shape[i] if isinstance(shape[i], int) else None
        c_first, c_last = at(1), at(3)
        if c_first in (1, 3, 4) and c_last not in (1, 3, 4):
            spec.update(layout="NCHW", channels=c_first, height=at(2), width=at(3))
        elif c_last in (1, 3, 4) and c_first not in (1, 3, 4):
            spec.update(layout="NHWC", channels=c_last, height=at(1), width=at(2))
    return spec


def describe_shape(shape: list) -> str:
    return "×".join(str(d) if isinstance(d, int) else f"{d or '动态'}" for d in shape) if shape else "未知"


# --------------------------------------------------------------------------- 共享会话
_FAILED_PROVIDERS: set[str] = set()       # 本进程内加载失败过的推理后端，不再重复尝试
_PROVIDER_NAMES = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
                   "tensorrt": "TensorrtExecutionProvider"}
_GPU_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")
_PROVIDER_FIX = {
    "CUDAExecutionProvider": "GPU 后端要装 GPU 版：pip uninstall onnxruntime && pip install onnxruntime-gpu"
                             "（或 pip install -e \".[gpu]\"），再装对应版本的 CUDA 12 与 cuDNN 9 运行库",
    "TensorrtExecutionProvider": "TensorRT 后端除了 GPU 版 onnxruntime，还要另外安装 TensorRT 本体；"
                                 "只想用显卡的话，推理后端选 cuda 或 auto 即可",
}
_WHEEL_CONFLICT_WARNED = False
_GPU_RUNTIME_PRELOADED = False
_PROVIDER_LOADABLE: dict[str, bool] = {}   # GPU 后端的动态库能否加载，探测一次就记住


def _preload_gpu_runtime(ort) -> None:
    """把 pip 装的 CUDA / cuDNN 运行库预加载进来，让 GPU 后端能被 dlopen 到。

    onnxruntime-gpu 的 CUDA 后端是一个单独的动态库，运行时才去找 libcublasLt、libcudnn 等等。
    用 pip 装的 nvidia-* 包并不在系统库搜索路径里（Linux 的 LD_LIBRARY_PATH / Windows 的 PATH），
    于是会出现"包里明明有 CUDA 后端却加载失败"。onnxruntime 自带的 preload_dlls() 正是为此准备的，
    Windows 与 Linux 都适用；没有这个接口的老版本就跳过，行为和以前一样。
    """
    global _GPU_RUNTIME_PRELOADED
    if _GPU_RUNTIME_PRELOADED or not hasattr(ort, "preload_dlls"):
        return
    _GPU_RUNTIME_PRELOADED = True
    try:
        ort.preload_dlls()
    except Exception as e:                             # pragma: no cover - 取决于本机安装
        log.debug("预加载 CUDA/cuDNN 运行库失败：%s", e)


def _provider_loadable(ort, provider: str) -> bool:
    """GPU 后端的动态库能不能真的加载进来，探测一次就记住。

    必须先剔掉不可用的后端再建会话：onnxruntime 的行为是只要列表里有一个后端加载失败，
    整个会话就退回 CPU——装了 GPU 版但没装 TensorRT 时，auto 会因为 TensorRT 失败而
    把本来能用的 CUDA 一起丢掉，结果整个流程都跑在 CPU 上。
    """
    cached = _PROVIDER_LOADABLE.get(provider)
    if cached is not None:
        return cached
    why = _why_provider_failed(ort, provider)
    _PROVIDER_LOADABLE[provider] = not why
    if why and provider == "TensorrtExecutionProvider":
        log.info("推理后端 %s 的运行库加载不了（%s），跳过它；没装 TensorRT 本体时这是正常的", provider, why)
    elif why:
        log.warning("推理后端 %s 的运行库加载不了（%s），本次运行跳过它，推理会落在 CPU 上", provider, why)
    return not why


def _why_provider_failed(ort, provider: str) -> str:
    """GPU 后端加载失败时，把缺的那个库名找出来（onnxruntime 自己只把它写到 stderr）。"""
    import ctypes
    import glob
    tag = "tensorrt" if provider == "TensorrtExecutionProvider" else "cuda"
    base = os.path.join(os.path.dirname(os.path.abspath(ort.__file__)), "capi")
    for lib in sorted(glob.glob(os.path.join(base, f"*providers_{tag}*"))):
        try:
            if os.name == "nt":
                # 按绝对路径加载时，Windows 默认不会去这个 DLL 自己的目录里找它的依赖
                # （provider 库要用到同目录的 onnxruntime_providers_shared.dll），
                # 必须带上 LOAD_WITH_ALTERED_SEARCH_PATH，否则会误判成"加载不了"。
                ctypes.CDLL(lib, winmode=0x00000008)
            else:
                ctypes.CDLL(lib)
        except OSError as e:
            return str(e)
        except Exception:                              # pragma: no cover
            return ""
    return ""


def _warn_if_both_wheels_installed() -> None:
    """同时装了 onnxruntime 和 onnxruntime-gpu 时提醒一次。

    两个包装的是同一个 Python 模块，谁后装谁覆盖，常见症状就是"明明装了 GPU 版还是只有 CPU 后端"。
    """
    global _WHEEL_CONFLICT_WARNED
    if _WHEEL_CONFLICT_WARNED:
        return
    _WHEEL_CONFLICT_WARNED = True
    try:
        from importlib.metadata import distributions
        names = {d.metadata["Name"].lower() for d in distributions() if d.metadata["Name"]}
    except Exception:                                  # pragma: no cover
        return
    if {"onnxruntime", "onnxruntime-gpu"} <= names:
        log.warning("同时安装了 onnxruntime 与 onnxruntime-gpu，它们会互相覆盖，可能导致 GPU 后端不可用；"
                    "建议 pip uninstall onnxruntime，只保留 onnxruntime-gpu")


class _SharedSession:
    """一份已加载的模型权重。

    共享的粒度由 ``_OnnxBase._session_key`` 决定：
    ``(模型文件, 修改时间, 后端, 显卡编号, 会话归属)``。归属为空表示全软件共用一份——
    CPU 上默认如此；GPU 上默认按合批分组/节点各加载一份，以便用各自的 CUDA 流并行。
    预处理参数不同不影响共享，它只决定批处理器怎么分。
    """

    def __init__(self, session, spec: dict, providers: list[str] | None = None) -> None:
        self.session = session
        self.spec = spec
        self.providers = providers or []       # 实际生效的推理后端，第一个是主用的
        self.refs = 0


_SESSIONS: dict[tuple, _SharedSession] = {}
_EXECUTORS: dict[tuple, "_Runner"] = {}
_CACHE_LOCK = threading.RLock()


def _get_or_create_session(key: tuple, factory) -> _SharedSession:
    with _CACHE_LOCK:
        holder = _SESSIONS.get(key)
        if holder is None:
            holder = factory()
            _SESSIONS[key] = holder
        return holder


def _acquire_preloaded(key: tuple, factory) -> None:
    """为"提前加载"占一个引用。

    权重是跨节点共享的，预加载也必须计数：否则两个节点都预加载了同一个模型，
    其中一个被禁用就会把另一个还在用的那份权重也删掉。
    """
    with _CACHE_LOCK:
        holder = _SESSIONS.get(key)
        if holder is None:
            holder = factory()
            _SESSIONS[key] = holder
        holder.refs += 1


def _release_preloaded(key: tuple) -> None:
    """放掉一个"提前加载"的引用，没人再用时才真正释放权重。"""
    with _CACHE_LOCK:
        holder = _SESSIONS.get(key)
        if holder is None:
            return
        holder.refs -= 1
        if holder.refs <= 0:
            _SESSIONS.pop(key, None)


def _drop_session(key: tuple) -> None:
    with _CACHE_LOCK:
        holder = _SESSIONS.get(key)
        if holder is not None and holder.refs <= 0:
            _SESSIONS.pop(key, None)


def active_sessions() -> int:
    """当前存活的推理会话数量，供测试与诊断使用。"""
    with _CACHE_LOCK:
        return len(_SESSIONS)


# --------------------------------------------------------------------------- 批处理执行器
class _Runner:
    """一组相同预处理配置共用的批处理执行器，背后是共享的推理会话。"""

    def __init__(self, session_key: tuple, holder: _SharedSession, max_batch: int, wait_s: float, name: str) -> None:
        self.session_key = session_key
        self.holder = holder
        self.refs = 0
        self._buffer: np.ndarray | None = None       # 批次输入缓冲，由领队线程独占使用
        holder.refs += 1
        # 模型的批次维是固定的时候只能一张一张推，批上限强制为 1
        self.executor = BatchExecutor(self._run_batch, max_batch=max_batch if self.dynamic_batch else 1,
                                      wait_s=wait_s, name=name)

    @property
    def spec(self) -> dict:
        return self.holder.spec

    @property
    def dynamic_batch(self) -> bool:
        return bool(self.holder.spec.get("dynamic_batch", True))

    def _assemble(self, items: list) -> np.ndarray:
        """把提交的条目拼成一个批次输入。

        条目是 ``(缓冲区, 行号)``。同一个节点一次提交的若干路输入共用一块连续缓冲，
        此时整批可以直接拿去推理，一次拷贝都不用；跨线程合批时才需要汇集到一起。
        """
        buf, _ = items[0]
        if all(b is buf for b, _ in items) and [i for _, i in items] == list(range(len(items))) \
                and buf.shape[0] == len(items):
            return buf
        shape = (len(items),) + buf.shape[1:]
        out = self._buffer
        if out is None or out.shape != shape or out.dtype != buf.dtype:
            out = self._buffer = np.empty(shape, dtype=buf.dtype)
        for k, (b, i) in enumerate(items):
            out[k] = b[i]
        return out

    def _run_batch(self, items: list):
        x = self._assemble(items)
        try:
            outs = self.holder.session.run(None, {self.spec["name"]: x})
        except Exception as e:
            raise NodeError(f"推理失败：模型期望输入 {describe_shape(self.spec['shape'])}，"
                            f"实际送入 {describe_shape(list(x.shape))}。"
                            f"请检查节点的宽度、高度、颜色、张量布局是否与模型一致。原始错误：{e}") from e
        n = len(items)
        return [(_slice_outputs(outs, i, n), n) for i in range(n)]

    def close(self) -> None:
        self.executor.close()
        self._buffer = None
        self.holder.refs -= 1
        if self.holder.refs <= 0:
            _drop_session(self.session_key)
        self.holder = None


def _slice_outputs(outs, i: int, n: int) -> list:
    """把批量输出按第一维拆给第 i 个输入，保留批次维，使后处理与单张推理完全一致。

    第一维不等于批大小的输出（例如与批无关的常量）原样给每一个调用者。
    """
    row = []
    for o in outs:
        a = np.asarray(o)
        row.append(a[i:i + 1] if a.ndim >= 1 and a.shape[0] == n else a)
    return row


def _acquire_runner(key: tuple, factory) -> _Runner:
    with _CACHE_LOCK:
        runner = _EXECUTORS.get(key)
        if runner is None:
            runner = factory()
            _EXECUTORS[key] = runner
        runner.refs += 1
        return runner


def _release_runner(key: tuple) -> None:
    with _CACHE_LOCK:
        runner = _EXECUTORS.get(key)
        if runner is None:
            return
        runner.refs -= 1
        if runner.refs <= 0:
            _EXECUTORS.pop(key, None)
            runner.close()


def active_runners() -> int:
    """当前存活的批处理执行器数量，供测试与诊断使用。"""
    with _CACHE_LOCK:
        return len(_EXECUTORS)


MAX_INPUTS = 8

_POOL: "ThreadPoolExecutor | None" = None
_POOL_LOCK = threading.Lock()


def _preprocess_pool() -> "ThreadPoolExecutor":
    """多路输入时用来并行做预处理的线程池，按需创建、全局共用。"""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=min(MAX_INPUTS, (os.cpu_count() or 4)),
                                       thread_name_prefix="dl-pre")
        return _POOL


def _input_port_name(i: int) -> str:
    return "image" if i == 0 else f"image{i + 1}"


def _output_port_name(base: str, i: int) -> str:
    return base if i == 0 else f"{base}{i + 1}"


class _OnnxBase(Node):
    """共享的模型加载、预处理与批量推理。

    节点可以设置多个图像输入。一次运行里，所有已连接输入的图像会被拼成一个批次做一次推理，
    再按来源把输出拆回各自的端口。

    图像不一定同时到达（例如每路相机一个流程、各自被 PLC 触发）。这种情况下给这些节点填上
    相同的「合批分组」，它们就共用一个批处理器：先到的在等待窗口里等后到的，凑够「批次上限」
    或窗口到期就执行，没赶上的留给下一批，谁都不会被永远卡住。
    """

    #: 每个输入各自对应的输出端口，子类覆盖
    per_input_outputs: list[tuple[str, DataType]] = []

    #: 改了这些参数就该重新预加载（它们决定加载哪个模型、加载到哪里）
    preload_params = ("model_path", "provider", "device_id", "session_scope")

    params = [Param("model_path", "", "file", filter="ONNX models (*.onnx)"),
              Param("input_count", 1, "int", min=1, max=MAX_INPUTS,
                    description="图像输入的个数。多个输入会被合并成一个批次，一次推理完成"),
              Param("max_batch", 8, "int", min=1, max=64,
                    description="一个批次最多几张图。模型的批次维必须是动态的才能大于 1"),
              Param("wait_ms", 0.0, "float", min=0, max=1000,
                    description="等其它流程/节点的图像一起合批的时间。只有填了合批分组才会生效："
                                "没有分组时队列里不可能出现第二个提交者，等待只会白白增加节拍"),
              Param("width", 224, "int", min=1, max=8192), Param("height", 224, "int", min=1, max=8192),
              Param("letterbox", False, "bool", description="Keep aspect ratio, pad to size"),
              Param("color", "rgb", "enum", choices=["rgb", "bgr", "gray"]),
              Param("scale", 1 / 255.0, "float", min=0, max=1000, description="Multiply pixels by this"),
              Param("mean", "0,0,0", "string", description="Per-channel mean (after scaling)"),
              Param("std", "1,1,1", "string", description="Per-channel std (after scaling)"),
              Param("layout", "NCHW", "enum", choices=["NCHW", "NHWC"]),
              Param("provider", "auto", "enum", choices=["auto", "cpu", "cuda", "tensorrt"],
                    description="推理后端。auto 按 TensorRT → CUDA → CPU 挑第一个能用的；"
                                "想确认实际跑在哪里，看日志里的「使用推理后端」一行"),
              Param("device_id", 0, "int", min=0, max=15,
                    description="用哪一块显卡（多卡时有意义）。CPU 后端会忽略这个参数"),
              Param("session_scope", "auto", "enum", choices=["auto", "shared", "exclusive"], advanced=True,
                    description="权重加载几份。shared=全软件共用一份；exclusive=按合批分组/节点各加载一份，"
                                "GPU 上可以用各自的流并行，代价是显存成倍；"
                                "auto（默认）在 CPU 上共享、在 GPU 上按合批分组/节点各一份"),
              Param("batch_group", "", "string", advanced=True,
                    description="留空（默认）每个节点独立推理，互不排队；填相同的名字则这些节点共用一个"
                                "批处理器，可跨节点跨流程合批——这是让不同时刻到达的图像凑成一批的唯一途径"),
              Param("timeout_s", 30.0, "float", min=0.1, max=600, advanced=True,
                    description="等待批处理结果的超时时间")]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._runner: _Runner | None = None
        self._runner_key: tuple | None = None
        self._failed_exclusive: tuple | None = None   # 独占会话失败过的键，不再反复重试
        self._preloaded_key: tuple | None = None      # 已经提前加载好的会话键
        self._rebuild_ports()

    # ---- 动态端口 ----
    @classmethod
    def build_ports(cls, count: int) -> tuple[list[Port], list[Port]]:
        count = max(1, min(MAX_INPUTS, int(count)))
        ins = [Port(_input_port_name(i), DataType.IMAGE, optional=i > 0,
                    description="第一个输入" if i == 0 else f"第 {i + 1} 个输入，未连接时该路输出为空")
               for i in range(count)]
        outs = [Port(_output_port_name(base, i), dtype) for i in range(count) for base, dtype in cls.per_input_outputs]
        outs += [Port("infer_ms", DataType.FLOAT, description="本次推理耗时，整批共用"),
                 Port("batch_size", DataType.INT, description="本次推理实际的批大小")]
        return ins, outs

    def _rebuild_ports(self) -> None:
        self.inputs, self.outputs = self.build_ports(self.get("input_count", 1))

    def on_param_changed(self, name: str, value) -> None:
        if name == "input_count":
            self._rebuild_ports()

    @property
    def input_count(self) -> int:
        return len(self.inputs)

    # ---- 会话与批处理执行器 ----
    def _batch_owner(self) -> str:
        """批处理器的归属：默认一个节点一个，填了合批分组的节点共用一个。"""
        group = str(self.get("batch_group")).strip()
        return f"group:{group}" if group else f"node:{self.id}"

    def _wait_s(self) -> float:
        """等待窗口。没填合批分组时固定为 0。

        同一个节点的多路输入是一次提交进去的，同一个流程里的节点又是顺序执行的，
        所以没有分组时队列里不可能出现第二个提交者，等下去只会白白增加节拍。
        """
        if not str(self.get("batch_group")).strip():
            return 0.0
        return max(0.0, float(self.get("wait_ms"))) / 1000.0

    def _preprocess_key(self) -> tuple:
        """批处理器的归属。

        默认每个节点独占一个批处理器，只有本节点的多路输入会合批，不同节点之间互不排队。
        填了合批分组名的节点才共用一个批处理器，从而能跨节点、跨流程合批（它们之间的推理会
        排队），这也是让不同时刻到达的图像凑成一批的唯一途径。
        """
        pre = (int(self.get("width")), int(self.get("height")), bool(self.get("letterbox")), self.get("color"),
               float(self.get("scale")), self.get("mean"), self.get("std"), self.get("layout"))
        return (pre, int(self.get("max_batch")), round(self._wait_s(), 4), self._batch_owner())

    def _resolve_providers(self) -> tuple[list[str], bool]:
        """定出本次要尝试的推理后端，并告诉调用方是否会落在 GPU 上。

        get_available_providers() 列的是编译进包里的后端，不代表运行时真能加载；
        本进程里已经失败过的后端记在 _FAILED_PROVIDERS 里，不再重复尝试。
        """
        try:
            import onnxruntime as ort
        except ImportError:                       # 真正的报错留给 _create_session
            return ["CPUExecutionProvider"], False
        avail = [p for p in ort.get_available_providers() if p not in _FAILED_PROVIDERS]
        prov = str(self.get("provider"))
        if prov == "auto":
            order = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
            providers = [p for p in order if p in avail]
        else:
            want = _PROVIDER_NAMES[prov]
            providers = [want] if want in avail else []
        if any(p in _GPU_PROVIDERS for p in providers):
            _preload_gpu_runtime(ort)             # 先把 CUDA/cuDNN 拉进来，否则探测必然失败
            providers = [p for p in providers if p not in _GPU_PROVIDERS or _provider_loadable(ort, p)]
        providers = providers or ["CPUExecutionProvider"]
        return providers, providers[0] != "CPUExecutionProvider"

    def _session_owner(self, on_gpu: bool) -> str:
        """推理会话（权重）的归属：空字符串＝全软件共用一份。

        CPU 上共享永远更划算：权重只占一份内存，而共用会话并不会让推理互相阻塞
        （onnxruntime 的 Run 本身线程安全），限制并行度的是核心数而不是会话。
        GPU 上则跟着批处理器走：能合批的节点共用一份权重（它们本来就要排队合批），
        不能合批的节点各加载一份，才有机会用各自的 CUDA 流并行，代价是显存成倍。
        """
        scope = str(self.get("session_scope"))
        if scope == "shared":
            return ""
        if scope == "exclusive":
            return self._batch_owner()
        return self._batch_owner() if on_gpu else ""

    def _session_key(self, path: str) -> tuple:
        providers, on_gpu = self._resolve_providers()
        device = int(self.get("device_id")) if on_gpu else -1
        return (path, os.path.getmtime(path), str(self.get("provider")), device, self._session_owner(on_gpu))

    def _ensure_runner(self) -> _Runner:
        path = paths.resolve(self.get("model_path"))
        if not path:
            raise NodeError("未设置模型文件")
        if not os.path.isfile(path):
            raise NodeError(f"模型文件不存在：{path}")
        skey = self._session_key(path)
        if skey[-1] and skey == self._failed_exclusive:
            skey = skey[:-1] + ("",)          # 这套配置上次独占就失败了，直接用共享的，别每次运行都重试
        try:
            holder = _get_or_create_session(skey, lambda: self._create_session(path))
        except Exception as e:
            if not skey[-1]:
                raise
            # 独占一份权重失败，多半是显存不够：退回共享会话，宁可慢一点也不要整条流程挂掉
            log.warning("%s：按独占方式加载模型失败（%s），改为与其它节点共用同一份权重", self.name, e)
            self._failed_exclusive = skey
            skey = skey[:-1] + ("",)
            holder = _get_or_create_session(skey, lambda: self._create_session(path))
        self.adopt_model_spec(holder.spec)      # 模型把输入形状写死时，按它修正预处理参数
        key = skey + self._preprocess_key()
        if self._runner_key == key and self._runner is not None:
            return self._runner
        self._release_runner()
        # 只在配置变化、真正要建执行器时提示一次，不在每次运行时刷屏
        if int(self.get("max_batch")) > 1 and not holder.spec.get("dynamic_batch", True):
            log.warning("%s：模型的批次维固定，无法合批，批上限设置不会生效（输入形状 %s）；"
                        "要用上合批需要把模型导出成批次维动态的", self.name, describe_shape(holder.spec["shape"]))
        try:
            self._runner = _acquire_runner(key, lambda: _Runner(
                skey, holder, int(self.get("max_batch")), self._wait_s(),
                name=os.path.basename(path)))
        except Exception:
            _drop_session(skey)
            raise
        self._runner_key = key
        return self._runner

    def adopt_model_spec(self, spec: dict) -> list[str]:
        """模型固定了输入形状时按它修正参数，并返回改动说明。

        形状写死的情况下任何别的取值都会被 onnxruntime 直接拒绝，所以这里的修正不存在歧义；
        形状是动态的则完全听节点参数的。
        """
        changed: list[str] = []
        for name, want, label in (("height", spec.get("height"), "高度"), ("width", spec.get("width"), "宽度")):
            if want and int(self.get(name)) != int(want):
                self.set(name, int(want))
                changed.append(f"{label}→{want}")
        if spec.get("layout") and self.get("layout") != spec["layout"]:
            self.set("layout", spec["layout"])
            changed.append(f"张量布局→{spec['layout']}")
        ch = spec.get("channels")
        if ch == 1 and self.get("color") != "gray":
            self.set("color", "gray")
            changed.append("颜色→gray")
        elif ch in (3, 4) and self.get("color") == "gray":
            self.set("color", "rgb")
            changed.append("颜色→rgb")
        if changed:
            log.info("%s：模型输入为 %s，已自动调整 %s", self.name, describe_shape(spec["shape"]), "，".join(changed))
        return changed

    def _create_session(self, path: str) -> _SharedSession:
        # 注意：get_available_providers() 列的是编译进包里的后端，不代表运行时真能加载。
        # 装了 GPU 版但缺 CUDA 运行库时，CUDA 后端会加载失败并静默回退到 CPU，
        # 所以失败过的后端记在 _FAILED_PROVIDERS 里不再重试，并把实际用的后端报出来。
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise NodeError("未安装推理运行时。CPU 用 pip install -e \".[cpu]\"，"
                            "显卡用 pip install -e \".[gpu]\"（两者互斥，只装一个）") from e
        providers, on_gpu = self._resolve_providers()
        if any(p in _GPU_PROVIDERS for p in providers):
            _preload_gpu_runtime(ort)                  # 先把 CUDA/cuDNN 拉进来，否则 GPU 后端会加载失败
        device = int(self.get("device_id"))
        # 显卡编号通过 provider 选项传给 CUDA / TensorRT；CPU 后端没有这个概念
        request = [(p, {"device_id": device}) if p in _GPU_PROVIDERS else p for p in providers]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            session = ort.InferenceSession(path, so, providers=request)
        except Exception as e:
            raise NodeError(f"加载模型失败：{os.path.basename(path)}：{e}") from e
        active = list(session.get_providers())
        for p in providers:
            if p != "CPUExecutionProvider" and p not in active:
                _FAILED_PROVIDERS.add(p)
                why = _why_provider_failed(ort, p)
                # Windows 的加载错误里不一定带库名，所以提示按后端给，不靠关键字猜
                if p == "TensorrtExecutionProvider":
                    hint = "；缺 TensorRT 本体，只想用显卡的话把推理后端改成 cuda 或 auto"
                elif "cudnn" in why:
                    hint = "；缺 cuDNN，可装 pip install nvidia-cudnn-cu12"
                else:
                    hint = "；缺 CUDA 运行库，pip install -e \".[gpu]\" 会把它们一起装上"
                log.warning("推理后端 %s 无法加载（%s）%s，本次运行不再尝试",
                            p, why or "通常是缺少对应版本的 CUDA/cuDNN 运行库", hint)
        prov = str(self.get("provider"))
        where = active[0] if active else "?"
        if active and active[0] in _GPU_PROVIDERS:
            where = f"{active[0]}（{device} 号卡）"
        if prov != "auto" and _PROVIDER_NAMES[prov] not in active:
            want = _PROVIDER_NAMES[prov]
            if want not in ort.get_available_providers():
                # 根本没编译进这个 onnxruntime 包里——装的多半是 CPU 版
                reason = "当前安装的 onnxruntime 里没有这个后端"
            else:
                reason = "这个后端在本机加载失败（缺少对应的运行库）"
            log.warning("%s：请求的推理后端 %s 不可用（%s），实际使用 %s。%s",
                        self.name, prov, reason, where, _PROVIDER_FIX.get(want, ""))
            _warn_if_both_wheels_installed()
        else:
            log.info("%s：模型 %s 使用推理后端 %s", self.name, os.path.basename(path), where)
        return _SharedSession(session, read_input_spec(session), active)

    def _release_runner(self) -> None:
        if self._runner_key is not None:
            _release_runner(self._runner_key)
        self._runner, self._runner_key = None, None

    def on_enabled_changed(self, enabled: bool) -> None:
        """禁用就把权重从显存/内存里放掉；重新启用时由界面触发预加载。"""
        if not enabled:
            self.unload()

    def unload(self) -> None:
        """释放本节点占用的推理资源。权重是共享的，只有没人再用时才真正被释放。"""
        self._release_runner()
        if self._preloaded_key is not None:
            stale, self._preloaded_key = self._preloaded_key, None
            _release_preloaded(stale)

    def preload(self) -> None:
        """把模型提前加载进内存/显存，让第一次运行不用再等几秒的加载时间。

        只预热共享的推理会话（真正重的那一步），不碰本节点的批处理器状态——
        界面是在后台线程调用这个方法的，而此刻流程可能正在运行，动 runner 会把正在用的执行器关掉。
        """
        path = paths.resolve(self.get("model_path"))
        if not path or not os.path.isfile(path) or not self.enabled:
            return                                    # 禁用的节点不占显存
        skey = self._session_key(path)
        if skey == self._preloaded_key:
            return                                    # 这套配置已经加载过了
        t0 = time.perf_counter()
        _acquire_preloaded(skey, lambda: self._create_session(path))
        stale, self._preloaded_key = self._preloaded_key, skey
        if stale is not None and stale != skey:
            _release_preloaded(stale)                 # 换了模型/后端/显卡：上一份没人用就放掉
        if not self.enabled:                          # 加载过程中被禁用了：别把它留在显存里
            self.unload()
            return
        log.info("%s：模型 %s 已就绪（加载耗时 %.0f ms）",
                 self.name, os.path.basename(path), (time.perf_counter() - t0) * 1000)

    def setup(self, ctx=None):
        if self.get("model_path"):
            self._ensure_runner()

    def teardown(self):
        self.unload()

    @property
    def supports_batching(self) -> bool:
        return self._runner is not None and self._runner.dynamic_batch

    # ---- 预处理 ----
    def _to_canvas(self, img: Image):
        """缩放或等比填充到网络输入尺寸，返回 uint8 画布（彩色保持 BGR）与还原坐标用的元信息。"""
        w, h = int(self.get("width")), int(self.get("height"))
        data = as_gray(img) if self.get("color") == "gray" else as_bgr(img)
        meta = {"scale": 1.0, "pad": (0, 0), "orig": (img.width, img.height)}
        if self.get("letterbox"):
            r = min(w / img.width, h / img.height)
            nw, nh = int(round(img.width * r)), int(round(img.height * r))
            resized = cv2.resize(data, (nw, nh))
            canvas = np.full((h, w) + (() if resized.ndim == 2 else (resized.shape[2],)), 114, dtype=np.uint8)
            dx, dy = (w - nw) // 2, (h - nh) // 2
            canvas[dy:dy + nh, dx:dx + nw] = resized
            meta["scale"], meta["pad"] = r, (dx, dy)
            return canvas, meta
        meta["scale"] = (w / img.width, h / img.height)
        return cv2.resize(data, (w, h)), meta

    def _norm_params(self, channels: int):
        scale = float(self.get("scale"))
        mean = np.array(parse_floats(self.get("mean"), channels, 0.0), dtype=np.float32)
        std = np.array(parse_floats(self.get("std"), channels, 1.0), dtype=np.float32)
        return scale, mean, std

    def _preprocess(self, img: Image):
        canvas, meta = self._to_canvas(img)
        c = 1 if canvas.ndim == 2 else canvas.shape[2]
        scale, mean, std = self._norm_params(c)
        w, h = int(self.get("width")), int(self.get("height"))
        rgb = self.get("color") == "rgb"
        # 快速路径：OpenCV 把缩放、减均值、乘系数、通道交换、转 NCHW 合成一次 C++ 调用，
        # 全程释放 GIL，多路输入并行预处理时比逐步的 numpy 运算快近一个数量级。
        # 它的 scalefactor 只能是标量，所以要求各通道 std 相同。
        if self.get("layout") == "NCHW" and scale != 0 and np.all(std == std[0]) and std[0] != 0:
            alpha = float(scale / std[0])
            mean_cv = tuple(float(v) for v in (mean / scale))      # OpenCV 先减均值再乘系数，均值是原始像素量纲
            blob = cv2.dnn.blobFromImage(canvas, alpha, (w, h), mean_cv if c > 1 else float(mean_cv[0]),
                                         swapRB=rgb, crop=False)
            return blob, meta
        # 通用路径：NHWC 布局或各通道 std 不同
        if rgb and canvas.ndim == 3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if canvas.ndim == 2:
            canvas = canvas[:, :, None]
        alpha = (scale / std).astype(np.float32)
        beta = (-mean / std).astype(np.float32)
        x = canvas.astype(np.float32)
        if np.all(alpha == alpha[0]) and not beta.any():
            if alpha[0] != 1.0:
                x *= float(alpha[0])
        else:
            x *= alpha
            x += beta
        if self.get("layout") == "NCHW":
            x = x.transpose(2, 0, 1)
        return np.ascontiguousarray(x[None]), meta

    # ---- 批量推理 ----
    def _gather_images(self, inputs: dict) -> list:
        imgs = []
        for i in range(self.input_count):
            v = inputs.get(_input_port_name(i))
            if v is None:
                imgs.append(None)
            else:
                imgs.append(require_image(v, _input_port_name(i)))
        if all(im is None for im in imgs):
            raise NodeError("没有任何已连接的输入图像")
        return imgs

    def _infer_batch(self, images: list):
        """对所有非空图像做一次批量推理。

        返回 ``(每路输出字典, 每路 meta 字典, 耗时毫秒, 实际批大小)``，键是输入序号。
        """
        runner = self._ensure_runner()
        order = [i for i, im in enumerate(images) if im is not None]
        if len(order) == 1:
            blob, meta = self._preprocess(images[order[0]])
            buf, metas = blob, {order[0]: meta}
        else:
            # 并行预处理，并直接写进同一块连续缓冲：拷贝也在并行区完成，
            # 之后整批推理不必再汇集一次。
            pool = _preprocess_pool()
            done = list(pool.map(lambda i: self._preprocess(images[i]), order))
            buf = np.empty((len(order),) + done[0][0].shape[1:], dtype=done[0][0].dtype)
            list(pool.map(lambda k: buf.__setitem__(k, done[k][0][0]), range(len(order))))
            metas = {i: meta for i, (_, meta) in zip(order, done)}
        t0 = time.perf_counter()
        results = runner.executor.submit([(buf, k) for k in range(len(order))], timeout=float(self.get("timeout_s")))
        ms = (time.perf_counter() - t0) * 1000
        outs = {i: results[k][0] for k, i in enumerate(order)}
        batch = max((results[k][1] for k in range(len(order))), default=0)
        return outs, metas, ms, batch

    def _empty_result(self) -> dict:
        """未连接的输入对应的输出，全部为 None，保持端口语义稳定。"""
        return {base: None for base, _ in self.per_input_outputs}

    def process(self, ctx, inputs):
        images = self._gather_images(inputs)
        outs, metas, ms, batch = self._infer_batch(images)
        result: dict = {"infer_ms": ms, "batch_size": batch}
        for i in range(self.input_count):
            if i in outs:
                # 每一路的叠加层都画出来并标记来源，图像窗口按当前查看的输入过滤
                with ctx.overlay_group(_input_port_name(i)):
                    one = self.decode(ctx, outs[i], metas[i], images[i])
            else:
                one = self._empty_result()
            for base, _ in self.per_input_outputs:
                result[_output_port_name(base, i)] = one.get(base)
        return result

    def decode(self, ctx, outs: list, meta: dict, img: Image) -> dict:
        """把一路的原始输出转成该路的输出字典，子类实现。"""
        raise NotImplementedError


@register
class OnnxInference(_OnnxBase):
    type_id = "dl.onnx"
    category = "Deep Learning"
    label = "ONNX Inference"
    description = "Run any ONNX model; outputs the raw tensors for downstream post-processing (e.g. a script node)."
    color = _COLOR
    per_input_outputs = [("outputs", DataType.LIST), ("output0", DataType.TENSOR)]
    inputs, outputs = Node.inputs, Node.outputs      # 占位，下面用 build_ports 覆盖

    def decode(self, ctx, outs, meta, img):
        return {"outputs": outs, "output0": outs[0]}


@register
class OnnxClassifier(_OnnxBase):
    type_id = "dl.onnx_classifier"
    category = "Deep Learning"
    label = "ONNX Classifier"
    description = "Image classification: softmax over the first output, labels from a text file."
    color = _COLOR
    per_input_outputs = [("class_id", DataType.INT), ("label", DataType.STRING),
                         ("score", DataType.FLOAT), ("probs", DataType.LIST)]
    params = _OnnxBase.params + [Param("labels_path", "", "file", filter="Text (*.txt)"),
                                 Param("softmax", True, "bool")]

    def decode(self, ctx, outs, meta, img):
        logits = np.asarray(outs[0], dtype=np.float32).reshape(-1)
        if self.get("softmax"):
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
        else:
            probs = logits
        cid = int(probs.argmax())
        labels = _load_labels(self.get("labels_path"))
        label = labels[cid] if cid < len(labels) else str(cid)
        ctx.add_overlay(Overlay.text(8, 20, f"{label} {float(probs[cid]):.3f}", "#ffff00"))
        return {"class_id": cid, "label": label, "score": float(probs[cid]), "probs": probs.tolist()}


@register
class OnnxDetector(_OnnxBase):
    type_id = "dl.onnx_detector"
    category = "Deep Learning"
    label = "ONNX Detector (YOLO)"
    description = "YOLOv5/v8/v11-style detector: decodes [1,84,N] or [1,N,85] outputs, applies NMS."
    color = _COLOR
    per_input_outputs = [("detections", DataType.LIST), ("count", DataType.INT),
                         ("boxes", DataType.LIST), ("best_label", DataType.STRING)]
    params = [p if p.name not in ("width", "height", "letterbox") else
              Param(p.name, {"width": 640, "height": 640, "letterbox": True}[p.name], p.kind, min=p.min, max=p.max,
                    description=p.description, advanced=p.advanced)
              for p in _OnnxBase.params] + [
        Param("labels_path", "", "file", filter="Text (*.txt)"),
        Param("conf", 0.25, "float", min=0, max=1, step=0.01), Param("iou", 0.45, "float", min=0, max=1, step=0.01)]

    def decode(self, ctx, outs, meta, img):
        pred = np.asarray(outs[0], dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            raise NodeError(
                f"无法解析检测输出：期望 [批, 属性, 候选框] 或 [批, 候选框, 属性] 的张量，"
                f"实际是 {describe_shape(list(np.shape(outs[0])))}。"
                f"该模型可能不是 YOLO 系列的检测模型；可以改用 ONNX 推理节点，再接 Python 脚本节点自行后处理。")
        # 属性维（4 个框坐标 + 类别分数）至少 5 个；v8 布局为 [84, N]，v5 为 [N, 85]
        if pred.shape[1] < 5 or (pred.shape[0] >= 5 and pred.shape[0] < pred.shape[1]):
            pred = pred.T
        if pred.shape[1] < 5:
            raise NodeError(
                f"无法解析检测输出：每个候选框至少要有 4 个坐标加 1 个类别分数，"
                f"实际只有 {pred.shape[1]} 个属性（输出形状 {describe_shape(list(np.shape(outs[0])))}）。")
        if pred.shape[1] > 5 and pred.shape[1] - 5 >= 1 and self._looks_v5(pred):
            obj = pred[:, 4:5]
            cls_scores = pred[:, 5:] * obj
        else:
            cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(1)
        scores = cls_scores.max(1)
        keep = scores >= float(self.get("conf"))
        boxes_xywh, scores, cls_ids = pred[keep, :4], scores[keep], cls_ids[keep]
        if self.get("letterbox"):
            r, (dx, dy) = meta["scale"], meta["pad"]
            sx = sy = 1.0 / r
            ox, oy = dx, dy
        else:
            sx, sy = 1.0 / meta["scale"][0], 1.0 / meta["scale"][1]
            ox = oy = 0
        rects = []
        for (cx, cy, w, h) in boxes_xywh:
            x0 = (cx - w / 2 - ox) * sx
            y0 = (cy - h / 2 - oy) * sy
            rects.append([int(x0), int(y0), int(w * sx), int(h * sy)])
        idx = cv2.dnn.NMSBoxes(rects, scores.tolist(), float(self.get("conf")), float(self.get("iou"))) if rects else []
        idx = np.asarray(idx).reshape(-1)
        labels = _load_labels(self.get("labels_path"))
        dets = []
        for i in idx:
            x, y, w, h = rects[i]
            cid = int(cls_ids[i])
            name = labels[cid] if cid < len(labels) else str(cid)
            dets.append({"x": x, "y": y, "w": w, "h": h, "score": float(scores[i]), "class_id": cid, "label": name})
            ctx.add_overlay(Overlay.rect(Rect(x, y, w, h), "#00ff00", f"{name} {scores[i]:.2f}"))
        dets.sort(key=lambda d: -d["score"])
        return {"detections": dets, "count": len(dets),
                "boxes": [Rect(d["x"], d["y"], d["w"], d["h"]) for d in dets],
                "best_label": dets[0]["label"] if dets else ""}

    @staticmethod
    def _looks_v5(pred: np.ndarray) -> bool:
        # v5: column 4 is objectness in [0,1] and there are 85 columns for COCO; v8 has 84 (no objectness).
        return pred.shape[1] in (85, 6) or (pred.shape[1] > 6 and float(pred[:, 4].max()) <= 1.0 and float(pred[:, 5:].max()) <= 1.0 and pred.shape[1] % 2 == 1)


for _cls in (OnnxInference, OnnxClassifier, OnnxDetector):
    _cls.inputs, _cls.outputs = _cls.build_ports(1)      # 类级端口用于节点库展示，实例按 input_count 重建


_SCRIPT_TEMPLATE = '''"""自定义节点代码。可用对象：np, cv2, Image, Overlay, Rect, Point, Line, Circle。

def setup(state): 可选，流程启动时执行一次。
def process(ctx, inputs, params, state) -> dict: 每张图执行一次。
   inputs: {"image": Image|None, "in1".."in4": 任意}
   返回 {"image": Image, "out1".."out4": 任意}
"""
def process(ctx, inputs, params, state):
    img = inputs["image"]
    if img is None:
        return {}
    gray = img.data if img.is_gray else cv2.cvtColor(img.data, cv2.COLOR_BGR2GRAY)
    state["runs"] = state.get("runs", 0) + 1
    ctx.add_overlay(Overlay.text(8, 20, f"script run {state['runs']}", "#ffff00"))
    return {"image": img.derive(gray), "out1": float(gray.mean()), "out2": state["runs"]}
'''


@register
class PythonScript(Node):
    type_id = "script.python"
    category = "Script"
    label = "Python Script"
    description = "Run user Python code as a node (the escape hatch for custom DL/RL logic)."
    color = "#ef6c00"
    inputs = [Port("image", DataType.IMAGE, optional=True)] + [Port(f"in{i}", DataType.ANY, optional=True) for i in range(1, 5)]
    outputs = [Port("image", DataType.IMAGE)] + [Port(f"out{i}", DataType.ANY) for i in range(1, 5)]
    params = [Param("code", _SCRIPT_TEMPLATE, "code")]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._ns: dict | None = None
        self._compiled_src = None

    def _namespace(self):
        src = self.get("code")
        if self._ns is None or self._compiled_src != src:
            from ..core.types import Circle, Line, Point
            ns = {"np": np, "cv2": cv2, "Image": Image, "Overlay": Overlay, "Rect": Rect,
                  "Point": Point, "Line": Line, "Circle": Circle, "NodeError": NodeError, "__name__": f"script_{self.id}"}
            exec(compile(src, f"<script {self.name}>", "exec"), ns)
            if "process" not in ns or not callable(ns["process"]):
                raise NodeError("脚本必须定义 process(ctx, inputs, params, state)")
            self._ns, self._compiled_src = ns, src
            if callable(ns.get("setup")):
                ns["setup"](self.state)
        return self._ns

    def on_param_changed(self, name, value):
        if name == "code":
            self._ns = None

    def setup(self, ctx=None):
        self._namespace()

    def process(self, ctx, inputs):
        ns = self._namespace()
        out = ns["process"](ctx, inputs, dict(self.values), self.state)
        return out or {}

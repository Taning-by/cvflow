"""命令行入口。

  cvflow gui [方案.json]                 打开桌面程序
  cvflow run 方案.json [-f main] [-n 5]  在终端运行流程 N 次
  cvflow serve 方案.json                 无界面生产模式：连接设备、启动全部流程
  cvflow nodes [--json]                  列出已注册的节点类型
  cvflow validate 方案.json              检查方案是否有问题
  cvflow new 方案.json                   新建一个空方案
  cvflow shortcut [方案.json]            在桌面创建快捷方式（点击图标打开软件）
  cvflow cameras                         看看各条取图路线分别认不认得本机的相机
  cvflow gpu [模型.onnx]                 自检推理环境：装了什么后端、能不能加载、实际跑在哪、占多少显存
                                         加 --require-gpu 时，CUDA 没跑起来就返回非零退出码
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path


def _load(path: str, plugins: list[str]):
    from .core import Solution, registry
    registry.load_builtins()
    for p in plugins:
        registry.load_plugin_dir(p)
    return Solution.load(path)


def cmd_nodes(args) -> int:
    from .core import registry
    registry.load_builtins()
    for p in args.plugins:
        registry.load_plugin_dir(p)
    if args.json:
        print(json.dumps([c.describe() for c in registry.all()], indent=2, ensure_ascii=False))
        return 0
    from .ui.i18n import tr
    for cat, classes in registry.categories().items():
        print(f"{tr(cat)}")
        for c in classes:
            ins = ", ".join(f"{p.name}:{p.dtype.value}" for p in c.inputs)
            outs = ", ".join(f"{p.name}:{p.dtype.value}" for p in c.outputs)
            print(f"  {c.type_id:28s} {tr(c.label):18s} 输入[{ins}] 输出[{outs}]")
    print(f"\n共 {len(registry.all())} 个节点类型")
    return 0


def _nvidia_smi(query: str, extra: list[str] | None = None) -> list[str]:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits", *(extra or [])],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []


def _vram_mib(pid: int) -> int | None:
    """本进程占用的显存（MiB）。

    Windows 的 WDDM 驱动不支持按进程查询，这一列会是 ``[N/A]``；WSL 里干脆没有这张表。
    查不到就返回 None，让调用方改用整卡的已用显存。
    """
    rows = _nvidia_smi("compute-apps=pid,used_memory")
    if not rows:
        return None
    for row in rows:
        parts = [x.strip() for x in row.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return int(parts[1]) if parts[1].isdigit() else None
    return 0                                   # 本进程还没用显存


def _device_vram_mib(index: int) -> int | None:
    """某块卡上已用的显存（含其它程序），按进程查不到时的兜底。"""
    rows = _nvidia_smi("gpu=memory.used", ["-i", str(index)])
    return int(rows[0]) if rows and rows[0].isdigit() else None


def _gpu_component_installed(ort) -> bool:
    """打包版里「GPU 推理支持」组件装没装：看 CUDA provider 那个大 DLL 在不在。

    安装程序把它和 CUDA / cuDNN 运行库划成可选组件（约 1.9 GB），不勾就不装，
    软件照常启动、推理跑 CPU。
    """
    import glob
    import os
    base = os.path.join(os.path.dirname(os.path.abspath(ort.__file__)), "capi")
    return bool(glob.glob(os.path.join(base, "*providers_cuda*")))


def _onnxruntime_outside_venv(ort) -> bool:
    """这份 onnxruntime 是不是虚拟环境外面的那一份。

    虚拟环境建的时候带了 --system-site-packages，系统里又装过 CPU 版 onnxruntime 的话，
    系统那一份会盖掉环境里的 GPU 版——而且 pip 在环境内卸不掉它（"outside environment"），
    症状是"明明装了 onnxruntime-gpu，后端里却只有 CPU"，极难自己看出来。
    """
    import os
    if sys.prefix == sys.base_prefix:                  # 没在虚拟环境里，谈不上覆盖
        return False
    where = os.path.realpath(os.path.dirname(ort.__file__))
    return not where.startswith(os.path.realpath(sys.prefix) + os.sep)


def _bench_session(session, feed, rounds: int) -> tuple[float, float, float]:
    """跑 rounds 次推理，返回（中位、最快、最慢）毫秒。

    前几次不计数：cuDNN 第一次要挑卷积算法，CPU 那边也要把权重读进缓存，
    算进去会让 GPU 看起来比实际慢得多。
    """
    for _ in range(3):
        session.run(None, feed)
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        session.run(None, feed)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[0], ts[-1]


def _bench_feed(session, size: int):
    """按模型第一个输入的形状造一份随机输入；动态维度用 batch=1 和给定的边长补上。"""
    import numpy as np
    from .operators import dl
    inp = session.get_inputs()[0]
    spec = dl.read_input_spec(session)
    shape = []
    for i, d in enumerate(inp.shape or [1, 3, size, size]):
        if isinstance(d, int) and d > 0:
            shape.append(d)
        elif i == 0:
            shape.append(1)                                   # 动态 batch
        elif spec["layout"] == "NHWC" and i == 3 or spec["layout"] == "NCHW" and i == 1:
            shape.append(3)                                   # 动态通道，按彩色算
        else:
            shape.append(size)                                # 动态高宽
    dtype = np.float16 if "float16" in inp.type else np.float32
    return {inp.name: np.random.rand(*shape).astype(dtype)}, shape


def _require_gpu_failed() -> int:
    """--require-gpu 下 CUDA 没跑起来：打一句结论并给非零退出码（安装脚本/CI 靠它判断）。"""
    print("\n✗ 要求用 GPU，但 CUDA 后端没能跑起来。按上面的提示修好再重试，"
          "完整步骤见 docs/install.md")
    return 2


def cmd_gpu(args) -> int:
    """自检推理环境：装了哪个 onnxruntime、后端能不能加载、模型实际跑在哪、占多少显存。"""
    import os
    print("== 推理运行时 ==")
    from .operators import dl
    try:
        import onnxruntime as ort
    except ImportError:
        print("  未安装 onnxruntime。" + (
            "这个打包版缺了推理运行时，请重新安装" if dl.frozen() else
            "CPU 装 pip install -e \".[cpu]\"，显卡装 pip install -e \".[gpu]\"（二选一）"))
        return 1
    print(f"  onnxruntime {ort.__version__}  {os.path.dirname(ort.__file__)}")
    if _onnxruntime_outside_venv(ort):
        print("  ⚠ 这份 onnxruntime 不在当前虚拟环境里，而是系统 site-packages 里的那一份——"
              "虚拟环境是带 --system-site-packages 建的，")
        print("    系统装过的 CPU 版会盖掉环境里的 GPU 版，而且 pip 在环境内卸不掉它"
              "（会提示 outside environment）。")
        print(f"    修法：把 {os.path.join(sys.prefix, 'pyvenv.cfg')} 里的 "
              "include-system-site-packages 改成 false，")
        print("    或者直接重跑 install.sh / install.ps1（会自动处理这种情况）")
    wheels: list[str] = []
    if dl.frozen():
        # 打包版里没有 pip 的安装记录，改看 GPU 组件的那个大 DLL 在不在
        print(f"  打包版（{'装了' if _gpu_component_installed(ort) else '没装'}「GPU 推理支持」组件）")
    try:
        from importlib.metadata import distributions
        names = {d.metadata["Name"].lower() for d in distributions() if d.metadata["Name"]}
        wheels = sorted(n for n in names if n in ("onnxruntime", "onnxruntime-gpu"))
        if not dl.frozen():
            print(f"  已安装的包：{'、'.join(wheels) or '未知'}")
        if len(wheels) > 1:
            print("  ⚠ 同时装了 CPU 版和 GPU 版，它们会互相覆盖。"
                  "请 pip uninstall -y onnxruntime onnxruntime-gpu 之后只装需要的那一个")
        if "torch" in names:
            # torch 自带一整套 CUDA/cuDNN，而 onnxruntime 发现 torch 已导入就会让位给它，
            # 版本不匹配时推理会悄悄退回 CPU。本项目故意不提供 torch 的 extra。
            print("  ⚠ 这个环境里还装了 torch。torch 自带一整套 CUDA / cuDNN，"
                  "onnxruntime 发现它已导入就会让位，")
            print("    版本不匹配时 GPU 后端会加载失败或在执行算子时降级成 CPU。"
                  "只跑 ONNX 推理的话建议把 torch 挪到另一个虚拟环境")
    except Exception:                                            # pragma: no cover
        pass

    print("\n== 后端 ==")
    avail = list(ort.get_available_providers())
    print(f"  包里编译进来的：{'、'.join(avail)}")
    if any(p in dl._GPU_PROVIDERS for p in avail):
        dl._preload_gpu_runtime(ort)      # 包里没有 GPU 后端时别预加载，否则会打一条误导人的警告
    cuda_ok = False
    for prov in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
        if prov not in avail:
            print(f"  {prov:28s} 包里没有")
            continue
        why = dl._why_provider_failed(ort, prov)
        cuda_ok = cuda_ok or (prov == "CUDAExecutionProvider" and not why)
        print(f"  {prov:28s} {'可用' if not why else '加载失败：' + why}")
        if why and prov == "TensorrtExecutionProvider":
            print("     （没装 TensorRT 本体，属正常情况：auto 会跳过它直接用 CUDA，不影响使用）")
        elif why:
            print("     " + ("（安装时没勾「GPU 推理支持」组件；重新运行一次安装程序勾上即可）"
                             if dl.frozen() else
                             "（缺 CUDA / cuDNN 运行库，pip install -e \".[gpu]\" 会把它们一起装上）"))
    if "CUDAExecutionProvider" not in avail:
        both = len(wheels) > 1
        print("  → 当前这份 onnxruntime 里没有 GPU 后端，推理只能在 CPU 上跑。")
        if dl.frozen():
            print("     这个打包版不带 GPU 后端，请重新安装并勾选「GPU 推理支持」组件")
        elif both:
            print("     原因是 CPU 版把 GPU 版覆盖了（两个包装的是同一个模块）。按下面三步修：")
        if not dl.frozen():
            if not both:
                print("     装 GPU 版：")
            print("       1) pip uninstall -y onnxruntime onnxruntime-gpu   （重复执行到两个都显示未安装）")
            print(f"       2) 确认 {os.path.dirname(ort.__file__)} 已经不存在，残留就手动删掉")
            print("       3) pip install -e \".[dev,gpu]\"")
            print("     完整步骤和常见错误：docs/install.md")

    print("\n== 显卡 ==")
    gpus = _nvidia_smi("gpu=index,name,driver_version,memory.used,memory.total")
    if not gpus:
        print("  没找到 nvidia-smi（没有 NVIDIA 显卡，或驱动没装好）")
    for row in gpus:
        idx, name, driver, used, total = [x.strip() for x in row.split(",")]
        print(f"  {idx} 号卡 {name}　驱动 {driver}　已用 {used}/{total} MiB")

    if not args.model:
        print("\n加上一个模型文件可以实测它跑在哪、占多少显存：cvflow gpu 你的模型.onnx")
        return 0 if (cuda_ok or not args.require_gpu) else _require_gpu_failed()

    print("\n== 实测加载 ==")
    from .core import paths, registry
    registry.load_builtins()
    paths.set_base_dir(None)
    node = registry.create("dl.onnx")
    node.set("model_path", str(Path(args.model).resolve()))
    node.set("provider", args.provider)
    node.set("device_id", args.device)
    pid = os.getpid()
    before, before_dev = _vram_mib(pid), _device_vram_mib(args.device)
    t0 = time.perf_counter()
    try:
        node.preload()
    except Exception as e:
        print(f"  加载失败：{e}")
        return 1
    ms = (time.perf_counter() - t0) * 1000
    holder = dl._SESSIONS.get(node._preloaded_key) if node._preloaded_key else None
    where = (holder.providers[0] if holder and holder.providers else "?")
    after, after_dev = _vram_mib(pid), _device_vram_mib(args.device)
    on_gpu = where in dl._GPU_PROVIDERS
    print(f"  实际使用的后端：{where}" + (f"（{args.device} 号卡）" if on_gpu else ""))
    print(f"  加载耗时：{ms:.0f} ms")
    if before is not None and after is not None:
        delta = after - before
        print(f"  本进程显存：{before} → {after} MiB（+{delta}）")
    elif before_dev is not None and after_dev is not None:
        delta = after_dev - before_dev
        print(f"  {args.device} 号卡已用显存：{before_dev} → {after_dev} MiB（+{delta}）"
              "　（驱动不支持按进程查询，这里是整卡数字，含其它程序）")
    else:
        delta = None
        print("  显存：这台机器查不到（没有 nvidia-smi 或驱动不支持），请手动对比 nvidia-smi 的输出")
    # 真跑一次：GPU 后端可能注册成功、执行算子时才失败，onnxruntime 会把会话悄悄重建成 CPU-only
    if on_gpu:
        try:
            feed, _ = _bench_feed(holder.session, args.size)
            holder.session.run(None, feed)
        except Exception as e:
            print(f"  试跑一次推理失败：{str(e).splitlines()[0][:160]}")
        after_run = list(holder.session.get_providers())
        if not any(p in dl._GPU_PROVIDERS for p in after_run):
            on_gpu = False
            where = after_run[0] if after_run else "?"
            print(f"  ⚠ 真跑一次之后，后端变成了 {'、'.join(after_run)}")
            print("     GPU 后端注册成功，但执行算子时失败了，onnxruntime 把会话重建成了 CPU-only——"
                  "上面 stderr 里的报错就是原因")
            print("     最常见：Could not locate cudnn_*.dll（cuDNN 的引擎子库没找到），"
                  "装的 cuDNN 版本比 onnxruntime 预加载名单新时就会这样")

    if not on_gpu:
        print("  → 跑在 CPU 上，所以显存不会变。按上面的提示修好后端即可")
    elif delta is not None and delta < 50:
        print("  ⚠ 跑在显卡上但显存几乎没涨。两种可能：看的不是同一块卡（--device 指定），"
              "或者后端注册上了但算子实际跑在 CPU 上——用 --bench 测一下就知道")

    if args.bench:
        _bench(str(Path(args.model).resolve()), holder, where, on_gpu, args)

    node.unload()
    return 0 if (on_gpu or not args.require_gpu) else _require_gpu_failed()


def _bench(model_path: str, holder, where: str, on_gpu: bool, args) -> None:
    """把当前后端和纯 CPU 各跑一遍，再看每个算子实际落在哪个后端上。

    「日志说用了 CUDA，但和 CPU 一样快」几乎都在这里露馅：get_providers() 报的是
    *注册成功* 的后端，而 onnxruntime 会把 CUDA 后端吃不下的算子静默切回 CPU，
    于是后端名字写着 CUDA、算力其实在 CPU 上。只有计时和 profiling 能分辨。
    """
    import onnxruntime as ort
    from .operators import dl
    print(f"\n== 实测推理（各跑 {args.bench} 次，取中位数）==")
    try:
        feed, shape = _bench_feed(holder.session, args.size)
    except Exception as e:
        print(f"  造不出随机输入，跳过实测：{e}")
        return
    dynamic = dl.read_input_spec(holder.session)["width"] is None
    print(f"  输入形状：{'×'.join(str(d) for d in shape)}"
          + ("（模型是动态尺寸，高宽按 --size 取，默认 640）" if dynamic else ""))
    med, lo, hi = _bench_session(holder.session, feed, args.bench)
    print(f"  {where:30s} 中位 {med:7.1f} ms（{lo:.1f} ~ {hi:.1f}）")
    if not on_gpu:
        print("  → 当前就跑在 CPU 上，没有可比的对象")
        return

    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        cpu = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        cmed, clo, chi = _bench_session(cpu, feed, args.bench)
    except Exception as e:
        # 纯 CPU 跑不起来不影响前面的结论，别让整条命令挂掉
        print(f"  （纯 CPU 的对比跑不了，跳过：{str(e).splitlines()[0][:120]}）")
        _profile_providers(model_path, holder.providers, feed, args.device)
        return
    print(f"  {'CPUExecutionProvider':30s} 中位 {cmed:7.1f} ms（{clo:.1f} ~ {chi:.1f}）")
    speedup = cmed / med if med else 0
    print(f"  → GPU 比 CPU 快 {speedup:.1f} 倍")

    _profile_providers(model_path, holder.providers, feed, args.device)

    if speedup < 1.5:
        print("\n  ⚠ 几乎没快，说明算力多半还在 CPU 上。对照上面的算子分布看：")
        print("     · CPU 那一行算子多、耗时占大头 → 模型里有 CUDA 后端吃不下的算子")
        print("       （量化过的 INT8/QDQ 模型最常见，整段会切回 CPU；NMS、某些 Resize/TopK 也会）")
        print("     · 两边都很快、差距不大 → 模型本身计算量太小，瓶颈在数据搬运不在算力")
        print("     · 想确认不是显卡选错：cvflow gpu 模型.onnx --bench --device 1")


def format_profile(events: list, runs: int) -> list[str]:
    """把 onnxruntime 的 profiling 事件汇总成「每个后端跑了几个算子、各花多少时间」。

    事件里的 dur 是微秒、而且是 runs 次的累计，这里换算成「每次推理的毫秒数」。
    """
    import collections
    count: collections.Counter = collections.Counter()
    spent: collections.Counter = collections.Counter()
    ops: dict = collections.defaultdict(collections.Counter)
    for e in events:
        if e.get("cat") != "Node":
            continue
        args = e.get("args") or {}
        prov = args.get("provider")
        if not prov:
            continue
        ms = e.get("dur", 0) / max(runs, 1) / 1000.0
        count[prov] += 1
        spent[prov] += ms
        ops[prov][args.get("op_name", "?")] += ms
    if not count:
        return []
    lines = ["", "  算子实际落在哪个后端（profiling 的绝对值偏大，它会按算子逐个同步，看的是比例）："]
    cpu = "CPUExecutionProvider"
    # 只在 CPU 反客为主时才标出来：GPU 占大头是正常的，标了反而像出了问题
    cpu_dominates = len(spent) > 1 and spent.get(cpu, 0) > sum(v for k, v in spent.items() if k != cpu)
    for prov, ms in spent.most_common():
        n = count[prov] // max(runs, 1) or count[prov]
        flag = "  ← 绝大部分时间耗在 CPU 上，GPU 没干多少活" if cpu_dominates and prov == cpu else ""
        lines.append(f"    {prov:30s} {n:4d} 个算子  {ms:7.1f} ms{flag}")
    if list(spent) == [cpu]:
        lines.append("    → 所有算子都在 CPU 上，显卡完全没参与计算")
    if len(spent) > 1 and cpu in ops:
        top = "、".join(f"{op}({ms:.1f} ms)" for op, ms in ops[cpu].most_common(4))
        lines.append(f"    落在 CPU 上最费时的算子：{top}")
        if not cpu_dominates:
            lines.append("    （CPU 这几个多半是算形状的小算子，动态尺寸模型里很常见，占比很小就不用管）")
    return lines


def _profile_providers(model_path: str, providers: list[str], feed: dict, device: int) -> None:
    """用 onnxruntime 的 profiling 统计每个算子实际落在哪个后端，按耗时排序。"""
    import collections
    import json
    import os
    import onnxruntime as ort
    from .operators import dl
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_profiling = True
    request = [(p, {"device_id": device}) if p in dl._GPU_PROVIDERS else p for p in providers]
    runs = 3
    prof = ""
    try:
        sess = ort.InferenceSession(model_path, so, providers=request)
        for _ in range(runs):
            sess.run(None, feed)
        prof = sess.end_profiling()
        with open(prof, encoding="utf-8") as f:
            events = json.load(f)
    except Exception as e:                                     # pragma: no cover - 取决于本机安装
        print(f"  （算子分布测不了：{e}）")
        return
    finally:
        try:
            os.remove(prof)
        except Exception:
            pass

    for line in format_profile(events, runs):
        print(line)


def cmd_cameras(args) -> int:
    """挨个报告每条取图路线认不认得本机的相机。

    现场最常见的问题是"相机插着但软件看不见"，而原因分散在好几层：驱动没装、
    被厂商软件独占、设备是 DirectShow 而不是 UVC、SDK 的 Python 封装没装……
    这里把每条路分别试一遍，省得逐个猜。
    """
    import platform
    print("== OpenCV（UVC / DirectShow / 视频流）==")
    from .camera.opencv_cam import probe_indices
    backends = ["auto", "dshow", "msmf"] if platform.system() == "Windows" else ["auto", "v4l2"]
    any_cv = False
    for b in backends:
        try:
            idx = probe_indices(b, args.probe)
        except Exception as e:                                   # pragma: no cover - 取决于本机
            print(f"  {b:8s} 探测失败：{e}")
            continue
        any_cv = any_cv or bool(idx)
        print(f"  {b:8s} {('设备号 ' + '、'.join(str(i) for i in idx)) if idx else '没有可打开的设备'}")
    if any_cv:
        print("  → 相机节点：取图方式 opencv，相机来源填上面的设备号，采集后端选对应的那个")

    print("\n== 迈德威视 MindVision ==")
    from .camera import mindvision_cam
    if mindvision_cam.sdk_available():
        devs = mindvision_cam.enumerate_mindvision()
        if devs:
            for i, d in enumerate(devs):
                print(f"  {i}  {d.user_name or d.model}　序列号 {d.serial}　{d.extra.get('port', '')}")
            print("  → 相机节点：取图方式 mindvision，相机来源填序列号或上面的序号")
        else:
            print("  SDK 能用，但没枚举到相机（驱动装了吗？是否被厂商软件占用？）")
    else:
        print("  没找到 Python 模块 mvsdk。它在厂商的 **Python Demo / SDK 开发包** 里，")
        print("  运行时目录（只有 MVCAMSDK.dll 那些）不带它。装好之后把 mvsdk.py 所在目录")
        print("  填进相机节点的「迈德威视 SDK 目录」，或设成环境变量 MVSDK_PATH")

    print("\n== 海康机器人 MVS ==")
    from .camera import hik_cam
    if hik_cam.sdk_available():
        try:
            devs = hik_cam.enumerate_hik()
            print(f"  SDK 可用，枚举到 {len(devs)} 台")
            for d in devs:
                print(f"    {d.display_name}　{d.ip or ''}　序列号 {d.serial}")
        except Exception as e:                                   # pragma: no cover
            print(f"  SDK 可用但枚举失败：{e}")
    else:
        print("  没找到 MVS SDK（没装海康相机的话正常）")

    if args.gige:
        print("\n== GigE Vision 广播发现 ==")
        from .camera.manager import enumerate_cameras
        devs = enumerate_cameras(timeout=args.timeout)
        for d in devs:
            print(f"  {d.display_name}　{d.ip}　{d.manufacturer} {d.model}")
        if not devs:
            print("  没发现网络相机")
    else:
        print("\n（加 --gige 还会广播搜索网络相机，需要几秒）")
    return 0


def cmd_validate(args) -> int:
    sol = _load(args.solution, args.plugins)
    problems = 0
    for g in sol.flows.values():
        for w in g.validate():
            print(f"[{g.name}] {w}")
            problems += 1
    print(f"{len(sol.flows)} 个流程，{sum(len(g.nodes) for g in sol.flows.values())} 个节点，{problems} 条警告")
    return 1 if problems else 0


def cmd_new(args) -> int:
    from .core import Graph, Solution, registry
    registry.load_builtins()
    sol = Solution(Path(args.solution).stem)
    sol.add_flow(Graph("main"))
    sol.save(args.solution)
    print("已创建", args.solution)
    return 0


def cmd_run(args) -> int:
    from .core import FlowRunner, Trigger, TriggerSource
    from .comm import CommManager, set_manager
    sol = _load(args.solution, args.plugins)
    flow = args.flow or next(iter(sol.flows))
    if flow not in sol.flows:
        print(f"没有名为 {flow!r} 的流程；可用流程：{list(sol.flows)}", file=sys.stderr)
        return 2
    mgr = CommManager(sol.bus, sol.variables)
    mgr.load_dict(sol.comm_config)
    set_manager(mgr)
    if args.connect:
        for e in mgr.connect_all():
            print("通信：", e, file=sys.stderr)
    runner = FlowRunner(sol.flows[flow], sol.variables, sol.bus)
    out = open(args.output, "a", encoding="utf-8") if args.output else None
    try:
        for i in range(args.count):
            r = runner.run_once(Trigger(TriggerSource.CLI))
            line = f"#{r.run_id:<5d} {r.status_text:5s} {r.duration_ms:7.1f} ms  {json.dumps(r.outputs, default=str, ensure_ascii=False)}"
            if r.error:
                line += f"  | {r.error}"
            print(line)
            if out:
                out.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            if args.interval:
                time.sleep(args.interval)
    finally:
        runner.engine.teardown_nodes()
        if out:
            out.close()
        mgr.shutdown()
    s = runner.stats
    print(f"\n共运行 {s.count} 次：OK {s.ok}，NG {s.ng}，错误 {s.error}，平均 {s.avg_ms:.1f} ms")
    return 0 if s.error == 0 else 1


def cmd_serve(args) -> int:
    from .core import FlowRunner
    from .comm import CommManager, set_manager
    sol = _load(args.solution, args.plugins)
    mgr = CommManager(sol.bus, sol.variables)
    for e in mgr.load_dict(sol.comm_config):
        print("通信：", e, file=sys.stderr)
    set_manager(mgr)
    runners = {name: FlowRunner(g, sol.variables, sol.bus) for name, g in sol.flows.items()}
    mgr.set_runners(runners)
    for e in mgr.connect_all():
        print("通信：", e, file=sys.stderr)
    for name, r in runners.items():
        for e in r.start():
            print(f"[{name}] {e}", file=sys.stderr)
        if args.continuous and name == (args.flow or next(iter(runners))):
            r.set_continuous(args.continuous)
    for d in mgr.devices.values():
        if hasattr(d, "bound_port"):      # 服务端类设备报告真实监听端口，配 0 时尤其有用
            print(f"设备 {d.name}（{d.kind}）监听 {d.config.get('host', '0.0.0.0')}:{d.bound_port}")
        else:
            print(f"设备 {d.name}（{d.kind}）")
    print(f"运行中：流程 {list(runners)}，按 Ctrl-C 停止")
    stop = False

    def _sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    last = time.time()
    while not stop:
        time.sleep(0.2)
        if time.time() - last >= args.stats_every:
            last = time.time()
            for name, r in runners.items():
                s = r.stats
                print(f"[{name}] 运行={s.count} OK={s.ok} NG={s.ng} 错误={s.error} 平均={s.avg_ms:.1f}ms "
                      f"排队={r.pending()} | " + " ".join(f"{d.name}:{'已连接' if d.connected else '未连接'}" for d in mgr.devices.values()))
    for r in runners.values():
        r.stop()
    mgr.shutdown()
    print("已停止")
    return 0


def cmd_shortcut(args) -> int:
    from .launcher import create_shortcut
    try:
        path = create_shortcut(args.solution, args.name)
    except Exception as e:
        print(f"创建快捷方式失败：{e}", file=sys.stderr)
        return 1
    print(f"已创建桌面快捷方式：{path}")
    return 0


def cmd_gui(args) -> int:
    from .ui import main as ui_main
    return ui_main([sys.argv[0]] + ([args.solution] if args.solution else []))


def force_utf8_output() -> None:
    """把标准输出/错误切成 UTF-8，让中文在任何 Windows 代码页下都不会崩。

    输出接到控制台时 Python 用 WriteConsoleW，中文没问题；但一旦被**重定向**
    （管道、`> log.txt`、CI 抓日志），用的就是本地编码——英文 Windows 是 cp1252，
    写中文直接抛 UnicodeEncodeError，整个进程挂掉。产线上 `cvflow serve > 日志.txt`
    就会踩到。errors="replace" 再兜一层：真遇到编不出来的字符也只是显示成 ?，不中断。
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        if enc.startswith("utf8") or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                          # pragma: no cover - 取决于流的类型
            pass


def main(argv=None) -> int:
    force_utf8_output()
    # 输出重定向到管道或文件时：编码不支持中文会直接崩溃，改用替代符；
    # 同时改成行缓冲，否则 serve 的日志会攒在缓冲区里，运维 tail 日志看不到东西。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="cvflow", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    ap.add_argument("-p", "--plugins", action="append", default=[], help="额外的插件目录（可重复）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gui", help="打开桌面程序"); g.add_argument("solution", nargs="?", help="方案文件"); g.set_defaults(fn=cmd_gui)
    r = sub.add_parser("run", help="在终端运行流程"); r.add_argument("solution", help="方案文件"); r.add_argument("-f", "--flow", help="流程名")
    r.add_argument("-n", "--count", type=int, default=1, help="运行次数"); r.add_argument("-i", "--interval", type=float, default=0.0, help="两次运行之间的间隔（秒）")
    r.add_argument("-o", "--output", help="把每次结果追加写入 JSON Lines 文件"); r.add_argument("--connect", action="store_true", help="连接通信设备")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("serve", help="无界面生产模式"); s.add_argument("solution", help="方案文件"); s.add_argument("-f", "--flow", help="定时触发的流程名")
    s.add_argument("-c", "--continuous", type=float, default=0.0, help="定时触发间隔（秒）")
    s.add_argument("--stats-every", type=float, default=10.0, help="统计信息打印间隔（秒）"); s.set_defaults(fn=cmd_serve)
    n = sub.add_parser("nodes", help="列出节点类型"); n.add_argument("--json", action="store_true", help="以 JSON 输出"); n.set_defaults(fn=cmd_nodes)
    v = sub.add_parser("validate", help="检查方案"); v.add_argument("solution", help="方案文件"); v.set_defaults(fn=cmd_validate)
    w = sub.add_parser("new", help="新建空方案"); w.add_argument("solution", help="方案文件"); w.set_defaults(fn=cmd_new)
    sc = sub.add_parser("shortcut", help="在桌面创建快捷方式"); sc.add_argument("solution", nargs="?", help="快捷方式打开的方案（可选）")
    sc.add_argument("--name", default="CVFlow", help="快捷方式名称"); sc.set_defaults(fn=cmd_shortcut)
    cp = sub.add_parser("cameras", help="看看各条取图路线分别认不认得本机的相机")
    cp.add_argument("--probe", type=int, default=4, metavar="个数", help="OpenCV 试到几号设备（默认 4）")
    cp.add_argument("--gige", action="store_true", help="同时广播搜索网络相机（要几秒）")
    cp.add_argument("--timeout", type=float, default=1.0, help="网络搜索超时秒数")
    cp.set_defaults(fn=cmd_cameras)
    gp = sub.add_parser("gpu", help="自检推理环境（后端、显卡、实测加载）")
    gp.add_argument("model", nargs="?", help="用来实测的 ONNX 模型文件（可选）")
    gp.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda", "tensorrt"], help="指定推理后端")
    gp.add_argument("--device", type=int, default=0, help="显卡编号")
    gp.add_argument("--require-gpu", action="store_true",
                    help="CUDA 没跑起来就以非零退出码结束，给安装脚本和 CI 用")
    gp.add_argument("--bench", nargs="?", type=int, const=20, default=0, metavar="次数",
                    help="实测推理耗时，并和纯 CPU 对比（默认 20 次）。"
                         "「日志说用了 CUDA 却和 CPU 一样快」就用它查")
    gp.add_argument("--size", type=int, default=640, metavar="边长",
                    help="动态尺寸模型做实测时用的高宽，默认 640")
    gp.set_defaults(fn=cmd_gpu)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

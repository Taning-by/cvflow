"""命令行入口。

  cvflow gui [方案.json]                 打开桌面程序
  cvflow run 方案.json [-f main] [-n 5]  在终端运行流程 N 次
  cvflow serve 方案.json                 无界面生产模式：连接设备、启动全部流程
  cvflow nodes [--json]                  列出已注册的节点类型
  cvflow validate 方案.json              检查方案是否有问题
  cvflow new 方案.json                   新建一个空方案
  cvflow shortcut [方案.json]            在桌面创建快捷方式（点击图标打开软件）
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


def _require_gpu_failed() -> int:
    """--require-gpu 下 CUDA 没跑起来：打一句结论并给非零退出码（安装脚本/CI 靠它判断）。"""
    print("\n✗ 要求用 GPU，但 CUDA 后端没能跑起来。按上面的提示修好再重试，"
          "完整步骤见 docs/install.md")
    return 2


def cmd_gpu(args) -> int:
    """自检推理环境：装了哪个 onnxruntime、后端能不能加载、模型实际跑在哪、占多少显存。"""
    import os
    print("== 推理运行时 ==")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  未安装 onnxruntime。CPU 装 pip install -e \".[cpu]\"，显卡装 pip install -e \".[gpu]\"（二选一）")
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
    try:
        from importlib.metadata import distributions
        names = {d.metadata["Name"].lower() for d in distributions() if d.metadata["Name"]}
        wheels = sorted(n for n in names if n in ("onnxruntime", "onnxruntime-gpu"))
        print(f"  已安装的包：{'、'.join(wheels) or '未知'}")
        if len(wheels) > 1:
            print("  ⚠ 同时装了 CPU 版和 GPU 版，它们会互相覆盖。"
                  "请 pip uninstall -y onnxruntime onnxruntime-gpu 之后只装需要的那一个")
    except Exception:                                            # pragma: no cover
        pass

    print("\n== 后端 ==")
    from .operators import dl
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
            print("     （缺 CUDA / cuDNN 运行库，pip install -e \".[gpu]\" 会把它们一起装上）")
    if "CUDAExecutionProvider" not in avail:
        both = len(wheels) > 1
        print("  → 当前这份 onnxruntime 里没有 GPU 后端，推理只能在 CPU 上跑。")
        if both:
            print("     原因是 CPU 版把 GPU 版覆盖了（两个包装的是同一个模块）。按下面三步修：")
        else:
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
    if not on_gpu:
        print("  → 跑在 CPU 上，所以显存不会变。按上面的提示修好后端即可")
    elif delta is not None and delta < 50:
        print("  ⚠ 跑在显卡上但显存几乎没涨，请确认看的是同一块卡（--device 可以指定）")
    node.unload()
    return 0 if (on_gpu or not args.require_gpu) else _require_gpu_failed()


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


def main(argv=None) -> int:
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
    gp = sub.add_parser("gpu", help="自检推理环境（后端、显卡、实测加载）")
    gp.add_argument("model", nargs="?", help="用来实测的 ONNX 模型文件（可选）")
    gp.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda", "tensorrt"], help="指定推理后端")
    gp.add_argument("--device", type=int, default=0, help="显卡编号")
    gp.add_argument("--require-gpu", action="store_true",
                    help="CUDA 没跑起来就以非零退出码结束，给安装脚本和 CI 用")
    gp.set_defaults(fn=cmd_gpu)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

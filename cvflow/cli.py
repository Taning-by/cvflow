"""命令行入口。

  cvflow gui [方案.json]                 打开桌面程序
  cvflow run 方案.json [-f main] [-n 5]  在终端运行流程 N 次
  cvflow serve 方案.json                 无界面生产模式：连接设备、启动全部流程
  cvflow nodes [--json]                  列出已注册的节点类型
  cvflow validate 方案.json              检查方案是否有问题
  cvflow new 方案.json                   新建一个空方案
  cvflow shortcut [方案.json]            在桌面创建快捷方式（点击图标打开软件）
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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

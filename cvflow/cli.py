"""Command line interface.

  cvflow gui [solution.json]                open the desktop application
  cvflow run solution.json [-f main] [-n 5] run a flow N times in the terminal
  cvflow serve solution.json                headless production mode: connect devices, start all flows
  cvflow nodes [--json]                     list registered node types
  cvflow validate solution.json             check a solution for problems
  cvflow new solution.json                  create an empty solution
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
    for cat, classes in registry.categories().items():
        print(f"{cat}")
        for c in classes:
            ins = ", ".join(f"{p.name}:{p.dtype.value}" for p in c.inputs)
            outs = ", ".join(f"{p.name}:{p.dtype.value}" for p in c.outputs)
            print(f"  {c.type_id:28s} {c.label:24s} in[{ins}] out[{outs}]")
    print(f"\n{len(registry.all())} node types")
    return 0


def cmd_validate(args) -> int:
    sol = _load(args.solution, args.plugins)
    problems = 0
    for g in sol.flows.values():
        for w in g.validate():
            print(f"[{g.name}] {w}")
            problems += 1
    print(f"{len(sol.flows)} flow(s), {sum(len(g.nodes) for g in sol.flows.values())} nodes, {problems} warning(s)")
    return 1 if problems else 0


def cmd_new(args) -> int:
    from .core import Graph, Solution, registry
    registry.load_builtins()
    sol = Solution(Path(args.solution).stem)
    sol.add_flow(Graph("main"))
    sol.save(args.solution)
    print("created", args.solution)
    return 0


def cmd_run(args) -> int:
    from .core import FlowRunner, Trigger, TriggerSource
    from .comm import CommManager, set_manager
    sol = _load(args.solution, args.plugins)
    flow = args.flow or next(iter(sol.flows))
    if flow not in sol.flows:
        print(f"no flow {flow!r}; available: {list(sol.flows)}", file=sys.stderr)
        return 2
    mgr = CommManager(sol.bus, sol.variables)
    mgr.load_dict(sol.comm_config)
    set_manager(mgr)
    if args.connect:
        for e in mgr.connect_all():
            print("comm:", e, file=sys.stderr)
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
    print(f"\n{s.count} runs: {s.ok} OK, {s.ng} NG, {s.error} ERROR, avg {s.avg_ms:.1f} ms")
    return 0 if s.error == 0 else 1


def cmd_serve(args) -> int:
    from .core import FlowRunner
    from .comm import CommManager, set_manager
    sol = _load(args.solution, args.plugins)
    mgr = CommManager(sol.bus, sol.variables)
    for e in mgr.load_dict(sol.comm_config):
        print("comm:", e, file=sys.stderr)
    set_manager(mgr)
    runners = {name: FlowRunner(g, sol.variables, sol.bus) for name, g in sol.flows.items()}
    mgr.set_runners(runners)
    for e in mgr.connect_all():
        print("comm:", e, file=sys.stderr)
    for name, r in runners.items():
        for e in r.start():
            print(f"[{name}] {e}", file=sys.stderr)
        if args.continuous and name == (args.flow or next(iter(runners))):
            r.set_continuous(args.continuous)
    print(f"serving {list(runners)} - devices: {[f'{d.name}({d.kind})' for d in mgr.devices.values()]} - Ctrl-C to stop")
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
                print(f"[{name}] runs={s.count} ok={s.ok} ng={s.ng} err={s.error} avg={s.avg_ms:.1f}ms "
                      f"pending={r.pending()} | " + " ".join(f"{d.name}:{'up' if d.connected else 'down'}" for d in mgr.devices.values()))
    for r in runners.values():
        r.stop()
    mgr.shutdown()
    print("stopped")
    return 0


def cmd_gui(args) -> int:
    from .ui import main as ui_main
    return ui_main([sys.argv[0]] + ([args.solution] if args.solution else []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cvflow", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-p", "--plugins", action="append", default=[], help="extra plugin directory (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gui"); g.add_argument("solution", nargs="?"); g.set_defaults(fn=cmd_gui)
    r = sub.add_parser("run"); r.add_argument("solution"); r.add_argument("-f", "--flow")
    r.add_argument("-n", "--count", type=int, default=1); r.add_argument("-i", "--interval", type=float, default=0.0)
    r.add_argument("-o", "--output", help="append results as JSON lines"); r.add_argument("--connect", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("serve"); s.add_argument("solution"); s.add_argument("-f", "--flow")
    s.add_argument("-c", "--continuous", type=float, default=0.0, help="timer trigger interval in seconds")
    s.add_argument("--stats-every", type=float, default=10.0); s.set_defaults(fn=cmd_serve)
    n = sub.add_parser("nodes"); n.add_argument("--json", action="store_true"); n.set_defaults(fn=cmd_nodes)
    v = sub.add_parser("validate"); v.add_argument("solution"); v.set_defaults(fn=cmd_validate)
    w = sub.add_parser("new"); w.add_argument("solution"); w.set_defaults(fn=cmd_new)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

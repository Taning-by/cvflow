"""测量「多个深度学习节点怎么用同一块显卡」的四种做法各有多快。

要回答的问题是：几个节点共用一份权重（于是推理要排队）比各自一份权重慢多少？
值不值得为了省显存去共用？

四种做法（对应深度学习节点的参数组合）：

  one-node       一个节点一次收齐 N 张图，整批推理         ——「同一节点内合批」的上限
  per-node       N 个节点各自一份权重、各自一个队列        ——默认（GPU 上 auto 就是这个）
  shared-session N 个节点共用一份权重、但各自一个队列      —— session_scope=shared
  group          N 个节点共用一份权重、且共用一个队列      —— 填了同名的分组

每个节点在自己的线程里跑，模拟「每路相机一个流程、各自被 PLC 触发」。

    python tools/bench_sharing.py 模型.onnx --nodes 4 --rounds 20 --size 512
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvflow.core import Image, registry                                    # noqa: E402
from cvflow.operators import dl                                            # noqa: E402


def _vram_mib(device: int) -> int | None:
    from cvflow.cli import _device_vram_mib
    return _device_vram_mib(device)


def make_nodes(count: int, model: str, size: int, mode: str, args) -> list:
    """按某种做法建好 count 个深度学习节点。"""
    nodes = []
    for i in range(count):
        values = {
            "model_path": model, "width": size, "height": size, "color": "rgb",
            "provider": args.provider, "device_id": args.device,
            "max_batch": args.nodes if mode in ("one-node", "group") else 1,
            "input_count": args.nodes if mode == "one-node" else 1,
            "wait_ms": args.wait_ms if mode == "group" else 0.0,
            "session_scope": {"shared-session": "shared", "group": "auto",
                              "per-node": "exclusive", "one-node": "auto"}[mode],
            "batch_group": "bench" if mode == "group" else "",
        }
        nodes.append(registry.create("dl.onnx", values=values))
    return nodes


def run_mode(mode: str, model: str, args) -> dict:
    """跑一种做法，返回耗时统计。"""
    size = args.size
    count = args.nodes
    nodes = make_nodes(1 if mode == "one-node" else count, model, size, mode, args)
    img = Image(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))

    before = _vram_mib(args.device)
    for n in nodes:
        n.preload()
    sessions = dl.active_sessions()
    after_load = _vram_mib(args.device)

    # 预热：第一次推理含 cuDNN 挑算法
    if mode == "one-node":
        nodes[0]._infer_batch({i: img for i in range(count)})
    else:
        for n in nodes:
            n._infer_batch({0: img})

    rounds = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        if mode == "one-node":
            nodes[0]._infer_batch({i: img for i in range(count)})          # N 张一次进去
        else:
            errs: list[BaseException] = []

            def one(node):
                try:
                    node._infer_batch({0: img})
                except BaseException as e:                                 # noqa: BLE001
                    errs.append(e)

            ts = [threading.Thread(target=one, args=(n,)) for n in nodes]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            if errs:
                raise errs[0]
        rounds.append((time.perf_counter() - t0) * 1000)

    for n in nodes:
        n.unload()
        n.teardown()
    rounds.sort()
    return {"mode": mode, "sessions": sessions, "median": rounds[len(rounds) // 2],
            "p10": rounds[max(0, int(len(rounds) * 0.1))], "p90": rounds[min(len(rounds) - 1, int(len(rounds) * 0.9))],
            "vram": (after_load - before) if (before is not None and after_load is not None) else None,
            "imgs": count}


def _run_child(mode: str, model: str, args) -> dict:
    """在子进程里跑一种做法并把结果打成 JSON。

    每种做法必须独立进程：CUDA 上下文（几百 MiB）只有第一次建会话时付钱，
    同进程里跑第二种做法时显存增量就失真了；上一种做法的会话也可能还没放掉。
    """
    import json
    import subprocess
    cmd = [sys.executable, str(Path(__file__).resolve()), model, "--one-mode", mode,
           "--nodes", str(args.nodes), "--rounds", str(args.rounds), "--size", str(args.size),
           "--provider", args.provider, "--device", str(args.device), "--wait-ms", str(args.wait_ms)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        raise RuntimeError(f"子进程没给出结果：\n{out.stdout[-500:]}\n{out.stderr[-800:]}")
    return json.loads(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="ONNX 模型文件")
    ap.add_argument("--nodes", type=int, default=4, help="几个节点 / 一批几张图（默认 4）")
    ap.add_argument("--rounds", type=int, default=20, help="每种做法跑几轮（默认 20）")
    ap.add_argument("--size", type=int, default=512, help="输入边长（默认 512）")
    ap.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda", "tensorrt"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--wait-ms", type=float, default=50.0, help="group 做法的等待窗口（默认 50 ms）")
    ap.add_argument("--modes", default="one-node,per-node,shared-session,group")
    ap.add_argument("--one-mode", default="", help="内部用：只跑这一种做法并输出 JSON")
    args = ap.parse_args(argv)

    registry.load_builtins()
    from cvflow.core import paths
    paths.set_base_dir(None)
    model = str(Path(args.model).resolve())

    if args.one_mode:                                  # 子进程：跑一种，打 JSON
        import json
        print(json.dumps(run_mode(args.one_mode, model, args)))
        return 0

    print(f"模型 {Path(model).name}　{args.nodes} 路　{args.size}×{args.size}　"
          f"每种做法 {args.rounds} 轮（每种一个独立进程）\n")
    print(f"{'做法':16s} {'权重份数':>8s} {'一轮耗时(中位)':>14s} {'p10~p90':>16s} {'单张均摊':>10s} {'显存增量':>10s}")
    base = None
    for mode in args.modes.split(","):
        r = _run_child(mode.strip(), model, args)
        per = r["median"] / r["imgs"]
        if base is None:
            base = per
        vram = f"{r['vram']} MiB" if r["vram"] is not None else "测不到"
        print(f"{r['mode']:16s} {r['sessions']:>8d} {r['median']:>11.1f} ms "
              f"{r['p10']:>7.1f}~{r['p90']:<7.1f} {per:>7.1f} ms {vram:>12s}"
              + (f"   ×{per / base:.2f}" if base else ""))
    print("\n「一轮耗时」= 这一轮里全部 N 张图都出结果所用的墙上时间；「单张均摊」= 除以 N，可直接横向比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

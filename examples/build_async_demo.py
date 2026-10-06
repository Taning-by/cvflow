"""生成事件驱动的示例方案（examples/solutions/demo_async.json）和它要用的小模型。

形态就是产线上的样子：两路图像源各自被触发，在**一个**深度学习节点上汇合，凑够批次或
等待窗口到期就一次推理，由先到的那一路带着整批结果继续往下游跑。

    python examples/build_async_demo.py
    cvflow gui examples/solutions/demo_async.json

打开后选中任一个「相机」节点，参数面板最上面有「触发一次」按钮：
点一下 → 这一路取一帧图、跑到深度学习节点等着；在 400 ms 内点另一路 → 两路合成一批推理。
只点一路不管它，窗口到期也会照常执行（batch_size 显示 1）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                       # noqa: E402

from cvflow.core import Graph, paths, registry                           # noqa: E402
from cvflow.core.runtime import Solution                                 # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "channel_mean.onnx"


def make_model(path: Path) -> None:
    """造一个极小的分类模型：按三个通道的均值分类，批次维是动态的（才能合批）。

    用它是为了示例能开箱就跑——换成你自己的 YOLO 只要改模型路径和节点类型。
    """
    import onnx
    from onnx import TensorProto, helper
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = [helper.make_node("ReduceMean", ["images"], ["pooled"], axes=[2, 3], keepdims=0)]
    graph = helper.make_graph(
        nodes, "channel_mean",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 8, 8])],
        [helper.make_tensor_value_info("pooled", TensorProto.FLOAT, ["N", 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 9
    onnx.save(model, path)
    print(f"模型已生成 {path.relative_to(ROOT)}")


def build() -> Solution:
    registry.load_builtins()
    # 方案里的路径都相对方案文件所在目录解析，所以先把基准目录设成它，整个 examples 目录
    # 可以随意搬动或从 git 克隆
    paths.set_base_dir(HERE / "solutions")
    g = Graph("main")
    model = "../models/" + MODEL.name

    dl = registry.create("dl.onnx_classifier", name="两路检测", values={
        "model_path": model, "input_count": 2, "width": 8, "height": 8, "color": "rgb",
        "arrival": "async",          # ← 事件驱动：两路各自到达，在这里汇合
        "wait_ms": 400.0,            # ← 先到的等后到的最多 400 ms
        "max_batch": 4,
        "weight_group": "线A",        # ← 和别的深度学习节点共用同一份权重（这里只有一个节点，摆个样子）
    })
    dl.position = [520, 160]
    g.add_node(dl)

    for k, port in enumerate(("image", "image2")):
        cam = registry.create("source.camera", name=f"相机{k + 1}", values={
            "camera": f"cam{k}", "kind": "folder",
            "source": "../images",                         # 用示例图像文件夹模拟相机
            "trigger_mode": "off",
        })
        cam.position = [60, 60 + k * 200]
        g.add_node(cam)
        pre = registry.create("preprocess.resize", name=f"缩放{k + 1}", values={"width": 8, "height": 8})
        pre.position = [290, 60 + k * 200]
        g.add_node(pre)
        g.add_link(cam.id, "image", pre.id, "image")
        g.add_link(pre.id, "image", dl.id, port)

    judge = registry.create("logic.judge", name="判定", values={})
    judge.position = [800, 160]
    g.add_node(judge)
    g.add_link(dl.id, "class_id", judge.id, "value")

    sol = Solution()
    sol.flows[g.name] = g
    return sol


def main() -> int:
    make_model(MODEL)
    out = HERE / "solutions" / "demo_async.json"
    build().save(out)
    print(f"方案已生成 {out.relative_to(ROOT)}")
    print("\n打开它：cvflow gui examples/solutions/demo_async.json")
    print("选中「相机1」→ 点参数面板上的「触发一次」；400 ms 内再点「相机2」的，两路就会合成一批。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# CVFlow — open, flow-based industrial machine vision

[中文说明 / Chinese README](README.zh-CN.md)

CVFlow is a Python industrial vision platform in the spirit of VisionMaster / VisionPro:
you build an inspection as a **flow of nodes**, tune parameters live on the image, and
connect the result to a **PLC or robot over TCP / UDP / serial / Modbus**. The difference
is that every algorithm node is an ordinary Python class, so your own OpenCV code, an
ONNX / PyTorch model or a reinforcement-learning policy plugs into the same flow editor
without touching the platform.

```
┌───────────────────────────── cvflow.ui (PySide6) ─────────────────────────────┐
│  node editor   image view (overlays, ROI drawing)   parameters   results      │
│  communication   variables   log                                              │
└───────────────────────────────────────────────────────────────────────────────┘
┌──────────────── cvflow.core ────────────────┐  ┌──────── cvflow.comm ────────┐
│ types  node  registry  graph  engine        │  │ TCP client/server  UDP      │
│ runtime (FlowRunner, Solution)  events      │  │ serial  Modbus TCP (both)   │
│ variables  draw  paths                      │  │ receive rules → triggers    │
└─────────────────────────────────────────────┘  │ send rules    → results     │
┌────────────── cvflow.operators ─────────────┐  └─────────────────────────────┘
│ Source Preprocess Analysis Deep Learning    │  ┌──────── cvflow.camera ──────┐
│ Script Logic Output + your plugins          │  │ folder (simulator) OpenCV   │
└─────────────────────────────────────────────┘  │ GenICam (harvesters, opt.)  │
                                                 └─────────────────────────────┘
```

The core (`cvflow.core`) has no Qt dependency and runs headless on a production PC.

## Install

Requires Python 3.10+ on Linux, Windows or macOS.

```bash
git clone https://github.com/Taning-by/cvflow.git
cd cvflow
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + GUI + tests
```

Optional extras: `.[gpu]` (onnxruntime-gpu), `.[genicam]` (harvesters for GigE / USB3 Vision
cameras), `.[dl]` (PyTorch for the plugin template).

## Run

```bash
cvflow nodes                                          # list the 40 built-in node types
cvflow run examples/solutions/demo_holes.json -n 10   # run the demo flow in the terminal
cvflow gui examples/solutions/demo_holes.json         # open the desktop editor
cvflow serve examples/solutions/demo_holes.json       # headless production mode
pytest -q                                             # 35 tests, GUI tests run offscreen
```

`python -m cvflow …` works too, without installing.

The demo solution `examples/solutions/demo_holes.json`:

* Flow **main** — folder source → gray → threshold → open → blob count (3 holes) → judge,
  plus an edge caliper measuring the part width → judge. Results are published, overlays
  rendered and NG images saved to `captures/NG`.
* Flow **learning** — the ε-greedy bandit node from `examples/plugins/bandit_threshold.py`
  tunes the threshold from the previous run's reward: a stateful, RL-style node inside a flow.
* Communication — a TCP server on port 6000; a frame starting with `TRIG` triggers `main`,
  the result goes back as `{status},{out.holes},{out.width:.1f}\n`.

Simulate the PLC with netcat: run `cvflow serve examples/solutions/demo_holes.json`, then
`printf 'TRIG\n' | nc -q1 127.0.0.1 6000`.

Paths inside a solution (image folders, models, templates, capture directory, plugin
directories) are relative to the solution file, so a solution folder can be moved or cloned.

## Using the editor

* Drag nodes from the palette (or double-click), drag from an output port to an input port
  to link, `Delete` removes, `F` fits the view, middle mouse / Alt+drag pans.
* Select a node to edit its parameters; for ROI parameters press **Draw** and drag a
  rectangle on the image.
* **Auto-run on change** re-runs the flow on every edit; **Run once** (F5); **Continuous**
  loops at the chosen interval.
* **Start run mode** (F9) connects all communication devices and starts a worker thread per
  flow. The flow is locked; triggers come from the PLC, timer or the Run button.
* The *Communication* tab manages devices, receive rules (what triggers a flow or sets a
  variable), send rules (what is sent when a flow finishes), a monitor and manual sending.

## Writing your own node

Put a `.py` file in a plugin directory (the solution's `plugin_dirs`, or *Plugins → Load
plugin folder*). Every `Node` subclass in it is registered automatically:

```python
from cvflow.core import Node, Port, Param, DataType, Overlay, Rect

class MyDefectNet(Node):
    type_id = "mycompany.defect_net"      # globally unique
    category = "Deep Learning"
    label = "My Defect Net"
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("score", DataType.FLOAT), Port("boxes", DataType.LIST)]
    params = [Param("weights", "", "file"), Param("threshold", 0.5, "float", min=0, max=1)]

    def setup(self, ctx=None):            # once, when the flow starts: load the model
        import torch
        self.model = torch.load(self.get("weights"))

    def process(self, ctx, inputs):       # per image
        img = inputs["image"]             # cvflow.core.Image; .data is a numpy BGR/gray array
        score, boxes = self.infer(img.data)
        for x, y, w, h in boxes:
            ctx.add_overlay(Overlay.rect(Rect(x, y, w, h), "#ff0000"))
        ctx.judge(score < self.get("threshold"), "defect")   # contributes to the run's OK/NG
        return {"score": score, "boxes": boxes}
```

* `self.state` is persisted with the solution — use it for learning statistics or counters
  (see the bandit plugin).
* `ctx.publish(name, value)` exposes a value to communication templates as `{out.name}`.
* For quick experiments use the built-in **Python Script** node and write
  `process(ctx, inputs, params, state)` directly in the parameter panel.
* The built-in **ONNX Inference / Classifier / Detector (YOLO)** nodes run exported models
  with onnxruntime (CPU, CUDA or TensorRT providers).

## Communication model

| Concept | Details |
|---|---|
| Devices | `tcp_client` `tcp_server` `udp` `serial` `modbus_tcp_client` (we are the master) `modbus_tcp_server` (we are the slave) |
| Receive rules | Text devices: `startswith / equals / contains / regex / any`. Modbus: `register_rising` (0→non-zero), `register_change`, `register_equals`. Action: trigger a flow, or store the frame in a global variable |
| Send rules | Fire when a flow finishes (`always / ok / ng / error`). Text template: `{status} {ok} {ng} {run_id} {duration_ms:.1f} {out.name} {var.name} {node[Node Name].port}`. Modbus: a list of register writes `[{"address": 10, "expr": "{out.count}", "kind": "int16"}]` with kinds int16 / uint16 / int32 / uint32 / float32 / bool |
| In-flow sending | The **Send Message** and **Modbus Write** nodes send from inside a flow |

The Modbus TCP slave is implemented natively (function codes 1/2/3/4/5/6/15/16) so every
PLC write is observed immediately. The master uses pymodbus and polls a watched register
block every `poll_ms`.

## Layout

```
cvflow/core        types, node model, registry/plugins, graph, engine, runtime, events, variables, overlays, paths
cvflow/operators   built-in nodes (source / preprocess / analysis / dl / logic / output)
cvflow/camera      camera abstraction: folder simulator, OpenCV, GenICam (harvesters)
cvflow/comm        communication devices and rule manager
cvflow/ui          PySide6 application
cvflow/cli.py      gui / run / serve / nodes / validate / new
examples/          demo image generator, demo solution, example plugins
tests/             pytest (core, operators, communication, offscreen UI)
```

## Current limits

* Classical operators cover what OpenCV offers directly: blobs, contours, NCC template
  matching (translation only), sub-pixel edge caliper, circle finding, intensity statistics,
  QR codes. No rotation/scale-invariant shape matching, OCR or camera calibration yet.
* A flow executes as a single-threaded DAG (flows run in parallel, nodes within a flow do
  not). Output-category nodes are scheduled after other ready nodes so saving/rendering sees
  the final judgement. Sub-20 ms cycle times need heavy nodes in a subprocess or C++ extension.
* GenICam cameras need `harvesters` plus a vendor GenTL producer and have not been tested on
  real hardware; no direct Hikrobot MVS SDK wrapper yet.
* Vendor PLC protocols (Siemens S7, Mitsubishi MC, Omron FINS) are not built in; implement a
  `CommDevice` subclass (python-snap7 / omniplc) to add them.
* No user management, result database, solution versioning or installer yet.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

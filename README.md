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
pip install -e ".[dev,cpu]"        # core + GUI + tests + CPU inference
```

**Pick exactly one of `cpu` and `gpu`.** `onnxruntime` and `onnxruntime-gpu` ship the same Python
module and overwrite each other when both are installed — the usual symptom is "installed the GPU
build but only the CPU backend is there". For a GPU box use instead:

```bash
pip install -e ".[dev,gpu]"        # onnxruntime-gpu + CUDA 12 / cuDNN 9 from pip (~2 GB)
```

The GPU extra pulls the CUDA 12 and cuDNN 9 pip packages with it, so **no system CUDA install and
no `LD_LIBRARY_PATH` / `PATH` fiddling are needed**: those libraries are not on the loader path, and
CVFlow preloads them before creating an inference session.

If both ended up installed, clean out first: `pip uninstall -y onnxruntime onnxruntime-gpu`, then
install the one you need.

Other extras: `.[genicam]` (harvesters for GigE / USB3 Vision cameras), `.[plc]` (python-snap7 for
Siemens S7), `.[dl]` (PyTorch for the plugin template).

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

## Launching from an icon

Installing creates a console-free entry point `cvflow-gui` (`.venv\Scripts\cvflow-gui.exe` on
Windows) that can be double-clicked. To get a desktop shortcut with the application icon:

```bash
cvflow shortcut                                     # desktop shortcut opening the app
cvflow shortcut examples/solutions/demo_holes.json  # shortcut that opens a given solution
```

Windows gets `CVFlow.lnk`, Linux `CVFlow.desktop` (also added to the application menu), macOS
`CVFlow.command`. *File → Create desktop shortcut* does the same from the GUI. Started without a
solution, the app reopens the last one; *File → Recent* lists the last eight.

## Using the editor

The workspace has five zones: a top bar grouping project actions (new / open / save, flow picker)
and execution actions (run once, continuous, run mode); a searchable node palette on the left;
**image** and **flow** side by side in the centre (drag the splitter to rebalance, maximise either
one from its header, switch between side-by-side and stacked from the *View* menu — the arrangement
is remembered); the selected node's parameters and last outputs on the right; results /
communication / variables / log at the bottom, with frame, run statistics and mode in the status bar.

* Drag nodes from the palette (or double-click), drag from an output port to an input port
  to link, `Delete` removes, `F` fits the view (while the flow area has focus), right-button
  (or middle / Alt+left) drag pans, wheel zooms.
* Select a node to edit its parameters; for ROI parameters press **Draw** and drag a
  rectangle on the image.
* **Auto-run on change** re-runs the flow on every edit; **Run once** (F5); **Continuous**
  loops at the chosen interval.
* **Start run mode** (F9) connects all communication devices and starts a worker thread per
  flow. The flow is locked; triggers come from the PLC, timer or the Run button.
* The *Communication* tab manages devices, receive rules (what triggers a flow or sets a
  variable), send rules (what is sent when a flow finishes), a monitor and manual sending.

## Camera acquisition

**Camera → Camera manager** works like VisionMaster's camera management:

* **Search** broadcasts GigE Vision discovery on every local interface and lists IP, MAC,
  vendor, model, serial and user name of each camera, flagging cameras on a foreign subnet.
  Discovery needs no SDK; with the Hikrobot MVS SDK installed the SDK enumeration (incl. USB3)
  is merged in, and with a GenTL `.cti` set, GenICam enumeration too.
* **Add by IP** sends a unicast discovery; a silent camera can still be added manually.
* **Connection test** reads the camera's GigE Vision version register and reports latency;
  Hikrobot cameras are additionally opened through the SDK.
* **Force IP** assigns a temporary IP by MAC when the camera is on the wrong subnet.
* **Add to flow** creates a Camera node using `hik` (MVS SDK) or `genicam` acquisition.

Camera node parameters: trigger mode (keep / off / software / hardware), exposure (µs), gain,
timeout and a JSON of arbitrary GenICam features. Point `MVCAM_SDK_PATH` at the MVS
`Samples/Python/MvImport` directory for the Hikrobot SDK; GenICam needs `pip install
harvesters` and a vendor `.cti`.

## PLC interoperation

Every communication device has a **connection test** (TCP connect latency, Modbus/MC/S7 read a
register and report value and latency, serial open). Devices:

| Kind | Notes |
|---|---|
| TCP client / server, UDP, serial | Text frames; optional heartbeat frame and interval |
| Modbus TCP master / slave | Master polls PLC registers; the slave is a native implementation the PLC reads/writes |
| Mitsubishi MC (3E binary) | Q/L/iQ-R/FX5 and compatibles (Inovance, Keyence); word devices D/W/R and bit devices M/X/Y, addresses like `D100`, `M20`; ships `McSimulatorServer` |
| Siemens S7 | python-snap7, a DB block as exchange area, byte-offset addresses; S7-1200/1500 need PUT/GET enabled and non-optimised DB access |

Register devices share one handshake:

* **Trigger** — a `register_rising` receive rule watches the trigger word for 0→non-zero.
* **Busy flag** `busy_address` — written 1 on trigger, 0 after the results are written.
* **Trigger reset** `trigger_reset` — the vision side clears the trigger word so the next PLC
  write produces a clean edge.
* **Heartbeat** `heartbeat_address` + `heartbeat_s` — incremented periodically so the PLC can
  tell the vision PC is alive; text devices send a heartbeat frame instead.
* **Results** — register writes in a send rule, kinds int16 / uint16 / int32 / uint32 / float32 / bool.

* Text in the results panel can be selected and copied: Ctrl+C copies the selected rows, the
  context menu copies a single cell, a row, or everything. Long content is shortened on screen
  but the tooltip and the clipboard carry the full text. Node errors appear in the Value column.

## Testing without a camera or a PLC

Two simulators ship with the repository; every path is "open solution → start run mode → run the simulator":

| What to test | Step 1 | Step 2 |
|---|---|---|
| Camera search / connection test / force IP | `python examples/sim_camera.py` | Camera → Camera manager → Search, or *Add by IP* with 127.0.0.1 |
| Camera grabbing | set the camera node kind to `folder` with source `examples/images` | Run once |
| TCP trigger and reply | `cvflow gui examples/solutions/demo_holes.json`, start run mode | `python examples/sim_plc.py tcp` |
| Modbus TCP (PLC as master) | `cvflow gui examples/solutions/demo_modbus.json`, start run mode | `python examples/sim_plc.py modbus` |
| Mitsubishi MC | `python examples/sim_plc.py mc` | `cvflow gui examples/solutions/demo_mc.json`, start run mode |
| Siemens S7 | `python examples/sim_plc.py s7` | `cvflow gui examples/solutions/demo_s7.json`, start run mode |

The PLC simulators trigger an inspection periodically and print whether the trigger word was
reset, whether the busy flag appeared, the result registers and the heartbeat counter. The
simulated camera implements discovery only, so use the folder camera for grabbing.

## Batched multi-input deep learning nodes

All three ONNX nodes accept several image inputs. Set the input count to N and the node grows
`image2` to `imageN` input ports plus a set of outputs per input, such as `count`, `count2`,
`count3`. In one run the images on all connected inputs are stacked into a single batch, inferred
once, and the outputs are split back to the port belonging to each input.

| Parameter | Meaning | Default |
|---|---|---|
| input_count | Number of image input ports, up to 8 | 1 |
| max_batch | Largest batch handed to one inference | 8 |
| wait_ms | How long to wait for same-group nodes before running; **only applied when a batch group is set** | 0 |
| batch_group | Empty keeps the node independent and un-queued; a shared name lets nodes batch together but serialises them | empty |
| provider | `auto` picks the first usable of TensorRT → CUDA → CPU, or force one | auto |
| device_id | Which GPU to use on a multi-GPU box; ignored by the CPU backend | 0 |
| session_scope (advanced) | How many copies of the weights: `shared` one per process, `exclusive` one per group/node, `auto` shares on CPU and splits on GPU | auto |
| overlay_input | Which input's results are drawn on the image view | 1 |

Selecting a node with several image inputs puts a picker above the image view: it switches
between each input's image and the node's own output image. Overlays are tagged with the input
they came from, so only the selected input's boxes and labels are drawn.

### Running on the GPU

Install with `.[dev,gpu]` (see [Install](#install) — `.[gpu]` and `.[cpu]` are mutually exclusive)
and the default `auto` already prefers the GPU. Check what the wheel actually carries:

```bash
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

No `CUDAExecutionProvider` in the list means a CPU wheel is installed (or it overwrote the GPU
one): `pip uninstall -y onnxruntime onnxruntime-gpu`, then reinstall `.[gpu]`. If it is listed but
inference still falls back to the CPU, a runtime library failed to load and the log names it, e.g.
`libcublasLt.so.12: cannot open shared object file`.
`TensorrtExecutionProvider` additionally needs TensorRT itself; `cuda` or `auto` is usually what
you want. **To see where it actually
runs, read this log line**:

```
ONNX Detector (YOLO): model best.onnx using backend CUDAExecutionProvider (device 0)
```

`CPUExecutionProvider` there means the GPU backend failed to load (usually missing CUDA runtime
libraries); a warning line explains why. On a multi-GPU box, `device_id` spreads nodes across
cards — the one form of GPU parallelism that reliably pays off.

### Weight sharing and batch sharing are separate

| | Keyed by | Sharing means |
|---|---|---|
| **Inference session** (weights) | model file + mtime + provider + device_id + session owner | one copy of the weights in RAM / VRAM |
| **Batch executor** (queue) | all of the above + preprocessing + max_batch + wait_ms + batch owner | these nodes batch together and queue behind each other |

The session owner comes from `session_scope`, `auto` by default:

* **Shared on CPU.** A shared session does not block concurrent inference (onnxruntime's `Run` is
  thread safe — measured 2.01× for two threads on one session), parallelism is limited by cores,
  so a second copy of the weights would only waste memory.
* **One copy per batch group / node on GPU.** One session means one CUDA stream, so sharing means
  queuing; separate copies can at least use separate streams — at the cost of **multiplied VRAM**.
  If an exclusive copy cannot be allocated, the node falls back to the shared session and says so
  in the log instead of failing the flow.
* Set `shared` to save VRAM, or `exclusive` to split on CPU as well.

Note that separate GPU copies only buy throughput when a single inference does not already
saturate the GPU. Watch `nvidia-smi dmon -s u` during a single-stream run: above ~95% SM
utilisation, batch instead.

**The wait window only matters across flows.** A node's inputs are submitted in one call and the
nodes of one flow run in sequence, so without a batch group no second submitter can ever reach
the queue — the wait is therefore treated as 0 rather than padding the cycle. Check `batch_size`
to confirm batching, and change `max_batch` rather than `wait_ms` to compare batched against
one-by-one.

### When the images do not arrive together

Typical case: several cameras, each triggered by the PLC at its own moment. Use **one flow per
camera** (receive rule → trigger flow), give the deep-learning node in each flow the **same batch
group**, and set **wait_ms** to the slack your cycle time allows:

* the first image to arrive becomes the leader and waits inside the window for the others;
* reaching `max_batch` runs immediately, without waiting out the window;
* when the window expires the batch runs with whatever arrived — stragglers go into the next
  batch and nothing is ever stuck;
* each flow gets its own results back, and `batch_size` reports how many images that batch held.

Note also that batching speeds up inference only. Measured on 8 inputs of a 640×640 detector,
inference drops from 17.8 ms to 7.1 ms while end to end goes from 55.8 ms to 41.3 ms, because
preprocessing and postprocessing do not batch. Compare `infer_ms` with the node's total time to
see the split.

The inputs of one node arrive together, so they batch immediately and the wait window never
delays them. The window only matters across threads: when several flows infer at once, the first
one waits up to `wait_ms` for the others to join, and runs with whatever has arrived when the
window expires. A single flow should leave it at 0 and pays no extra latency.

Real batching needs a model exported with a dynamic batch dimension. A model with a fixed batch
of 1 falls back to one image at a time, and the `batch_size` output reports what actually ran.

**Input geometry adapts to the model.** When a model fixes its input shape, say it only accepts
512×512, the node's width, height, tensor layout and colour are corrected to match and the change
is logged. A fixed shape leaves no room for another value, so the correction is unambiguous; a
dynamic shape is left entirely to the node parameters. When nothing can be corrected
automatically, the error reports both the expected and the actual shape and names the parameter
to change.

Nodes with identical configuration share one inference session, so the weights are loaded once.
Note that sharing saves the weights; activation memory still grows with the number of images
inferred at the same time.

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
  with onnxruntime (CPU, CUDA or TensorRT providers). Installing the GPU build does not by
  itself mean the GPU is used: without matching CUDA and cuDNN runtime libraries the provider
  fails to load and falls back to CPU, so the log states which provider is actually in use.

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

## Node black-box test report

`docs/node-test-report.md` is a per-node black-box test report: contract tests cover every
registered node automatically (metadata, declared outputs present with the declared types, bad
input isolated), functional tests verify each mode and parameter against synthetic images with
analytically known ground truth. Regenerate with:

```bash
python tools/node_test_report.py
```

`docs/ui-test-report.md` is the GUI black-box report: the interface is driven only through
user-visible actions (menus, toolbar, mouse drags, keyboard, dialogs, table edits) and checked
through visible results, covering startup layout, file and flow management, palette, node
editor, parameter panel, run controls, image view, results/variables/log/communication panels,
camera manager, plugins and help. Regenerate with:

```bash
python tools/ui_test_report.py
```

`docs/comm-test-report.md` is the communication black-box report: the peer side is played by
real sockets, a Linux pseudo-terminal serial port, third-party pymodbus and snap7, and
hand-assembled protocol frames checked byte by byte; it covers TCP/UDP/serial, Modbus master and
slave, Mitsubishi MC, Siemens S7, receive/send rules, templates, handshake, heartbeat, events,
persistence, connection tests and the headless serve mode. Regenerate with:

```bash
python tools/comm_test_report.py
```

## Current limits

* Classical operators cover what OpenCV offers directly: blobs, contours, NCC template
  matching (translation only), sub-pixel edge caliper, circle finding, intensity statistics,
  QR codes. No rotation/scale-invariant shape matching, OCR or camera calibration yet.
* A flow executes as a single-threaded DAG (flows run in parallel, nodes within a flow do
  not). Output-category nodes are scheduled after other ready nodes so saving/rendering sees
  the final judgement. Sub-20 ms cycle times need heavy nodes in a subprocess or C++ extension.
* The Hikrobot MVS wrapper and GenICam grabbing are not yet verified on real cameras (none
  on the development machine); GigE discovery, force-IP and the connection test are covered by
  automated tests against a simulated camera.
* Mitsubishi MC and Siemens S7 are verified only against the simulator / snap7 server; Omron
  FINS and Rockwell EtherNet/IP are not implemented.
* No user management, result database, solution versioning or installer yet.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

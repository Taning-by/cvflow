"""Template plugin wrapping a PyTorch model. Copy, rename, fill in `build_model`.

torch is imported lazily so the plugin loads (and the rest of the platform works)
on machines without PyTorch installed.
"""
from __future__ import annotations

import numpy as np

from cvflow.core import DataType, Node, NodeError, Overlay, Param, Port


class TorchModelNode(Node):
    type_id = "learning.torch_template"
    category = "Learning"
    label = "PyTorch Model (template)"
    description = "Runs a PyTorch model on the input image. Edit this plugin to load your own network."
    color = "#c62828"
    inputs = [Port("image", DataType.IMAGE)]
    outputs = [Port("output", DataType.TENSOR), Port("score", DataType.FLOAT)]
    params = [Param("weights", "", "file", filter="PyTorch weights (*.pt *.pth)"),
              Param("device", "auto", "enum", choices=["auto", "cpu", "cuda"]),
              Param("size", 224, "int", min=16, max=4096)]

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._model = None
        self._torch = None

    def build_model(self, torch):
        """Return an nn.Module. Replace with your architecture."""
        import torch.nn as nn
        return nn.Sequential(nn.Conv2d(3, 8, 3, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 1))

    def setup(self, ctx=None):
        try:
            import torch
        except ImportError as e:
            raise NodeError("PyTorch is not installed (pip install torch)") from e
        self._torch = torch
        dev = self.get("device")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.build_model(torch)
        if self.get("weights"):
            model.load_state_dict(torch.load(self.get("weights"), map_location=dev))
        self._model = model.to(dev).eval()
        self._device = dev

    def process(self, ctx, inputs):
        if self._model is None:
            self.setup()
        torch = self._torch
        img = inputs["image"]
        import cv2
        data = img.data if not img.is_gray else cv2.cvtColor(img.data, cv2.COLOR_GRAY2BGR)
        s = int(self.get("size"))
        x = cv2.resize(data, (s, s))[:, :, ::-1].astype(np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self._device)
        with torch.no_grad():
            out = self._model(t).float().cpu().numpy()
        score = float(out.reshape(-1)[0])
        ctx.add_overlay(Overlay.text(8, 20, f"score {score:.3f}"))
        return {"output": out, "score": score}

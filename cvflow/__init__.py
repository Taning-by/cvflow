"""CVFlow - open, flow-based industrial machine vision platform.

Layers (bottom to top):
  cvflow.core      Qt-free data types, node model, graph, execution engine, runtime
  cvflow.camera    image acquisition abstraction (folder simulator, OpenCV, GenICam)
  cvflow.operators built-in algorithm nodes (source / preprocess / analysis / dl / logic / output)
  cvflow.comm      industrial communication (TCP/UDP/serial/Modbus) and trigger/send rules
  cvflow.ui        PySide6 desktop application (node editor, image view, parameter panel)
"""
__version__ = "0.1.0"

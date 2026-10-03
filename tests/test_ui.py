import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "solutions" / "demo_holes.json"
SHOT_DIR = Path(os.environ.get("CVFLOW_SHOT_DIR", "")) if os.environ.get("CVFLOW_SHOT_DIR") else None


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv[:1])


def _pump(app, n=5):
    for _ in range(n):
        app.processEvents()


def test_main_window_end_to_end(app, tmp_path):
    from cvflow.ui.main_window import MainWindow
    w = MainWindow(str(DEMO))
    w.resize(1500, 920)
    w.show()
    _pump(app)
    try:
        assert set(w.solution.flows) == {"main", "learning"}
        assert w.current_flow == "main"
        assert len(w.scene.node_items) == len(w.solution.flows["main"].nodes)
        assert len(w.scene.link_items) == len(w.solution.flows["main"].links)

        w.autorun.setChecked(False)
        w.run_once()
        _pump(app)
        w._apply_pending_result()
        assert w.results_panel.banner.text().startswith("OK")
        assert w.image_view.image is not None

        holes = w.solution.flows["main"].find_by_name("孔")
        w.scene.select_node(holes.id)
        _pump(app)
        assert w.selected_node == holes.id
        assert w.param_panel._node is holes
        assert len(w.image_view._overlay_items) >= 3  # blob rectangles

        w._on_param_changed(holes.id, "min_area", 100000)
        w.run_once()
        _pump(app)
        w._apply_pending_result()
        assert w.results_panel.banner.text().startswith("NG")
        w._on_param_changed(holes.id, "min_area", 500)

        # editing: add + link a node through the scene, then save and reload
        n = w.scene.add_node("preprocess.blur", QPointF(10, 400))
        assert n is not None and n.id in w.solution.flows["main"].nodes
        gray = w.solution.flows["main"].find_by_name("灰度")
        w.solution.flows["main"].add_link(gray.id, "image", n.id, "image")
        w.scene.set_graph(w.solution.flows["main"])
        out = tmp_path / "copy.json"
        assert w.save_solution(str(out)) and out.exists()

        # run mode start/stop
        w.toggle_run_mode(True)
        _pump(app)
        assert w.run_mode and w.runners["main"].running and w.comm.devices["plc"].connected
        r = w.runners["main"].trigger(wait=True, timeout=5)
        assert r is not None and r.status_text in ("OK", "NG")
        w.toggle_run_mode(False)
        _pump(app)
        assert not w.runners["main"].running

        # switch flow
        w.flow_combo.setCurrentText("learning")
        _pump(app)
        assert w.current_flow == "learning" and len(w.scene.node_items) == len(w.solution.flows["learning"].nodes)
        w.flow_combo.setCurrentText("main")
        _pump(app)
        w.scene.select_node(holes.id)
        w.run_once()
        _pump(app)
        w._apply_pending_result()
        if SHOT_DIR:
            SHOT_DIR.mkdir(parents=True, exist_ok=True)
            w.grab().save(str(SHOT_DIR / "main_window.png"))
    finally:
        w._dirty = False
        w.close()
        _pump(app)


def test_camera_dialog_and_comm_test_button(app, tmp_path, monkeypatch):
    from cvflow.camera.discovery import GigEDevice
    from cvflow.ui import camera_dialog
    from cvflow.ui.camera_dialog import CameraDialog
    from cvflow.ui.main_window import MainWindow
    monkeypatch.setattr(camera_dialog.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(camera_dialog.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    fake = [GigEDevice(ip="192.168.1.20", mac="C4:2F:90:11:22:33", model="MV-CA050-10GM", serial="DA1234567",
                       user_name="cam1", extra={"sources": ["gvcp"]})]
    w = MainWindow(str(DEMO))
    w.autorun.setChecked(False)
    try:
        dlg = CameraDialog(w, enumerator=lambda **kw: fake)
        got = []
        dlg.add_requested.connect(lambda k, s, n, c: got.append((k, s, n, c)))
        dlg._search()
        _pump(app)
        assert dlg.table.rowCount() == 1 and dlg.table.item(0, 3).text() == "DA1234567"
        dlg._add_to_flow()
        assert got and got[0][1] == "DA1234567" and got[0][2] == "cam1"
        before = len(w.scene.node_items)
        w._add_camera_node(*got[0])
        assert len(w.scene.node_items) == before + 1
        node = w.solution.flows["main"].nodes[w.selected_node]
        assert node.type_id == "source.camera" and node.get("source") == "DA1234567" and node.name.startswith("相机 cam1")
        ok, msg = w.comm.test_connection("plc")
        assert ok and "可用" in msg
    finally:
        w._dirty = False
        w.close()
        _pump(app)

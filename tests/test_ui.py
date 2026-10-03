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

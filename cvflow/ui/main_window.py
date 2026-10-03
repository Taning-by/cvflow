"""Main window: wires solution, flows, runners, communication and all panels together."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QPointF, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QTabWidget,
                               QToolBar, QVBoxLayout, QWidget)

from ..comm import CommManager, set_manager
from ..core import FlowRunner, Graph, Solution, Trigger, TriggerSource, events, registry
from ..core.engine import RunResult
from ..core.events import EventBus
from ..core.types import Image, Overlay, Rect
from .bridge import EventBridge
from .comm_panel import CommPanel
from .image_view import ImageView
from .log_panel import LogPanel
from .node_editor import NodeScene, NodeView
from .palette import NodePalette
from .param_panel import ParamPanel
from .results_panel import ResultsPanel
from .variables_panel import VariablesPanel

log = logging.getLogger("cvflow.ui")


class MainWindow(QMainWindow):
    def __init__(self, solution_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("CVFlow")
        self.resize(1500, 920)
        registry.load_builtins()

        self.solution: Solution | None = None
        self.runners: dict[str, FlowRunner] = {}
        self.comm: CommManager | None = None
        self.bridge: EventBridge | None = None
        self.current_flow = ""
        self.selected_node: str | None = None
        self.run_mode = False
        self._dirty = False
        self._pending_result: RunResult | None = None

        self._autorun_timer = QTimer(self)
        self._autorun_timer.setSingleShot(True)
        self._autorun_timer.setInterval(150)
        self._autorun_timer.timeout.connect(self.run_once)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(40)
        self._refresh_timer.timeout.connect(self._apply_pending_result)
        self._edit_loop_timer = QTimer(self)
        self._edit_loop_timer.timeout.connect(self.run_once)
        self._vars_timer = QTimer(self)
        self._vars_timer.setSingleShot(True)
        self._vars_timer.setInterval(200)

        self._build_ui()
        self._build_toolbar()
        self._build_menu()
        self._vars_timer.timeout.connect(self.variables_panel.refresh)

        s = QSettings()
        if s.value("geometry") is not None:
            self.restoreGeometry(s.value("geometry"))
        if s.value("state") is not None:
            self.restoreState(s.value("state"))

        if solution_path:
            self.open_solution(solution_path)
        else:
            self.new_solution()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self) -> None:
        self.scene = NodeScene(self)
        self.view = NodeView(self.scene)
        self.setCentralWidget(self.view)
        self.scene.node_selected.connect(self._on_node_selected)
        self.scene.graph_changed.connect(self._on_graph_changed)
        self.scene.message.connect(lambda m: self.statusBar().showMessage(m, 4000))

        self.palette = NodePalette()
        self.palette.node_activated.connect(self._add_node_at_center)
        self._dock("Nodes", self.palette, Qt.LeftDockWidgetArea, "dock_palette")

        img_w = QWidget()
        il = QVBoxLayout(img_w)
        il.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.image_title = QLabel("—")
        fit = QPushButton("Fit")
        fit.setFixedWidth(40)
        self.all_overlays = QCheckBox("All overlays")
        self.all_overlays.toggled.connect(lambda _: self._update_image())
        bar.addWidget(self.image_title, 1)
        bar.addWidget(self.all_overlays)
        bar.addWidget(fit)
        self.image_view = ImageView()
        fit.clicked.connect(self.image_view.fit)
        self.pixel_label = QLabel("")
        self.image_view.pixel_info.connect(self.pixel_label.setText)
        il.addLayout(bar)
        il.addWidget(self.image_view, 1)
        il.addWidget(self.pixel_label)
        self._dock("Image", img_w, Qt.RightDockWidgetArea, "dock_image")

        self.param_panel = ParamPanel()
        self.param_panel.param_changed.connect(self._on_param_changed)
        self.param_panel.node_renamed.connect(self._on_node_renamed)
        self.param_panel.node_enabled_changed.connect(self.scene.set_node_enabled)
        self.param_panel.roi_edit_requested.connect(self._begin_roi_edit)
        self.param_panel.roi_show_requested.connect(self._show_roi)
        self._dock("Parameters", self.param_panel, Qt.RightDockWidgetArea, "dock_params")

        self.results_panel = ResultsPanel()
        self.log_panel = LogPanel()
        self.comm_panel = CommPanel(CommManager(EventBus()), lambda: list(self.solution.flows) if self.solution else [])
        self.comm_panel.config_changed.connect(self._mark_dirty)
        self.variables_panel = VariablesPanel()
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self.results_panel, "Results")
        self.bottom_tabs.addTab(self.comm_panel, "Communication")
        self.bottom_tabs.addTab(self.variables_panel, "Variables")
        self.bottom_tabs.addTab(self.log_panel, "Log")
        self._dock("Output", self.bottom_tabs, Qt.BottomDockWidgetArea, "dock_bottom")

        self.status_run = QLabel("edit mode")
        self.status_stats = QLabel("")
        self.statusBar().addPermanentWidget(self.status_stats)
        self.statusBar().addPermanentWidget(self.status_run)

    def _dock(self, title: str, widget: QWidget, area, name: str) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        self.addDockWidget(area, d)
        return d

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("toolbar_main")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_new = QAction("New", self, shortcut=QKeySequence.New, triggered=self.new_solution)
        self.act_open = QAction("Open…", self, shortcut=QKeySequence.Open, triggered=self._open_dialog)
        self.act_save = QAction("Save", self, shortcut=QKeySequence.Save, triggered=lambda: self.save_solution())
        for a in (self.act_new, self.act_open, self.act_save):
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" Flow: "))
        self.flow_combo = QComboBox()
        self.flow_combo.setMinimumWidth(140)
        self.flow_combo.currentTextChanged.connect(self._set_current_flow)
        tb.addWidget(self.flow_combo)
        self.act_add_flow = QAction("+", self, toolTip="Add flow", triggered=self._add_flow)
        self.act_del_flow = QAction("−", self, toolTip="Remove flow", triggered=self._remove_flow)
        tb.addAction(self.act_add_flow)
        tb.addAction(self.act_del_flow)
        tb.addSeparator()
        self.act_run = QAction("▶ Run once", self, shortcut="F5", triggered=self.run_once)
        tb.addAction(self.act_run)
        self.autorun = QCheckBox("Auto-run on change")
        self.autorun.setChecked(True)
        tb.addWidget(self.autorun)
        self.continuous = QCheckBox("Continuous")
        self.continuous.toggled.connect(self._apply_continuous)
        tb.addWidget(self.continuous)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0.0, 3600)
        self.interval.setDecimals(2)
        self.interval.setValue(0.5)
        self.interval.setSuffix(" s")
        self.interval.setToolTip("Interval for continuous runs")
        self.interval.valueChanged.connect(lambda _: self._apply_continuous(self.continuous.isChecked()))
        tb.addWidget(self.interval)
        tb.addSeparator()
        self.act_run_mode = QAction("● Start run mode", self, shortcut="F9", checkable=True)
        self.act_run_mode.toggled.connect(self.toggle_run_mode)
        tb.addAction(self.act_run_mode)
        self.act_fit = QAction("Fit", self, shortcut="F", triggered=self.view.fit_all)
        tb.addAction(self.act_fit)

    def _build_menu(self) -> None:
        m = self.menuBar()
        f = m.addMenu("&File")
        f.addActions([self.act_new, self.act_open, self.act_save])
        f.addAction(QAction("Save &As…", self, shortcut=QKeySequence.SaveAs, triggered=self._save_as_dialog))
        f.addSeparator()
        f.addAction(QAction("E&xit", self, shortcut=QKeySequence.Quit, triggered=self.close))
        fl = m.addMenu("&Flow")
        fl.addActions([self.act_add_flow, self.act_del_flow])
        fl.addAction(QAction("Rename flow…", self, triggered=self._rename_flow))
        fl.addSeparator()
        fl.addAction(self.act_run)
        fl.addAction(self.act_run_mode)
        fl.addAction(QAction("Validate flow", self, triggered=self._validate))
        p = m.addMenu("&Plugins")
        p.addAction(QAction("Load plugin folder…", self, triggered=self._load_plugin_dir))
        p.addAction(QAction("Reload plugins", self, triggered=self._reload_plugins))
        h = m.addMenu("&Help")
        h.addAction(QAction("About", self, triggered=lambda: QMessageBox.about(
            self, "CVFlow", "CVFlow – open, flow-based industrial vision platform.\n\n"
                             "Nodes are Python plugins; flows talk to PLCs over TCP/UDP/serial/Modbus.")))

    # ------------------------------------------------------------------ solution lifecycle
    def _teardown_current(self) -> None:
        if self.run_mode:
            self.act_run_mode.setChecked(False)
        self._edit_loop_timer.stop()
        for r in self.runners.values():
            r.stop()
        if self.comm is not None:
            self.comm.shutdown()
        if self.bridge is not None:
            self.bridge.detach()
        set_manager(None)
        self.runners = {}

    def _bind_solution(self, sol: Solution) -> None:
        self._teardown_current()
        self.solution = sol
        self.bridge = EventBridge(sol.bus, self)
        self.bridge.event.connect(self._on_bus_event)
        self.comm = CommManager(sol.bus, sol.variables)
        for e in self.comm.load_dict(sol.comm_config):
            log.error("comm config: %s", e)
        set_manager(self.comm)
        self._sync_runners()
        self.comm_panel.set_manager(self.comm)
        self.variables_panel.set_variables(sol.variables)
        self.palette.reload()
        self.flow_combo.blockSignals(True)
        self.flow_combo.clear()
        self.flow_combo.addItems(list(sol.flows))
        self.flow_combo.blockSignals(False)
        self.current_flow = ""
        self._set_current_flow(next(iter(sol.flows), ""))
        self._dirty = False
        self._update_title()

    def _sync_runners(self) -> None:
        assert self.solution is not None
        for name in list(self.runners):
            if name not in self.solution.flows:
                self.runners.pop(name).stop()
        for name, g in self.solution.flows.items():
            if name not in self.runners or self.runners[name].graph is not g:
                self.runners[name] = FlowRunner(g, self.solution.variables, self.solution.bus)
        if self.comm is not None:
            self.comm.set_runners(self.runners)

    def new_solution(self) -> None:
        if not self._confirm_discard():
            return
        sol = Solution("untitled")
        sol.add_flow(Graph("main"))
        self._bind_solution(sol)

    def open_solution(self, path: str) -> None:
        if not self._confirm_discard():
            return
        try:
            sol = Solution.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", f"{path}\n\n{type(e).__name__}: {e}")
            return
        self._bind_solution(sol)
        QSettings().setValue("last_file", path)
        self.statusBar().showMessage(f"opened {path}", 4000)

    def save_solution(self, path: str | None = None) -> bool:
        assert self.solution is not None
        path = path or (str(self.solution.path) if self.solution.path else None)
        if not path:
            return self._save_as_dialog()
        if self.comm is not None:
            self.solution.comm_config = self.comm.to_dict()
        try:
            self.solution.save(path)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return False
        self._dirty = False
        self._update_title()
        QSettings().setValue("last_file", path)
        self.statusBar().showMessage(f"saved {path}", 3000)
        return True

    def _open_dialog(self) -> None:
        start = QSettings().value("last_file", "") or ""
        path, _ = QFileDialog.getOpenFileName(self, "Open solution", str(Path(start).parent) if start else "", "CVFlow solution (*.json)")
        if path:
            self.open_solution(path)

    def _save_as_dialog(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save solution", "", "CVFlow solution (*.json)")
        if not path:
            return False
        if not path.endswith(".json"):
            path += ".json"
        return self.save_solution(path)

    def _confirm_discard(self) -> bool:
        if not self._dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save changes to the current solution?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save_solution()
        return True

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = str(self.solution.path.name) if self.solution and self.solution.path else (self.solution.name if self.solution else "")
        self.setWindowTitle(f"CVFlow – {name}{' *' if self._dirty else ''}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        s = QSettings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("state", self.saveState())
        self._teardown_current()
        self.log_panel.detach()
        event.accept()

    # ------------------------------------------------------------------ flows
    @property
    def graph(self) -> Graph | None:
        return self.solution.flows.get(self.current_flow) if self.solution else None

    @property
    def runner(self) -> FlowRunner | None:
        return self.runners.get(self.current_flow)

    def _set_current_flow(self, name: str) -> None:
        if not self.solution or not name or name not in self.solution.flows:
            return
        self.current_flow = name
        self.scene.set_graph(self.solution.flows[name])
        self.scene.set_locked(self.run_mode)
        self.view.fit_all()
        self.selected_node = None
        self.param_panel.set_node(None, None)
        r = self.runner
        self.results_panel.show_result(r.last_result if r else None, self.graph)
        self._update_image()
        if self.flow_combo.currentText() != name:
            self.flow_combo.blockSignals(True)
            self.flow_combo.setCurrentText(name)
            self.flow_combo.blockSignals(False)

    def _add_flow(self) -> None:
        if self.run_mode or not self.solution:
            return
        name, ok = QInputDialog.getText(self, "New flow", "Flow name:", text=f"flow{len(self.solution.flows) + 1}")
        if ok and name.strip():
            g = self.solution.add_flow(Graph(name.strip()))
            self._sync_runners()
            self.flow_combo.addItem(g.name)
            self.flow_combo.setCurrentText(g.name)
            self._mark_dirty()

    def _remove_flow(self) -> None:
        if self.run_mode or not self.solution or len(self.solution.flows) <= 1:
            return
        if QMessageBox.question(self, "Remove flow", f"Remove flow '{self.current_flow}'?") != QMessageBox.Yes:
            return
        self.solution.remove_flow(self.current_flow)
        self._sync_runners()
        self.flow_combo.removeItem(self.flow_combo.currentIndex())
        self._mark_dirty()

    def _rename_flow(self) -> None:
        if self.run_mode or not self.solution:
            return
        name, ok = QInputDialog.getText(self, "Rename flow", "New name:", text=self.current_flow)
        if ok and name.strip() and name.strip() != self.current_flow:
            try:
                self.solution.rename_flow(self.current_flow, name.strip())
            except ValueError as e:
                QMessageBox.warning(self, "Rename", str(e))
                return
            self._sync_runners()
            idx = self.flow_combo.currentIndex()
            self.flow_combo.setItemText(idx, name.strip())
            self.current_flow = name.strip()
            self._mark_dirty()

    def _validate(self) -> None:
        g = self.graph
        if g is None:
            return
        w = g.validate()
        QMessageBox.information(self, "Validate", "\n".join(w) if w else "No problems found.")

    # ------------------------------------------------------------------ editing callbacks
    def _add_node_at_center(self, type_id: str) -> None:
        pos = self.view.mapToScene(self.view.viewport().rect().center())
        self.scene.add_node(type_id, QPointF(pos.x() - 80, pos.y() - 30))

    def _on_graph_changed(self) -> None:
        self._mark_dirty()
        self._schedule_autorun()

    def _on_node_selected(self, node_id) -> None:
        self.selected_node = node_id
        g = self.graph
        node = g.nodes.get(node_id) if (g and node_id) else None
        self.param_panel.set_node(node, g)
        self._update_image()

    def _on_param_changed(self, node_id: str, name: str, value) -> None:
        g = self.graph
        if g is None or node_id not in g.nodes:
            return
        node = g.nodes[node_id]
        try:
            node.set(name, value)
        except (KeyError, ValueError) as e:
            self.statusBar().showMessage(str(e), 4000)
            return
        item = self.scene.node_items.get(node_id)
        if item is not None:
            item.update()
        if name == "roi" or node.param_def(name).kind == "rect":
            self.param_panel.set_node(node, g)
        self._mark_dirty()
        self._schedule_autorun()

    def _on_node_renamed(self, node_id: str, name: str) -> None:
        self.scene.rename_node(node_id, name)
        g = self.graph
        if g:
            self.param_panel.set_node(g.nodes[node_id], g)

    def _schedule_autorun(self) -> None:
        if self.autorun.isChecked() and not self.run_mode:
            self._autorun_timer.start()

    # ------------------------------------------------------------------ ROI editing
    def _input_image(self, node_id: str, result: RunResult | None) -> Image | None:
        """Find the image flowing into ``node_id`` (first IMAGE input, following links upstream)."""
        g = self.graph
        if g is None or result is None or node_id not in g.nodes:
            return None
        node = g.nodes[node_id]
        for port in node.inputs:
            link = g.input_links(node_id).get(port.name)
            if link is None:
                continue
            up = result.node_results.get(link.src_node)
            if up is not None:
                v = up.outputs.get(link.src_port)
                if isinstance(v, Image):
                    return v
        return None

    def _begin_roi_edit(self, node_id: str, param: str) -> None:
        r = self.runner
        res = r.last_result if r else None
        img = self._input_image(node_id, res)
        if img is None and res is None:
            self.run_once()
            res = self.runner.last_result if self.runner else None
            img = self._input_image(node_id, res)
        if img is not None:
            self.image_view.set_image(img)
            self.image_view.set_overlays([])
            self.image_title.setText(f"{self.graph.nodes[node_id].name} – draw {param}")
        self.image_view.begin_roi_edit(lambda rect, n=node_id, p=param: self._on_param_changed(n, p, rect))

    def _show_roi(self, node_id: str, param: str) -> None:
        g = self.graph
        if g is None:
            return
        node = g.nodes[node_id]
        rect = node.rect(param)
        r = self.runner
        img = self._input_image(node_id, r.last_result if r else None)
        if img is not None:
            self.image_view.set_image(img)
        self.image_view.set_overlays([Overlay.rect(rect, "#ffa500", param)] if rect else [])

    # ------------------------------------------------------------------ running
    def run_once(self) -> None:
        r = self.runner
        if r is None:
            return
        if self.run_mode:
            r.trigger(Trigger(TriggerSource.MANUAL))
            return
        QApplication.setOverrideCursor(Qt.BusyCursor)
        try:
            r.run_once(Trigger(TriggerSource.MANUAL))
        finally:
            QApplication.restoreOverrideCursor()

    def toggle_run_mode(self, on: bool) -> None:
        if on == self.run_mode:
            return
        self.run_mode = on
        if on:
            self._edit_loop_timer.stop()
            errors = self.comm.connect_all() if self.comm else []
            for name, r in self.runners.items():
                errors += [f"[{name}] {e}" for e in r.start()]
            if errors:
                QMessageBox.warning(self, "Run mode", "Started with problems:\n" + "\n".join(errors))
            self._apply_continuous(self.continuous.isChecked())
            self.act_run_mode.setText("■ Stop run mode")
            self.status_run.setText("RUN MODE")
            self.status_run.setStyleSheet("color:#66bb6a; font-weight:bold")
        else:
            for r in self.runners.values():
                r.stop()
            self.act_run_mode.setText("● Start run mode")
            self.status_run.setText("edit mode")
            self.status_run.setStyleSheet("")
            self._apply_continuous(self.continuous.isChecked())
        self.scene.set_locked(on)
        self.param_panel.set_locked(on)
        self.palette.setEnabled(not on)
        for a in (self.act_add_flow, self.act_del_flow, self.act_new, self.act_open):
            a.setEnabled(not on)
        if self.act_run_mode.isChecked() != on:
            self.act_run_mode.setChecked(on)
        self.comm_panel.refresh()

    def _apply_continuous(self, on: bool) -> None:
        interval = max(0.0, float(self.interval.value()))
        if self.run_mode:
            self._edit_loop_timer.stop()
            for name, r in self.runners.items():
                r.set_continuous(interval if (on and name == self.current_flow) else None)
        else:
            if on:
                self._edit_loop_timer.start(max(10, int(interval * 1000)))
            else:
                self._edit_loop_timer.stop()

    # ------------------------------------------------------------------ bus events (GUI thread)
    def _on_bus_event(self, event: str, payload: dict) -> None:
        if event == events.RUN_FINISHED:
            result: RunResult = payload["result"]
            if result.flow == self.current_flow:
                self._pending_result = result
                if not self._refresh_timer.isActive():
                    self._refresh_timer.start()
        elif event == events.RUNNER_STATE:
            pass
        elif event.startswith("comm_"):
            self.comm_panel.on_event(event, payload)
        elif event == events.VAR_CHANGED:
            if not self._vars_timer.isActive():
                self._vars_timer.start()

    def _apply_pending_result(self) -> None:
        result = self._pending_result
        self._pending_result = None
        if result is None:
            return
        self.scene.refresh_status()
        self.results_panel.show_result(result, self.graph)
        self.param_panel.refresh()
        self._update_image(result)
        r = self.runner
        st = r.stats if r else None
        if st:
            self.status_stats.setText(f"runs {st.count}  OK {st.ok}  NG {st.ng}  ERR {st.error}  avg {st.avg_ms:.1f} ms  ")
        self.statusBar().showMessage(f"run {result.run_id}: {result.status_text} in {result.duration_ms:.1f} ms"
                                     + (f" – {result.error}" if result.error else ""), 5000)

    def _update_image(self, result: RunResult | None = None) -> None:
        r = self.runner
        result = result or (r.last_result if r else None)
        g = self.graph
        if result is None or g is None or self.image_view.roi_editing:
            if result is None:
                self.image_view.clear()
                self.image_title.setText("—")
            return
        nid = self.selected_node
        if nid is None or nid not in g.nodes:
            # default: the last node in execution order that produced an image
            nid = None
            for cand in reversed(list(result.node_results)):
                if any(isinstance(v, Image) for v in result.node_results[cand].outputs.values()):
                    nid = cand
                    break
            if nid is None:
                self.image_view.clear()
                return
        nr = result.node_results.get(nid)
        node = g.nodes[nid]
        out_img = next((v for v in nr.outputs.values() if isinstance(v, Image)), None) if nr else None
        in_img = self._input_image(nid, result)
        overlays = list(nr.overlays) if nr else []
        if self.all_overlays.isChecked():
            overlays = [ov for res in result.node_results.values() for ov in res.overlays]
            img = in_img or out_img
        else:
            img = in_img if (overlays and in_img is not None) else (out_img or in_img)
        if img is None:
            self.image_view.clear()
            self.image_title.setText(f"{node.name}: no image")
            return
        self.image_view.set_image(img)
        self.image_view.set_overlays(overlays)
        self.image_title.setText(f"{node.name}  ({img.width}×{img.height}, frame {img.frame_id})")

    # ------------------------------------------------------------------ plugins
    def _load_plugin_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Plugin folder")
        if not d or not self.solution:
            return
        found = registry.load_plugin_dir(d)
        if d not in self.solution.plugin_dirs:
            self.solution.plugin_dirs.append(d)
            self._mark_dirty()
        self.palette.reload()
        self.statusBar().showMessage(f"loaded {len(found)} node type(s) from {d}", 5000)

    def _reload_plugins(self) -> None:
        if not self.solution:
            return
        n = 0
        base = self.solution.path.parent if self.solution.path else Path.cwd()
        for d in self.solution.plugin_dirs:
            p = Path(d)
            n += len(registry.load_plugin_dir(p if p.is_absolute() else base / p))
        self.palette.reload()
        self.statusBar().showMessage(f"reloaded {n} plugin node type(s)", 5000)

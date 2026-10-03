"""界面黑盒测试：通过用户可见的操作（菜单、工具栏、鼠标、键盘、对话框、表格编辑）驱动界面，
检查用户可见的结果。测试函数命名 test_ui_<功能区>_<场景>，报告脚本据此归类。
功能区：startup file flow palette editor params run image results variables log comm camera plugins help
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QDropEvent, QMouseEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QLabel,  # noqa: E402
                               QLineEdit, QMenu, QPlainTextEdit, QPushButton, QSpinBox)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "solutions" / "demo_holes.json"


# ---------------------------------------------------------------- 公共夹具
@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication(sys.argv[:1])
    a.setOrganizationName("cvflow-bbtest")
    a.setApplicationName("cvflow-bbtest")
    from cvflow.ui.theme import apply_theme
    apply_theme(a)
    return a


@pytest.fixture
def settings(app, tmp_path):
    """把 QSettings 隔离到临时目录，测试之间不共享“上次打开的文件”等状态。"""
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    QSettings().clear()
    yield QSettings()


class Modal:
    """记录模态对话框调用并立即返回预设值（离屏环境里真正弹窗会阻塞）。"""

    def __init__(self):
        self.calls: list[tuple] = []
        self.answers: dict = {}

    def stub(self, name, default):
        def fn(*a, **k):
            self.calls.append((name, a[1:] if a and not isinstance(a[0], str) else a))
            v = self.answers.get(name, default)
            return v() if callable(v) else v
        return staticmethod(fn)

    def texts(self, name):
        return [" ".join(str(x) for x in c[1]) for c in self.calls if c[0] == name]


@pytest.fixture
def modal(monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
    from cvflow.ui import camera_dialog, comm_panel, main_window, param_panel, variables_panel
    m = Modal()
    for mod in (main_window, comm_panel, camera_dialog, variables_panel, param_panel):
        for cls_name, attrs in (("QMessageBox", {"information": QMessageBox.Ok, "warning": QMessageBox.Ok,
                                                 "critical": QMessageBox.Ok, "question": QMessageBox.Yes, "about": None}),
                                ("QFileDialog", {"getOpenFileName": ("", ""), "getSaveFileName": ("", ""), "getExistingDirectory": ""}),
                                ("QInputDialog", {"getText": ("", False)})):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            for attr, default in attrs.items():
                monkeypatch.setattr(cls, attr, m.stub(attr, default))
    return m


def pump(app, ms=0):
    """处理事件并等待 ms 毫秒（手动循环，不依赖 QTest.qWait 在离屏平台上的行为）。"""
    app.processEvents()
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def plain(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>|&nbsp;", " ", text)


def spin(panel, label_text):
    w = param_widget(panel, label_text)
    return w if isinstance(w, (QSpinBox, QDoubleSpinBox)) else w.findChild(QSpinBox) or w.findChild(QDoubleSpinBox)


@pytest.fixture
def win(app, settings, modal):
    from cvflow.ui.main_window import MainWindow
    w = MainWindow(str(DEMO))
    w.resize(1500, 920)
    w.show()
    w.autorun.setChecked(False)
    pump(app)
    yield w
    w._dirty = False
    w.toggle_run_mode(False)
    w.close()
    pump(app)


# ---------------------------------------------------------------- 操作工具（只用用户可见的途径）
def action(widget, text: str):
    for a in widget.findChildren(type(widget.menuBar().actions()[0])) if hasattr(widget, "menuBar") else []:
        if a.text() == text:
            return a
    raise LookupError(text)


def menu_action(win, menu_text: str, item_text: str):
    for m in win.menuBar().actions():
        if m.text() == menu_text:
            for a in m.menu().actions():
                if a.text() == item_text:
                    return a
    raise LookupError(f"{menu_text} > {item_text}")


def toolbar_action(win, text: str):
    for a in win.findChild(type(win.children()[0]), "toolbar_main").actions() if False else win.toolbar_actions():
        if a.text() == text:
            return a
    raise LookupError(text)


def button(parent, text: str) -> QPushButton:
    for b in parent.findChildren(QPushButton):
        if b.text() == text and b.isVisibleTo(parent):
            return b
    raise LookupError(text)


def param_widget(panel, label_text: str):
    """参数面板里某个标签对应的编辑控件。"""
    for form in panel.widget().findChildren(QFormLayout):
        for row in range(form.rowCount()):
            lab = form.itemAt(row, QFormLayout.LabelRole)
            fld = form.itemAt(row, QFormLayout.FieldRole)
            if lab and fld and isinstance(lab.widget(), QLabel) and lab.widget().text() == label_text:
                return fld.widget()
    raise LookupError(label_text)


def mouse(app, widget, kind, pos: QPoint, button=Qt.LeftButton, buttons=None, mods=Qt.NoModifier):
    buttons = button if buttons is None else buttons
    if kind == QEvent.MouseButtonRelease:
        buttons = Qt.NoButton
    ev = QMouseEvent(kind, QPointF(pos), widget.mapToGlobal(pos).toPointF(), button, buttons, mods)
    app.sendEvent(widget, ev)
    app.processEvents()


def drag(app, widget, p1: QPoint, p2: QPoint, button=Qt.LeftButton, steps=6):
    mouse(app, widget, QEvent.MouseButtonPress, p1, button)
    for i in range(1, steps + 1):
        p = QPoint(p1.x() + (p2.x() - p1.x()) * i // steps, p1.y() + (p2.y() - p1.y()) * i // steps)
        mouse(app, widget, QEvent.MouseMove, p, Qt.NoButton, button)
    mouse(app, widget, QEvent.MouseButtonRelease, p2, button)


def view_pos(view, scene_pt: QPointF) -> QPoint:
    return view.mapFromScene(scene_pt)


def node_by_name(win, name):
    return win.solution.flows[win.current_flow].find_by_name(name)


def node_item(win, name):
    return win.scene.node_items[node_by_name(win, name).id]


# =============================================================== 启动
def test_ui_startup_layout_and_title(win):
    assert "demo_holes.json" in win.windowTitle()
    docks = {d.windowTitle() for d in win.findChildren(type(win.findChild(type(win.palette.parent()))))} if False else None
    titles = {d.windowTitle() for d in win.findChildren(__import__("PySide6.QtWidgets", fromlist=["QDockWidget"]).QDockWidget)}
    assert {"节点库", "图像", "参数", "输出"} <= titles
    assert [win.bottom_tabs.tabText(i) for i in range(win.bottom_tabs.count())] == ["结果", "通信", "变量", "日志"]
    assert win.flow_combo.currentText() == "main" and win.flow_combo.count() == 2
    assert len(win.scene.node_items) == 11 and len(win.scene.link_items) == 13
    assert win.status_run.text() == "编辑模式"


def test_ui_startup_without_file_reopens_last(app, settings, modal):
    from cvflow.ui.main_window import MainWindow
    w = MainWindow(str(DEMO)); w.close(); pump(app)
    w2 = MainWindow(None); pump(app)
    try:
        assert w2.solution.path is not None and w2.solution.path.name == "demo_holes.json"
    finally:
        w2.close(); pump(app)
    settings.clear()
    w3 = MainWindow(None); pump(app)
    try:
        assert w3.solution.path is None and win_title_has(w3, "untitled") and len(w3.scene.node_items) == 0
    finally:
        w3.close(); pump(app)


def win_title_has(w, text):
    return text in w.windowTitle()


# =============================================================== 文件
def test_ui_file_new_open_save_saveas_recent(win, modal, tmp_path, app):
    # 另存为 → 文件生成、标题变化、不再带 *
    out = tmp_path / "copy.json"
    modal.answers["getSaveFileName"] = (str(out), "")
    menu_action(win, "文件(&F)", "另存为(&A)…").trigger(); pump(app)
    assert out.exists() and win.windowTitle().endswith("copy.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert {f["name"] for f in data["flows"]} == {"main", "learning"}
    # 修改后标题带 *，保存后消失
    win.scene.add_node("preprocess.blur", QPointF(0, 0)); pump(app)
    assert win.windowTitle().endswith("*")
    menu_action(win, "文件(&F)", "保存").trigger(); pump(app)
    assert not win.windowTitle().endswith("*")
    assert any(n["type"] == "preprocess.blur" for n in json.loads(out.read_text(encoding="utf-8"))["flows"][0]["nodes"])
    # 最近打开列出该文件
    recent = menu_action(win, "文件(&F)", "最近打开").menu()
    recent.aboutToShow.emit(); pump(app)
    assert any(a.text() == str(out.resolve()) for a in recent.actions())
    # 新建 → 空方案
    menu_action(win, "文件(&F)", "新建").trigger(); pump(app)
    assert len(win.scene.node_items) == 0 and "untitled" in win.windowTitle()
    # 打开 → 回到示例
    modal.answers["getOpenFileName"] = (str(DEMO), "")
    menu_action(win, "文件(&F)", "打开…").trigger(); pump(app)
    assert len(win.scene.node_items) == 11 and "demo_holes" in win.windowTitle()
    # 打开失败 → 错误提示
    modal.answers["getOpenFileName"] = (str(tmp_path / "nope.json"), "")
    menu_action(win, "文件(&F)", "打开…").trigger(); pump(app)
    assert modal.texts("critical") and "打开失败" in modal.texts("critical")[-1]


def test_ui_file_close_with_unsaved_changes_asks(win, modal, app):
    from PySide6.QtWidgets import QMessageBox
    win.scene.add_node("preprocess.blur", QPointF(0, 0)); pump(app)
    modal.answers["question"] = QMessageBox.Cancel
    assert win.close() is False and win.isVisible()
    modal.answers["question"] = QMessageBox.Discard
    assert win.close() is True


def test_ui_file_create_shortcut_menu(win, modal, monkeypatch, app):
    from cvflow import launcher
    monkeypatch.setattr(launcher, "create_shortcut", lambda sol=None, name="CVFlow": Path("/tmp/CVFlow.desktop"))
    menu_action(win, "文件(&F)", "创建桌面快捷方式…").trigger(); pump(app)
    assert modal.texts("information") and "CVFlow.desktop" in modal.texts("information")[-1]


# =============================================================== 流程
def test_ui_flow_add_rename_remove_switch_validate(win, modal, app):
    modal.answers["getText"] = ("检测2", True)
    menu_action(win, "流程(&L)", "").trigger() if False else None
    win.act_add_flow.trigger(); pump(app)
    assert win.flow_combo.currentText() == "检测2" and len(win.scene.node_items) == 0 and "检测2" in win.solution.flows
    modal.answers["getText"] = ("检测3", True)
    menu_action(win, "流程(&L)", "重命名流程…").trigger(); pump(app)
    assert "检测3" in win.solution.flows and win.flow_combo.currentText() == "检测3"
    # 在空流程里放一个阈值节点（输入未连接）→ 检查流程给出提示
    win.scene.add_node("preprocess.threshold", QPointF(0, 0)); pump(app)
    menu_action(win, "流程(&L)", "检查流程").trigger(); pump(app)
    assert "未连接" in modal.texts("information")[-1]
    win.flow_combo.setCurrentText("main"); pump(app)
    assert len(win.scene.node_items) == 11
    win.flow_combo.setCurrentText("检测3"); pump(app)
    win.act_del_flow.trigger(); pump(app)
    assert "检测3" not in win.solution.flows and win.flow_combo.count() == 2


# =============================================================== 节点库
def test_ui_palette_filter_and_double_click_adds_node(win, app):
    tree = win.palette.tree
    win.palette.filter.setText("二维码"); pump(app)
    visible = [tree.topLevelItem(i).child(j).text(0) for i in range(tree.topLevelItemCount())
               for j in range(tree.topLevelItem(i).childCount()) if not tree.topLevelItem(i).child(j).isHidden() and not tree.topLevelItem(i).isHidden()]
    assert visible == ["二维码"]
    win.palette.filter.setText(""); pump(app)
    item = next(tree.topLevelItem(i).child(j) for i in range(tree.topLevelItemCount()) for j in range(tree.topLevelItem(i).childCount())
                if tree.topLevelItem(i).child(j).text(0) == "二维码")
    tree.scrollToItem(item)
    before = len(win.scene.node_items)
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, tree.visualItemRect(item).center())
    QTest.mouseDClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, tree.visualItemRect(item).center()); pump(app)
    assert len(win.scene.node_items) == before + 1
    assert node_by_name(win, "二维码") is not None and win.param_panel._node.type_id == "analysis.qrcode"


def test_ui_palette_drag_drop_into_editor(win, app):
    from cvflow.ui.node_editor.view import MIME
    mime = QMimeData(); mime.setData(MIME, b"preprocess.canny")
    before = len(win.scene.node_items)
    ev = QDropEvent(QPointF(200, 150), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    win.view.dropEvent(ev); pump(app)      # 离屏平台没有真实拖放会话，直接把放下事件交给视图
    assert len(win.scene.node_items) == before + 1
    n = node_by_name(win, "Canny 边缘")
    assert n is not None and abs(n.position[0] - win.view.mapToScene(QPoint(200, 150)).x()) < 1


# =============================================================== 节点编辑器
def test_ui_editor_select_move_delete_node(win, app):
    view = win.view
    item = node_item(win, "灰度")
    center = view_pos(view, item.mapToScene(QPointF(80, 12)))
    mouse(app, view.viewport(), QEvent.MouseButtonPress, center); mouse(app, view.viewport(), QEvent.MouseButtonRelease, center)
    assert win.param_panel._node.name == "灰度" and item.isSelected()
    start = list(item.node.position)
    drag(app, view.viewport(), center, QPoint(center.x() + 60, center.y() + 40))
    assert item.node.position[0] > start[0] + 30 and item.node.position[1] > start[1] + 20
    n_links = len(win.scene.link_items)
    QTest.keyClick(view, Qt.Key_Delete); pump(app)
    assert node_by_name(win, "灰度") is None and len(win.scene.link_items) < n_links


def test_ui_editor_connect_ports_by_drag_and_reject_type_mismatch(win, app):
    view = win.view
    win.scene.add_node("preprocess.canny", QPointF(600, 450)); pump(app)
    src = node_item(win, "取图").outputs["image"]
    dst = node_item(win, "Canny 边缘").inputs["image"]
    before = len(win.scene.link_items)
    drag(app, view.viewport(), view_pos(view, src.scene_center()), view_pos(view, dst.scene_center()))
    assert len(win.scene.link_items) == before + 1
    g = win.solution.flows["main"]
    assert any(l.dst_node == node_by_name(win, "Canny 边缘").id and l.src_node == node_by_name(win, "取图").id for l in g.links)
    # 类型不匹配：STRING(path) → IMAGE 输入
    src2 = node_item(win, "取图").outputs["path"]
    drag(app, view.viewport(), view_pos(view, src2.scene_center()), view_pos(view, dst.scene_center()))
    assert len(win.scene.link_items) == before + 1 and "类型不匹配" in win.statusBar().currentMessage()
    # 把已有的输入连线拖到空白处 → 断开
    drag(app, view.viewport(), view_pos(view, dst.scene_center()), view_pos(view, dst.scene_center() + QPointF(0, 120)))
    assert len(win.scene.link_items) == before


def test_ui_editor_context_menu_add_duplicate_disable(win, app):
    view = win.view
    empty = QPoint(40, 40)
    mouse(app, view.viewport(), QEvent.MouseButtonPress, empty, Qt.RightButton); mouse(app, view.viewport(), QEvent.MouseButtonRelease, empty, Qt.RightButton)
    menus = [m for m in view.findChildren(QMenu) if m.isVisible()]
    assert menus
    add = next(a for a in menus[0].actions() if a.text() == "添加节点").menu()
    cat = next(a for a in add.actions() if a.text() == "预处理").menu()
    before = len(win.scene.node_items)
    next(a for a in cat.actions() if a.text() == "滤波").trigger(); menus[0].close(); pump(app)
    assert len(win.scene.node_items) == before + 1 and node_by_name(win, "滤波") is not None
    # 在节点上右键 → 复制 / 禁用
    item = node_item(win, "滤波")
    p = view_pos(view, item.mapToScene(QPointF(80, 12)))
    mouse(app, view.viewport(), QEvent.MouseButtonPress, p, Qt.RightButton); mouse(app, view.viewport(), QEvent.MouseButtonRelease, p, Qt.RightButton)
    menus = [m for m in view.findChildren(QMenu) if m.isVisible()]
    next(a for a in menus[0].actions() if a.text() == "复制").trigger(); menus[0].close(); pump(app)
    assert node_by_name(win, "滤波 2") is not None
    mouse(app, view.viewport(), QEvent.MouseButtonPress, p, Qt.RightButton); mouse(app, view.viewport(), QEvent.MouseButtonRelease, p, Qt.RightButton)
    menus = [m for m in view.findChildren(QMenu) if m.isVisible()]
    next(a for a in menus[0].actions() if a.text() == "禁用").trigger(); menus[0].close(); pump(app)
    assert item.node.enabled is False


def test_ui_editor_zoom_and_fit(win, app):
    view = win.view
    from PySide6.QtGui import QWheelEvent
    s0 = view.transform().m11()
    ev = QWheelEvent(QPointF(300, 300), view.mapToGlobal(QPoint(300, 300)).toPointF(), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
    app.sendEvent(view.viewport(), ev); pump(app)
    assert view.transform().m11() > s0
    QTest.keyClick(view, Qt.Key_F); pump(app)
    assert view.transform().m11() <= 1.0


# =============================================================== 参数面板
def test_ui_params_edit_spinbox_combo_checkbox_rename(win, app):
    win.scene.select_node(node_by_name(win, "孔").id); pump(app)
    panel = win.param_panel
    sb = spin(panel, "最小面积")
    sb.setValue(777); pump(app)
    assert node_by_name(win, "孔").get("min_area") == 777
    combo = param_widget(panel, "排序")
    combo.setCurrentText("left_to_right"); pump(app)
    assert node_by_name(win, "孔").get("sort") == "left_to_right"
    name = param_widget(panel, "名称")
    name.setText("孔洞"); name.editingFinished.emit(); pump(app)
    assert node_by_name(win, "孔洞") is not None
    en = next(c for c in panel.widget().findChildren(QCheckBox) if c.text() == "启用")
    en.setChecked(False); pump(app)
    assert node_by_name(win, "孔洞").enabled is False
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    assert "已禁用" in win.results_panel.tree.findItems("孔洞", Qt.MatchExactly, 0)[0].text(1)


def test_ui_params_advanced_toggle_and_json_code_apply(win, app):
    win.scene.select_node(node_by_name(win, "阈值").id); pump(app)
    panel = win.param_panel
    assert "显示高级参数" in button(panel.widget(), "显示高级参数 (2)").text()
    button(panel.widget(), "显示高级参数 (2)").click(); pump(app)
    assert param_widget(panel, "块大小") is not None
    win.scene.add_node("script.python", QPointF(0, 0)); pump(app)
    editor = param_widget(panel, "代码").findChild(QPlainTextEdit)
    editor.setPlainText("def process(ctx, inputs, params, state):\n    return {'out1': 99}\n")
    button(panel.widget(), "应用").click(); pump(app)
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    row = win.results_panel.tree.findItems("Python 脚本", Qt.MatchExactly, 0)[0]
    assert any(row.child(i).text(0) == "out1" and row.child(i).text(1) == "99" for i in range(row.childCount()))


def test_ui_params_roi_draw_show_clear(win, app):
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    win.scene.select_node(node_by_name(win, "宽度").id); pump(app)
    panel = win.param_panel
    button(panel.widget(), "绘制").click(); pump(app)
    assert win.image_view.roi_editing
    vp = win.image_view.viewport()
    drag(app, vp, QPoint(60, 60), QPoint(200, 120))
    r = node_by_name(win, "宽度").rect("roi")
    assert r is not None and r.w > 10 and r.h > 10 and not win.image_view.roi_editing
    button(panel.widget(), "显示").click(); pump(app)
    assert len(win.image_view._overlay_items) >= 1
    button(panel.widget(), "✕").click(); pump(app)
    assert node_by_name(win, "宽度").rect("roi") is None


# =============================================================== 运行控制
def test_ui_run_once_autorun_continuous(win, app):
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    assert win.results_panel.banner.text().startswith("OK") and "运行  1" in plain(win.status_stats.text())
    win.autorun.setChecked(True)
    win.scene.select_node(node_by_name(win, "孔").id); pump(app)
    spin(win.param_panel, "最小面积").setValue(100000)
    pump(app, 500); win._apply_pending_result()
    assert win.results_panel.banner.text().startswith("NG")
    win.autorun.setChecked(False)
    spin(win.param_panel, "最小面积").setValue(500)
    before = win.runner.stats.count
    win.interval.setValue(0.05); win.continuous.setChecked(True); pump(app, 400); win.continuous.setChecked(False)
    assert win.runner.stats.count >= before + 3


def test_ui_run_mode_locks_editing_and_triggers(win, modal, app):
    win.act_run_mode.trigger(); pump(app)
    assert win.run_mode and win.status_run.text() == "运行模式" and win.act_run_mode.text() == "退出运行模式"
    assert not win.palette.isEnabled() and not win.param_panel.widget().isEnabled()
    assert win.scene.add_node("preprocess.blur", QPointF(0, 0)) is None and "运行模式" in win.statusBar().currentMessage()
    before = win.runner.stats.count
    win.act_run.trigger(); pump(app, 400)
    assert win.runner.stats.count == before + 1
    assert win.comm.devices["plc"].connected
    win.act_run_mode.trigger(); pump(app)
    assert not win.run_mode and win.palette.isEnabled() and win.status_run.text() == "编辑模式"


# =============================================================== 图像窗口
def test_ui_image_shows_selected_node_overlays_and_pixel_info(win, app):
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    win.scene.select_node(node_by_name(win, "孔").id); pump(app)
    assert win.image_view.image is not None and "孔" in win.image_title.text()
    n_single = len(win.image_view._overlay_items)
    assert n_single >= 3
    win.all_overlays.setChecked(True); pump(app)
    assert len(win.image_view._overlay_items) > n_single
    win.all_overlays.setChecked(False); pump(app)
    vp = win.image_view.viewport()
    mouse(app, vp, QEvent.MouseMove, QPoint(vp.width() // 2, vp.height() // 2), Qt.NoButton, Qt.NoButton)
    assert win.pixel_label.text().startswith("x=")
    s0 = win.image_view.transform().m11()
    from PySide6.QtGui import QWheelEvent
    ev = QWheelEvent(QPointF(50, 50), vp.mapToGlobal(QPoint(50, 50)).toPointF(), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
    app.sendEvent(vp, ev); pump(app)
    assert win.image_view.transform().m11() > s0
    button(win.image_view.parent(), "适配").click(); pump(app)
    assert abs(win.image_view.transform().m11() - s0) < 0.05 * s0


# =============================================================== 结果面板
def test_ui_results_banner_tree_ok_and_ng(win, app):
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    tree = win.results_panel.tree
    names = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert names[0] == "发布的结果" and {"取图", "孔", "宽度", "孔数判定"} <= set(names)
    pub = tree.topLevelItem(0)
    assert {pub.child(i).text(0): pub.child(i).text(1) for i in range(pub.childCount())} == {"holes": "3", "width": "401"}
    node_by_name(win, "孔数判定").set("low", 99)
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    assert win.results_panel.banner.text().startswith("NG") and "a12d2d" in win.results_panel.banner.styleSheet()


# =============================================================== 变量面板
def test_ui_variables_add_edit_remove(win, modal, app):
    win.bottom_tabs.setCurrentWidget(win.variables_panel); pump(app)
    table = win.variables_panel.table
    names = {table.item(r, 0).text() for r in range(table.rowCount())}
    assert {"product", "reward"} <= names
    modal.answers["getText"] = ("line_speed", True)
    button(win.variables_panel, "添加").click(); pump(app)
    row = next(r for r in range(table.rowCount()) if table.item(r, 0).text() == "line_speed")
    table.item(row, 1).setText("12.5"); pump(app)
    assert win.solution.variables.get("line_speed") == 12.5
    table.setCurrentCell(row, 0)
    button(win.variables_panel, "删除").click(); pump(app)
    assert not win.solution.variables.has("line_speed")


# =============================================================== 日志面板
def test_ui_log_shows_messages_level_filter_clear(win, app):
    win.bottom_tabs.setCurrentWidget(win.log_panel); pump(app)
    logging.getLogger("cvflow.test").warning("黑盒日志-警告"); logging.getLogger("cvflow.test").debug("黑盒日志-调试"); pump(app)
    text = win.log_panel.text.toPlainText()
    assert "黑盒日志-警告" in text and "黑盒日志-调试" not in text
    win.log_panel.level.setCurrentText("DEBUG")
    logging.getLogger("cvflow.test").debug("黑盒日志-调试2"); pump(app)
    assert "黑盒日志-调试2" in win.log_panel.text.toPlainText()
    button(win.log_panel, "清空").click(); pump(app)
    assert win.log_panel.text.toPlainText() == ""


# =============================================================== 通信面板
def test_ui_comm_device_add_edit_test_connect_remove(win, modal, monkeypatch, app):
    from cvflow.ui import comm_panel as cp
    win.bottom_tabs.setCurrentWidget(win.comm_panel); pump(app)
    panel = win.comm_panel
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget); dev_tab = panel.dev_table.parent(); tabs.setCurrentWidget(dev_tab); pump(app)
    assert panel.dev_table.rowCount() == 1 and panel.dev_table.item(0, 1).text() == "TCP 服务端"

    def fill_device(self):
        self.name.setText("hmi"); self.kind.setCurrentIndex(self.kind.findData("tcp_server"))
        self._fields["host"].setText("127.0.0.1"); self._fields["port"].setValue(0)
        return QDialog.Accepted
    monkeypatch.setattr(cp.DeviceDialog, "exec", fill_device)
    button(dev_tab, "添加").click(); pump(app)
    assert panel.dev_table.rowCount() == 2 and "hmi" in win.comm.devices and win.windowTitle().endswith("*")
    panel.dev_table.selectRow(1); panel.dev_table.setCurrentCell(1, 0)
    button(dev_tab, "连接").click(); pump(app)
    assert win.comm.devices["hmi"].connected and "已连接" in panel.dev_table.item(1, 2).text()
    button(dev_tab, "测试连接").click(); pump(app)
    assert "监听" in modal.texts("information")[-1]
    button(dev_tab, "断开").click(); pump(app)
    assert not win.comm.devices["hmi"].connected
    button(dev_tab, "删除").click(); pump(app)
    assert panel.dev_table.rowCount() == 1 and "hmi" not in win.comm.devices


def test_ui_comm_rules_add_edit_remove_and_monitor(win, modal, monkeypatch, app):
    from cvflow.ui import comm_panel as cp
    win.bottom_tabs.setCurrentWidget(win.comm_panel); pump(app)
    panel = win.comm_panel
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)
    rx_tab = panel.rx_table.parent(); tx_tab = panel.tx_table.parent()
    assert panel.rx_table.rowCount() == 1 and panel.tx_table.rowCount() == 1
    tabs.setCurrentWidget(rx_tab); pump(app)

    def fill_rx(self):
        self.name.setText("second"); self.match.setCurrentText("equals"); self.pattern.setText("GO"); return QDialog.Accepted
    monkeypatch.setattr(cp.ReceiveRuleDialog, "exec", fill_rx)
    button(rx_tab, "添加").click(); pump(app)
    assert panel.rx_table.rowCount() == 2 and panel.rx_table.item(1, 0).text() == "second" and win.comm.receive_rules[1].pattern == "GO"
    panel.rx_table.setCurrentCell(1, 0)
    button(rx_tab, "删除").click(); pump(app)
    assert panel.rx_table.rowCount() == 1

    def fill_tx(self):
        self.name.setText("hb"); self.when.setCurrentText("ng"); self.template.setText("NG\\n"); self._regs = []; return QDialog.Accepted
    monkeypatch.setattr(cp.SendRuleDialog, "exec", fill_tx)
    tabs.setCurrentWidget(tx_tab); pump(app)
    button(tx_tab, "添加").click(); pump(app)
    assert panel.tx_table.rowCount() == 2 and win.comm.send_rules[1].when == "ng"
    panel.tx_table.setCurrentCell(1, 0); button(tx_tab, "删除").click(); pump(app)
    assert panel.tx_table.rowCount() == 1
    # 监视：连接设备后手动发送，监视列表出现记录
    import socket
    dev = win.comm.devices["plc"]; dev.config["port"] = 0; dev.connect(); pump(app)
    client = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3); pump(app, 200)
    tabs.setCurrentWidget(panel.monitor.parent()); pump(app)
    panel.send_dev.setCurrentText("plc"); panel.send_text.setText("HELLO\\n")
    button(panel.monitor.parent(), "发送").click(); pump(app, 200)
    client.settimeout(3); assert client.recv(32) == b"HELLO\n"
    client.sendall(b"PING\n"); pump(app, 300)
    items = [panel.monitor.item(i).text() for i in range(panel.monitor.count())]
    assert any("HELLO" in t and "▶" in t for t in items) and any("PING" in t and "◀" in t for t in items)
    client.close(); dev.disconnect()


# =============================================================== 相机管理
def test_ui_camera_dialog_search_add_by_ip_test_force_add(win, modal, monkeypatch, app):
    from cvflow.camera.discovery import GigEDevice
    from cvflow.ui import camera_dialog as cd
    fake = [GigEDevice(ip="192.168.1.20", mac="C4:2F:90:11:22:33", model="MV-CA050", serial="SN1", user_name="cam1", extra={"sources": ["gvcp"]})]
    monkeypatch.setattr(cd, "enumerate_cameras", lambda **kw: fake if not kw.get("targets") else [GigEDevice(ip=kw["targets"][0], mac="", model="manual", serial="", source="gvcp")])
    monkeypatch.setattr(cd.discovery, "test_connection", lambda ip, timeout=1.0: (True, f"{ip} 在线"))
    monkeypatch.setattr(cd.discovery, "force_ip", lambda mac, ip, subnet, gw: True)
    monkeypatch.setattr(cd.ForceIpDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(cd.CameraDialog, "exec", lambda self: (self.show(), pump(app), QDialog.Accepted)[-1])
    captured = {}
    orig_init = cd.CameraDialog.__init__

    def init(self, parent=None, enumerator=None):
        orig_init(self, parent, enumerator); captured["dlg"] = self
    monkeypatch.setattr(cd.CameraDialog, "__init__", init)
    win.toolbar_actions()
    next(a for a in win.toolbar_actions() if a.text() == "相机管理").trigger(); pump(app)
    dlg = captured["dlg"]
    button(dlg, "搜索相机").click(); pump(app)
    assert dlg.table.rowCount() == 1 and dlg.table.item(0, 3).text() == "SN1"
    modal.answers["getText"] = ("10.0.0.9", True)
    button(dlg, "按 IP 添加…").click(); pump(app)
    assert dlg.table.rowCount() == 2 and dlg.table.item(1, 4).text() == "10.0.0.9"
    dlg.table.selectRow(0); dlg.table.setCurrentCell(0, 0)
    button(dlg, "连接测试").click(); pump(app)
    assert "在线" in modal.texts("information")[-1]
    button(dlg, "强制 IP…").click(); pump(app)
    assert any("强制 IP" in t for t in modal.texts("information"))
    dlg.table.selectRow(0); dlg.table.setCurrentCell(0, 0)
    before = len(win.scene.node_items)
    button(dlg, "添加到流程").click(); pump(app)
    assert len(win.scene.node_items) == before + 1
    n = win.solution.flows["main"].nodes[win.selected_node]
    assert n.type_id == "source.camera" and n.get("source") == "SN1" and n.name.startswith("相机 cam1")


# =============================================================== 插件
def test_ui_plugins_load_folder_shows_in_palette(win, modal, tmp_path, app):
    plug = tmp_path / "plugins"; plug.mkdir()
    (plug / "my_node.py").write_text(
        "from cvflow.core import Node, Port, DataType\n"
        "class BBNode(Node):\n    type_id='bbtest.hello'; category='黑盒'; label='黑盒节点'\n"
        "    outputs=[Port('v', DataType.INT)]\n    def process(self, ctx, inputs):\n        return {'v': 1}\n", encoding="utf-8")
    modal.answers["getExistingDirectory"] = str(plug)
    menu_action(win, "插件(&P)", "加载插件文件夹…").trigger(); pump(app)
    tree = win.palette.tree
    labels = [tree.topLevelItem(i).child(j).text(0) for i in range(tree.topLevelItemCount()) for j in range(tree.topLevelItem(i).childCount())]
    assert "黑盒节点" in labels and str(plug) in win.solution.plugin_dirs
    menu_action(win, "插件(&P)", "重新加载插件").trigger(); pump(app)
    assert "插件" in win.statusBar().currentMessage()


# =============================================================== 帮助
def test_ui_help_about(win, modal, app):
    menu_action(win, "帮助(&H)", "关于").trigger(); pump(app)
    assert modal.calls and modal.calls[-1][0] == "about"

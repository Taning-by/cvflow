"""界面黑盒测试：通过用户可见的操作（菜单、工具栏、鼠标、键盘、对话框、表格编辑）驱动界面，
检查用户可见的结果。测试函数命名 test_ui_<功能区>_<场景>，报告脚本据此归类。
功能区：startup file flow palette editor params run image results variables log comm camera plugins help
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
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
    titles = {d.windowTitle() for d in win.findChildren(__import__("PySide6.QtWidgets", fromlist=["QDockWidget"]).QDockWidget)}
    assert {"节点库", "参数", "输出"} <= titles                      # 左 / 右 / 底 三个停靠面板
    assert win.center.count() == 2                                  # 中央：图像与流程两个主工作区
    assert win.center.widget(0) is win.image_pane and win.center.widget(1) is win.flow_pane
    assert win.image_view.isVisible() and win.view.isVisible()
    assert [win.bottom_tabs.tabText(i) for i in range(win.bottom_tabs.count())] == ["结果", "通信", "变量", "日志"]
    assert win.flow_combo.currentText() == "main" and win.flow_combo.count() == 2
    assert len(win.scene.node_items) == 11 and len(win.scene.link_items) == 13
    assert win.status_run.text() == "编辑模式"


def test_ui_startup_panels_stay_resizable_with_long_titles(win, app):
    """面板之间的分隔条必须一直拖得动（回归）。

    曾经的问题：工作区标题栏里的普通 QLabel 最小宽度等于整行文字宽度，运行一次、
    选中一个名字长的节点之后，中央区的最小宽度被撑到和实际宽度一样，分隔条就彻底拖不动了。
    """
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    win.scene.select_node(node_by_name(win, "孔").id); pump(app)
    win.image_title.setText("一个名字特别长的检测节点" * 4 + "（4096×3000，第 123456 帧） · 输入 8")
    win.flow_info.setText("节点与连线统计" * 12)
    win.pixel_label.setText("x=4095 y=2999 value=(255, 255, 255)")
    pump(app)
    assert win.centralWidget().minimumSizeHint().width() <= 480      # 中央区能缩，分隔条才有行程

    before = win.dock_palette.width()
    sep_x = win.dock_palette.geometry().right() + 3                  # 节点库与中央区之间的分隔条
    y = win.dock_palette.geometry().center().y()
    drag(app, win, QPoint(sep_x, y), QPoint(sep_x + 150, y))
    pump(app)
    assert win.dock_palette.width() >= before + 100                  # 真的被拖宽了
    assert win.image_title.text().startswith("一个名字特别长的检测节点")   # 完整文字仍可读取


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


def test_ui_editor_clicking_a_partly_visible_node_does_not_move_it(win, app):
    """放大后点选一个只露出一半的节点：视图不该滚动，节点也不该被挪走（回归）。

    曾经的问题：选中时会把节点滚进可视区，滚动改变了光标与场景的对应关系，
    接着一两像素的手抖就把刚点中的节点拖出几百个单位。
    """
    view = win.view
    target = node_by_name(win, "阈值")
    item = win.scene.node_items[target.id]
    win.scene.select_node(node_by_name(win, "取图").id)         # 先选中另一个节点
    pump(app)
    view.set_zoom(1.5)                                          # 放大到局部
    view.centerOn(item)
    pump(app)
    # 把目标推到窗口右边缘：右半边伸到视野外，左半边还能点
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() - (view.viewport().width() // 2 - 100))
    pump(app)
    visible = view.mapToScene(view.viewport().rect()).boundingRect()
    assert not visible.contains(item.sceneBoundingRect())       # 前提：节点确实只露出一部分

    before_pos = list(target.position)
    before_scroll = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
    p = view_pos(view, item.mapToScene(QPointF(30, 12)))       # 点它露在外面的标题栏
    assert view.viewport().rect().contains(p)
    drag(app, view.viewport(), p, QPoint(p.x() + 3, p.y() + 2), steps=1)   # 带 3px 手抖的点击
    pump(app)

    assert win.param_panel._node is target                     # 确实选中了它
    assert (view.horizontalScrollBar().value(), view.verticalScrollBar().value()) == before_scroll
    assert max(abs(a - b) for a, b in zip(target.position, before_pos)) < 8   # 只跟着手抖几像素


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


def test_ui_params_choosing_a_model_loads_it_right_away(win, app, tmp_path):
    """在参数面板里选好模型文件后，模型应当立刻在后台加载，而不是等到第一次运行才加载。"""
    pytest.importorskip("onnxruntime")
    from cvflow.operators.dl import active_sessions
    from nodes_helpers import make_constant_detector
    model = str(make_constant_detector(tmp_path / "det.onnx", [(320, 320, 80, 40, 0, 0.9)]))
    node = win.scene.add_node("dl.onnx_detector", QPointF(100, 620))
    win.scene.select_node(node.id)
    pump(app)
    before = active_sessions()
    win._on_param_changed(node.id, "model_path", model)      # 等同于在参数面板里选中文件
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and active_sessions() == before:
        pump(app, 50)
    assert active_sessions() == before + 1                   # 一次都没运行，权重已经就位
    node.teardown()


def test_ui_params_disabling_a_model_node_unloads_its_weights(win, app, tmp_path):
    """取消勾选"启用"之后，模型权重要从显存/内存里放掉；重新勾选再加载回来。"""
    pytest.importorskip("onnxruntime")
    from cvflow.operators.dl import active_sessions
    from nodes_helpers import make_constant_detector
    model = str(make_constant_detector(tmp_path / "det2.onnx", [(320, 320, 80, 40, 0, 0.9)]))
    node = win.scene.add_node("dl.onnx_detector", QPointF(100, 640))
    win.scene.select_node(node.id)
    pump(app)
    before = active_sessions()
    win._on_param_changed(node.id, "model_path", model)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and active_sessions() == before:
        pump(app, 50)
    assert active_sessions() == before + 1

    enabled_box = next(c for c in win.param_panel.widget().findChildren(QCheckBox) if c.text() == "启用")
    enabled_box.setChecked(False)                      # 用户在参数面板里取消启用
    pump(app)
    assert not node.enabled and active_sessions() == before          # 权重已卸载

    enabled_box.setChecked(True)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and active_sessions() == before:
        pump(app, 50)
    assert active_sessions() == before + 1                           # 重新启用后又加载回来
    node.teardown()


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
    from cvflow.ui.theme import C
    style = win.results_panel.banner.styleSheet()
    assert win.results_panel.banner.text().startswith("NG") and C["ng"] in style and C["ng_bg"] in style


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


def test_ui_comm_rules_add_edit_remove(win, modal, monkeypatch, app):
    from cvflow.ui import comm_dialogs as cd
    win.bottom_tabs.setCurrentWidget(win.comm_panel); pump(app)
    panel = win.comm_panel
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)
    rx_tab = panel.rx_table.parent(); tx_tab = panel.tx_table.parent()
    assert panel.rx_table.rowCount() == 1 and panel.tx_table.rowCount() == 1
    tabs.setCurrentWidget(rx_tab); pump(app)

    def fill_rx(self):
        self.name.setText("second")
        self.match.setCurrentIndex(self.match.findData("equals"))
        self.pattern.setText("GO")
        return QDialog.Accepted
    monkeypatch.setattr(cd.ReceiveRuleDialog, "exec", fill_rx)
    button(rx_tab, "添加").click(); pump(app)
    assert panel.rx_table.rowCount() == 2 and panel.rx_table.item(1, 0).text() == "second"
    assert win.comm.receive_rules[1].pattern == "GO" and win.comm.receive_rules[1].match == "equals"
    panel.rx_table.setCurrentCell(1, 0)
    button(rx_tab, "启用/禁用").click(); pump(app)
    assert not win.comm.receive_rules[1].enabled and "已停用" in panel.rx_table.item(1, 0).text()
    button(rx_tab, "删除").click(); pump(app)
    assert panel.rx_table.rowCount() == 1

    def fill_tx(self):
        self.name.setText("hb")
        self.when.setCurrentIndex(self.when.findData("ng"))
        self.template.setText("NG\\n")
        return QDialog.Accepted
    monkeypatch.setattr(cd.SendRuleDialog, "exec", fill_tx)
    tabs.setCurrentWidget(tx_tab); pump(app)
    button(tx_tab, "添加").click(); pump(app)
    assert panel.tx_table.rowCount() == 2 and win.comm.send_rules[1].when == "ng"
    panel.tx_table.setCurrentCell(1, 0); button(tx_tab, "删除").click(); pump(app)
    assert panel.tx_table.rowCount() == 1


def test_ui_comm_debug_manual_send_log_filter_and_export(win, modal, monkeypatch, app, tmp_path):
    """调试区：手动发文本与 HEX，日志显示方向/设备/对端/文本/十六进制，可筛选、暂停、导出。"""
    import socket
    panel = win.comm_panel
    win.bottom_tabs.setCurrentWidget(panel); pump(app)
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)
    dev = win.comm.devices["plc"]; dev.config["port"] = 0; dev.connect(); pump(app, 200)
    client = socket.create_connection(("127.0.0.1", dev.bound_port), timeout=3); pump(app, 300)
    tabs.setCurrentWidget(panel.log_table.parent()); pump(app)
    idx = panel.send_dev.findData(dev.id); panel.send_dev.setCurrentIndex(idx)
    panel.send_text.setText("HELLO\\n")
    button(panel.log_table.parent(), "发送").click(); pump(app, 300)
    client.settimeout(3); assert client.recv(32) == b"HELLO\n"
    # 十六进制发送
    panel.send_mode.setCurrentIndex(panel.send_mode.findData("hex"))
    panel.send_text.setText("02 41 42 03")
    button(panel.log_table.parent(), "发送").click(); pump(app, 300)
    assert client.recv(32) == b"\x02AB\x03"
    client.sendall(b"PING\n"); pump(app, 500)
    rows = [[panel.log_table.item(r, c).text() if panel.log_table.item(r, c) else ""
             for c in range(panel.log_table.columnCount())]
            for r in range(panel.log_table.rowCount())]
    assert any(r[1] == "发" and "HELLO" in r[5] for r in rows)
    assert any(r[1] == "收" and "PING" in r[5] and r[3] for r in rows)   # 收到的带来源对端
    assert any("41 42" in r[6] for r in rows)                            # 十六进制列
    # 按方向筛选
    panel.log_filter_dir.setCurrentIndex(panel.log_filter_dir.findData("rx")); pump(app, 300)
    dirs = {panel.log_table.item(r, 1).text() for r in range(panel.log_table.rowCount())}
    assert dirs == {"收"}
    panel.log_filter_dir.setCurrentIndex(0); pump(app, 300)
    # 停止记录
    panel.log_stop.setChecked(True); pump(app)
    before = panel.log_table.rowCount()
    client.sendall(b"QUIET\n"); pump(app, 400)
    assert panel.log_table.rowCount() == before
    panel.log_stop.setChecked(False); pump(app)
    # 导出
    out = tmp_path / "log.csv"
    modal.answers["getSaveFileName"] = (str(out), "")
    button(panel.log_table.parent(), "导出 CSV").click(); pump(app)
    assert out.exists() and "HELLO" in out.read_text(encoding="utf-8-sig")
    button(panel.log_table.parent(), "清空").click(); pump(app)
    assert panel.log_table.rowCount() == 0
    client.close(); dev.disconnect()


def test_ui_comm_data_points_add_edit_and_manual_readwrite(win, modal, monkeypatch, app):
    """数据点页：新增/编辑数据点，界面显示参考地址与字节示例；调试区手动读写。"""
    from cvflow.ui import comm_dialogs as cd
    panel = win.comm_panel
    win.bottom_tabs.setCurrentWidget(panel); pump(app)
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)

    def add_modbus(self):
        self.name.setText("mb"); self.kind.setCurrentIndex(self.kind.findData("modbus_tcp_server"))
        self._fields["host"].setText("127.0.0.1"); self._fields["port"].setValue(0)
        return QDialog.Accepted
    monkeypatch.setattr(cd.DeviceDialog, "exec", add_modbus)
    tabs.setCurrentWidget(panel.dev_table.parent()); pump(app)
    button(panel.dev_table.parent(), "添加").click(); pump(app)
    mb = win.comm.devices["mb"]

    tabs.setCurrentWidget(panel.point_table.parent()); pump(app)
    panel.point_device.setCurrentIndex(panel.point_device.findData(mb.id)); pump(app)

    def add_point(self):
        self.name.setText("x")
        self.address.setValue(20)
        self.dtype.setCurrentIndex(self.dtype.findData("float32"))
        self.layout_box.setCurrentIndex(self.layout_box.findData("CDAB"))
        self.direction.setCurrentIndex(self.direction.findData("read_write"))
        assert "40021" in self.ref.text()            # 参考地址换算显示给用户
        assert "56 78 12 34" in self.layout_hint.text()   # 字节序的实际字节示例
        return QDialog.Accepted
    monkeypatch.setattr(cd.DataPointDialog, "exec", add_point)
    button(panel.point_table.parent(), "添加").click(); pump(app)
    assert panel.point_table.rowCount() == 1
    assert panel.point_table.item(0, 0).text() == "x"
    assert "40021" in panel.point_table.item(0, 3).text()
    assert panel.point_table.item(0, 7).text() == "CDAB"

    # 重名的数据点被拒绝（校验信息是中文）
    dlg = cd.DataPointDialog(panel, None, ["x"], [])
    dlg.name.setText("x")
    assert "已经有一个数据点" in dlg.validate()
    # 只读区不能设为写
    dlg2 = cd.DataPointDialog(panel, None, [], [])
    dlg2.area.setCurrentIndex(dlg2.area.findData("input"))
    dlg2.direction.setCurrentIndex(dlg2.direction.findData("write"))
    assert "不可写" in dlg2.validate()

    # 调试区手动读写这个数据点
    mb.connect(); pump(app, 200)
    tabs.setCurrentWidget(panel.log_table.parent()); pump(app)
    panel.mb_dev.setCurrentIndex(panel.mb_dev.findData(mb.id)); pump(app)
    panel.mb_point.setCurrentText("x")
    panel.mb_value.setText("12.5")
    button(panel.log_table.parent(), "写入").click(); pump(app)
    assert "已写入" in panel.mb_result.text()
    button(panel.log_table.parent(), "读取").click(); pump(app)
    assert "12.5" in panel.mb_result.text()
    panel.mb_value.setText("abc")
    button(panel.log_table.parent(), "写入").click(); pump(app)
    assert "不是数字" in panel.mb_result.text()
    mb.disconnect()


def test_ui_comm_handshake_and_rule_library(win, modal, monkeypatch, app):
    """解析/格式化规则与检测握手：能建、能存、状态表显示实时状态。"""
    from cvflow.ui import comm_dialogs as cd
    panel = win.comm_panel
    win.bottom_tabs.setCurrentWidget(panel); pump(app)
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)
    tabs.setCurrentWidget(panel.parse_table.parent().parent()); pump(app)

    def add_parse(self):
        self.name.setText("req")
        self.kind.setCurrentIndex(self.kind.findData("delimited"))
        return QDialog.Accepted
    monkeypatch.setattr(cd.ParseRuleDialog, "exec", add_parse)
    button(panel.parse_table.parent(), "添加").click(); pump(app)
    assert panel.parse_table.rowCount() == 1 and "req" in win.comm.parse_rules

    def add_format(self):
        self.name.setText("res")
        self.kind.setCurrentIndex(self.kind.findData("text"))
        self.template.setText("{status},{request_id}\\n")
        return QDialog.Accepted
    monkeypatch.setattr(cd.FormatRuleDialog, "exec", add_format)
    button(panel.format_table.parent(), "添加").click(); pump(app)
    assert panel.format_table.rowCount() == 1 and "res" in win.comm.format_rules

    tabs.setCurrentWidget(panel.hs_table.parent()); pump(app)

    def add_hs(self):
        self.name.setText("hs")
        self.mode.setCurrentIndex(self.mode.findData("text"))
        self._formats["result_format"].setCurrentText("res")
        return QDialog.Accepted
    monkeypatch.setattr(cd.HandshakeDialog, "exec", add_hs)
    button(panel.hs_table.parent(), "添加").click(); pump(app)
    assert panel.hs_table.rowCount() == 1 and win.comm.handshakes[0].name == "hs"
    assert panel.hs_table.item(0, 4).text() in ("离线", "就绪")
    # 文本模式没选结果格式化规则时要报错
    dlg = cd.HandshakeDialog(panel, None, [(win.comm.devices["plc"].id, "plc")], ["main"], [], {}, [])
    dlg.name.setText("x"); dlg.mode.setCurrentIndex(dlg.mode.findData("text"))
    assert "结果回复" in dlg.validate()
    button(panel.hs_table.parent(), "删除").click(); pump(app)
    assert panel.hs_table.rowCount() == 0


def test_ui_comm_device_delete_warns_about_references(win, modal, monkeypatch, app):
    """删除被引用的设备时，提示里要列出**具体的引用位置**。"""
    panel = win.comm_panel
    win.bottom_tabs.setCurrentWidget(panel); pump(app)
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)
    tabs.setCurrentWidget(panel.dev_table.parent()); pump(app)
    panel.dev_table.setCurrentCell(0, 0)
    modal.answers["question"] = lambda: __import__("PySide6.QtWidgets", fromlist=["QMessageBox"]).QMessageBox.No
    button(panel.dev_table.parent(), "删除").click(); pump(app)
    asked = modal.texts("question")[-1]
    assert "接收规则「trigger」" in asked and "发送规则「result」" in asked
    assert "plc" in win.comm.devices                       # 选了“否”，没有删掉


def test_ui_comm_device_copy_and_enable_toggle(win, modal, monkeypatch, app):
    panel = win.comm_panel
    win.bottom_tabs.setCurrentWidget(panel); pump(app)
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)
    tabs.setCurrentWidget(panel.dev_table.parent()); pump(app)
    panel.dev_table.setCurrentCell(0, 0)
    button(panel.dev_table.parent(), "复制").click(); pump(app)
    assert panel.dev_table.rowCount() == 2 and "plc_副本" in win.comm.devices
    panel.dev_table.setCurrentCell(1, 0)
    button(panel.dev_table.parent(), "启用/禁用").click(); pump(app)
    assert not win.comm.devices["plc_副本"].enabled
    assert "已禁用" in panel.dev_table.item(1, 2).text()
    button(panel.dev_table.parent(), "删除").click(); pump(app)
    assert panel.dev_table.rowCount() == 1


def test_ui_comm_node_params_offer_live_choices(win, modal, app):
    """通信节点的设备/数据点参数在参数面板里是**下拉候选**，不用手敲名字。

    下拉框显示设备名、存的是稳定 id，所以改名不会让节点失去引用；选了设备之后，
    数据点的候选跟着那台设备变。
    """
    from PySide6.QtWidgets import QComboBox
    from cvflow.core import registry
    from cvflow.ui.param_options import options_for
    # 先建一台带数据点的设备
    mb = win.comm.add_device("mb", "modbus_tcp_server", {"host": "127.0.0.1", "port": 0}, points=[
        {"name": "x", "area": "holding", "address": 20, "dtype": "float32", "direction": "read_write"},
        {"name": "y", "area": "holding", "address": 22, "dtype": "float32", "direction": "read_write"}])
    node = win.graph.add_node(registry.create("comm.write_point", name="写X"))
    win.param_panel.set_node(node, win.graph); pump(app)
    combos = [c for c in win.param_panel.findChildren(QComboBox) if c.isEditable()]
    assert len(combos) >= 2
    dev_combo = combos[0]
    assert [dev_combo.itemText(i) for i in range(dev_combo.count())] == ["", "mb"]
    assert dev_combo.itemData(1) == mb.id            # 显示名字、存 id
    # 选设备
    dev_combo.setCurrentIndex(1); dev_combo.activated.emit(1); pump(app)
    assert node.values["device"] == mb.id
    # 数据点候选跟着设备走
    assert options_for("comm.points", node) == ["x", "y"]
    win.param_panel.set_node(node, win.graph); pump(app)
    combos = [c for c in win.param_panel.findChildren(QComboBox) if c.isEditable()]
    point_combo = next(c for c in combos if "x" in [c.itemText(i) for i in range(c.count())])
    i = [point_combo.itemText(k) for k in range(point_combo.count())].index("y")
    point_combo.setCurrentIndex(i); point_combo.activated.emit(i); pump(app)
    assert node.values["point"] == "y"
    # 改设备名之后，节点引用照旧（存的是 id）
    win.comm.rename_device(mb.id, "mb2")
    assert win.comm.resolve(node.values["device"]) is win.comm.devices["mb2"]
    # 候选里没有的值也能手填（先写节点、后建设备的顺序）
    node.set("point", "还没建的点")
    assert node.values["point"] == "还没建的点"


def test_ui_comm_validate_button_reports_problems(win, modal, app):
    """「检查配置」把引用了不存在的设备/流程/规则一次性列出来，正常时给一句统计。"""
    panel = win.comm_panel
    win.bottom_tabs.setCurrentWidget(panel); pump(app)
    from PySide6.QtWidgets import QTabWidget
    tabs = panel.findChild(QTabWidget)
    tabs.setCurrentWidget(panel.dev_table.parent()); pump(app)
    button(panel.dev_table.parent(), "检查配置").click(); pump(app)
    assert "没有问题" in modal.texts("information")[-1]
    win.comm.receive_rules[0].device = "ghost"
    win.comm.add_send_rule(type(win.comm.send_rules[0])(name="坏的", device="ghost", format="没有这条"))
    button(panel.dev_table.parent(), "检查配置").click(); pump(app)
    warned = modal.texts("warning")[-1]
    assert "接收规则「trigger」" in warned and "不存在的设备" in warned
    assert "格式化规则" in warned


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


# =============================================================== 参数面板重建的稳定性（回归）
def _dl_node(win, app):
    """放一个深度学习节点并选中，它的"输入个数"会改变端口数量，触发面板与图元重建。"""
    node = win.scene.add_node("dl.onnx_detector", QPointF(100, 500))
    win.scene.select_node(node.id)
    pump(app)
    return node


def test_ui_params_typing_into_port_count_does_not_crash(win, app):
    """在"输入个数"里打字后回车。曾经会把正在发信号的控件同步销毁，导致进程段错误。"""
    node = _dl_node(win, app)
    sb = spin(win.param_panel, "输入个数")
    sb.setFocus()
    sb.selectAll()
    QTest.keyClicks(sb, "4")
    QTest.keyClick(sb, Qt.Key_Return)
    pump(app, 100)
    assert node.get("input_count") == 4
    assert [p.name for p in node.inputs] == ["image", "image2", "image3", "image4"]
    assert "count4" in [p.name for p in node.outputs]
    assert win.scene.node_items[node.id].inputs.keys() == {"image", "image2", "image3", "image4"}
    assert win.param_panel._node is node                      # 面板仍然显示该节点，可继续编辑
    assert spin(win.param_panel, "输入个数").value() == 4


def test_ui_params_typing_then_focus_out_applies_and_survives(win, app):
    node = _dl_node(win, app)
    sb = spin(win.param_panel, "输入个数")
    sb.setFocus()
    sb.selectAll()
    QTest.keyClicks(sb, "3")
    sb.clearFocus()                                            # 不按回车，靠失焦提交
    pump(app, 100)
    assert node.get("input_count") == 3 and len(node.inputs) == 3
    assert spin(win.param_panel, "输入个数").value() == 3


def test_ui_params_repeated_port_edits_keep_panel_usable(win, app):
    node = _dl_node(win, app)
    for want in (4, 2, 6, 1):
        sb = spin(win.param_panel, "输入个数")
        sb.setFocus()
        sb.selectAll()
        QTest.keyClicks(sb, str(want))
        QTest.keyClick(sb, Qt.Key_Return)
        pump(app, 80)
        assert node.get("input_count") == want and len(node.inputs) == want
    assert len(win.scene.node_items[node.id].inputs) == 1


def test_ui_params_reducing_ports_drops_links(win, app):
    node = _dl_node(win, app)
    node.set("input_count", 3)
    win.scene.rebuild_node(node.id)
    pump(app)
    src = node_by_name(win, "取图")
    g = win.solution.flows["main"]
    g.add_link(src.id, "image", node.id, "image3")
    win.scene.set_graph(g)
    win.scene.select_node(node.id)
    pump(app)
    assert any(l.dst_port == "image3" for l in g.links)
    sb = spin(win.param_panel, "输入个数")
    sb.setFocus(); sb.selectAll(); QTest.keyClicks(sb, "1"); QTest.keyClick(sb, Qt.Key_Return)
    pump(app, 100)
    assert not any(l.dst_node == node.id for l in g.links)     # 指向消失端口的连线被断开
    assert "断开" in win.statusBar().currentMessage()


def test_ui_params_rename_field_does_not_crash(win, app):
    """名称输入框提交后也会重建面板，属于同一类风险。"""
    win.scene.select_node(node_by_name(win, "孔").id)
    pump(app)
    name = param_widget(win.param_panel, "名称")
    name.setFocus()
    name.selectAll()
    QTest.keyClicks(name, "HoleArea")      # QTest 送不了非拉丁字符，用 ASCII 走同一条提交路径
    QTest.keyClick(name, Qt.Key_Return)
    pump(app, 100)
    assert node_by_name(win, "HoleArea") is not None
    assert win.param_panel._node.name == "HoleArea"


def test_ui_params_roi_clear_button_does_not_crash(win, app):
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    win.scene.select_node(node_by_name(win, "宽度").id)
    pump(app)
    assert node_by_name(win, "宽度").rect("roi") is not None
    button(win.param_panel.widget(), "✕").click()              # 清除按钮会就地重建面板
    pump(app, 100)
    assert node_by_name(win, "宽度").rect("roi") is None
    assert button(win.param_panel.widget(), "绘制") is not None


def test_ui_params_advanced_toggle_does_not_crash(win, app):
    win.scene.select_node(node_by_name(win, "阈值").id)
    pump(app)
    button(win.param_panel.widget(), "显示高级参数 (2)").click()   # 按钮点击中重建自身所在面板
    pump(app, 100)
    assert param_widget(win.param_panel, "块大小") is not None
    button(win.param_panel.widget(), "隐藏高级参数 (2)").click()
    pump(app, 100)
    with pytest.raises(LookupError):
        param_widget(win.param_panel, "块大小")


# =============================================================== 结果面板的复制
def _error_row(win, app):
    """放一个未设置模型文件的 ONNX 节点并运行，结果面板的"值"一列就会出现报错信息。"""
    node = win.scene.add_node("dl.onnx_detector", QPointF(100, 500))
    src = node_by_name(win, "取图")
    win.solution.flows["main"].add_link(src.id, "image", node.id, "image")
    win.scene.set_graph(win.solution.flows["main"])
    win.act_run.trigger()
    pump(app)
    win._apply_pending_result()
    tree = win.results_panel.tree
    rows = tree.findItems(node.name, Qt.MatchExactly, 0)
    assert rows, "结果面板里没有该节点"
    return tree, rows[0]


def test_ui_results_error_text_is_visible_in_value_column(win, app):
    tree, row = _error_row(win, app)
    assert "模型文件" in row.text(1)                       # 报错信息显示在"值"一列
    assert row.toolTip(1) == row.text(1) or row.text(1).endswith("…")   # 悬停能看到完整内容


def test_ui_results_ctrl_c_copies_error_row(win, app):
    tree, row = _error_row(win, app)
    QApplication.clipboard().clear()
    tree.setCurrentItem(row)
    row.setSelected(True)
    QTest.keyClick(tree, Qt.Key_C, Qt.ControlModifier)
    pump(app)
    text = QApplication.clipboard().text()
    assert "模型文件" in text and row.text(0) in text      # 整行连同报错一起进了剪贴板


def test_ui_results_context_menu_copies_cell_row_and_all(win, app):
    tree, row = _error_row(win, app)
    rect = tree.visualItemRect(row)
    pos = QPoint(tree.columnWidth(0) + 20, rect.center().y())      # 落在"值"那一列
    tree.customContextMenuRequested.emit(pos)
    pump(app)
    menus = [m for m in tree.findChildren(QMenu) if m.isVisible()]
    assert menus, "右键没有弹出菜单"
    labels = [a.text() for a in menus[0].actions() if a.text()]
    assert "复制此格" in labels and "复制该行" in labels and "复制全部结果" in labels
    QApplication.clipboard().clear()
    next(a for a in menus[0].actions() if a.text() == "复制此格").trigger()
    pump(app)
    cell = QApplication.clipboard().text()
    assert "模型文件" in cell and row.text(0) not in cell          # 只复制了这一格
    QApplication.clipboard().clear()
    next(a for a in menus[0].actions() if a.text() == "复制全部结果").trigger()
    pump(app)
    everything = QApplication.clipboard().text()
    assert "模型文件" in everything and "取图" in everything and everything.count("\n") > 3
    for m in menus:
        m.close()


def test_ui_results_long_value_is_truncated_but_copied_in_full(win, app):
    node = win.scene.add_node("script.python", QPointF(100, 600))
    node.set("code", "def process(ctx, inputs, params, state):\n    return {'out1': 'X' * 500}\n")
    win.act_run.trigger()
    pump(app)
    win._apply_pending_result()
    tree = win.results_panel.tree
    row = tree.findItems(node.name, Qt.MatchExactly, 0)[0]
    child = next(row.child(i) for i in range(row.childCount()) if row.child(i).text(0) == "out1")
    assert len(child.text(1)) < 500 and child.text(1).endswith("…")   # 显示被截断
    assert len(child.toolTip(1)) == 500                                # 悬停是完整的
    QApplication.clipboard().clear()
    tree.setCurrentItem(child)
    child.setSelected(True)
    QTest.keyClick(tree, Qt.Key_C, Qt.ControlModifier)
    pump(app)
    assert QApplication.clipboard().text().count("X") == 500            # 复制到的是完整值


def test_ui_results_banner_text_is_selectable(win, app):
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    assert win.results_panel.banner.textInteractionFlags() & Qt.TextSelectableByMouse


# =============================================================== 多输入节点的图像查看
def test_ui_image_input_picker_hidden_for_single_input_node(win, app):
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    win.scene.select_node(node_by_name(win, "孔").id)
    pump(app)
    assert not win.input_pick.isVisible()          # 单输入节点不显示选择框


def test_ui_image_input_picker_switches_between_two_inputs(win, app):
    """位运算节点有 a、b 两路图像输入，选择框应能切换查看哪一路。"""
    g = win.solution.flows["main"]
    node = win.scene.add_node("preprocess.bitwise", QPointF(100, 520))
    g.add_link(node_by_name(win, "灰度").id, "image", node.id, "a")
    g.add_link(node_by_name(win, "阈值").id, "image", node.id, "b")
    win.scene.set_graph(g)
    win.scene.select_node(node.id)
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    assert win.input_pick.isVisible() and win.input_pick.count() == 3
    assert [win.input_pick.itemText(i) for i in range(3)] == ["输入 1", "输入 2", "输出"]
    assert "输出" in win.image_title.text()                               # 该节点有图像输出，默认看输出
    win.input_pick.setCurrentIndex(0)
    pump(app)
    first = win.image_view.image.data.copy()
    assert "输入 1" in win.image_title.text()
    win.input_pick.setCurrentIndex(1)
    pump(app)
    second = win.image_view.image.data
    assert second.shape == first.shape and not (second == first).all()   # 确实换了另一路的图
    assert "输入 2" in win.image_title.text()
    assert set(np.unique(second)) <= {0, 255}                            # 第二路是阈值后的二值图


def test_ui_image_multi_input_dl_node_shows_each_input_and_its_overlays(win, app, tmp_path):
    """两路输入的 ONNX 检测节点：每一路都能看到自己的图和自己的检测框。"""
    from nodes_helpers import make_constant_detector
    model = str(make_constant_detector(tmp_path / "det.onnx", [(320, 320, 80, 40, 0, 0.9)]))
    g = win.solution.flows["main"]
    node = win.scene.add_node("dl.onnx_detector", QPointF(100, 620))
    node.set("input_count", 2)
    node.set("model_path", model)
    win.scene.rebuild_node(node.id)
    g.add_link(node_by_name(win, "取图").id, "image", node.id, "image")
    g.add_link(node_by_name(win, "渲染").id, "image", node.id, "image2")
    win.scene.set_graph(g)
    win.scene.select_node(node.id)
    win.act_run.trigger(); pump(app, 200); win._apply_pending_result()
    nr = win.runner.last_result.node_results[node.id]
    assert nr.status.value == "ok", nr.error
    assert nr.outputs["count"] == 1 and nr.outputs["count2"] == 1        # 两路各自有结果
    assert win.input_pick.isVisible() and win.input_pick.count() == 2
    boxes_1 = len([o for o in win.image_view._overlay_items])
    assert boxes_1 > 0                                                   # 第一路看到检测框
    win.input_pick.setCurrentIndex(1)
    pump(app)
    assert "输入 2" in win.image_title.text()
    assert len(win.image_view._overlay_items) > 0                        # 第二路也看到自己的检测框
    groups = {o.group for o in nr.overlays if o.kind == "rect"}
    assert groups == {"image", "image2"}                                 # 叠加层按来源分开标记


def test_ui_image_all_overlays_ignores_input_filter(win, app):
    g = win.solution.flows["main"]
    node = win.scene.add_node("preprocess.bitwise", QPointF(100, 520))
    g.add_link(node_by_name(win, "灰度").id, "image", node.id, "a")
    g.add_link(node_by_name(win, "阈值").id, "image", node.id, "b")
    win.scene.set_graph(g)
    win.scene.select_node(node.id)
    win.act_run.trigger(); pump(app); win._apply_pending_result()
    win.all_overlays.setChecked(True)
    pump(app)
    assert len(win.image_view._overlay_items) > 0                        # 勾选后显示全流程的叠加层
    win.all_overlays.setChecked(False)


def test_ui_source_nodes_have_a_trigger_icon_on_the_node(win, app):
    """相机 / 图像文件 / 图像文件夹在画布上的卡片里带一个触发图标，点它只触发这一路。

    产线上这一步由 PLC 报文触发，这个图标是现场手工验证用的——所以它必须真的接到
    FlowRunner.trigger_source 上，而且不能顺手把节点拖走。
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneMouseEvent
    from cvflow.core.registry import registry

    calls: list = []
    win.runner.trigger_source = lambda nid, *a, **k: calls.append(nid)

    folder = next(n for n in win.graph.nodes.values() if n.type_id == "source.image_folder")
    item = win.scene.node_items[folder.id]
    rect = item.trigger_rect()
    assert rect is not None                                   # 可触发的节点才有这个图标
    before = item.pos()

    def send(kind, pos):
        ev = QGraphicsSceneMouseEvent(kind)
        ev.setPos(pos)
        ev.setScenePos(item.mapToScene(pos))
        ev.setButton(Qt.LeftButton)
        ev.setButtons(Qt.LeftButton)
        ev.setButtonDownScenePos(Qt.LeftButton, item.mapToScene(pos))
        (item.mousePressEvent if kind == QEvent.GraphicsSceneMousePress else item.mouseReleaseEvent)(ev)

    send(QEvent.GraphicsSceneMousePress, rect.center())
    send(QEvent.GraphicsSceneMouseRelease, rect.center())      # 抬手也要拦，否则 Qt 会选中节点
    pump(app)
    assert calls == [folder.id]                               # 点图标＝触发这一路
    assert item.pos() == before                               # 没有被当成拖拽
    assert item.isSelected() is False                         # 也不该选中它（右边参数面板不会乱跳）

    # 点在标题别处：不触发，但照常选中节点
    send(QEvent.GraphicsSceneMousePress, rect.center() + QPointF(-60, 0))
    send(QEvent.GraphicsSceneMouseRelease, rect.center() + QPointF(-60, 0))
    pump(app)
    assert calls == [folder.id]
    assert item.isSelected() is True

    # 另外两个源节点也可触发；别的节点没有这个图标
    for type_id in ("source.image_file", "source.camera"):
        assert registry.get(type_id).triggerable is True, type_id
    thresh = next(n for n in win.graph.nodes.values() if n.type_id == "preprocess.threshold")
    assert win.scene.node_items[thresh.id].trigger_rect() is None


def test_ui_trigger_icon_shows_running_state(win, app):
    """图标有两个状态：空闲画"暂停"两道竖杠，正在触发画实心三角。"""
    folder = next(n for n in win.graph.nodes.values() if n.type_id == "source.image_folder")
    item = win.scene.node_items[folder.id]
    assert item.is_triggering() is False                      # 没触发时是空闲
    win.scene.branch_busy = lambda nid: nid == folder.id       # 装成这一路正在跑
    assert item.is_triggering() is True
    win.scene.branch_busy = lambda nid: (_ for _ in ()).throw(RuntimeError("探测失败"))
    assert item.is_triggering() is False                      # 探测出错也不能让界面崩

def test_ui_dl_node_shows_arrival_and_weight_group(win, app, tmp_path):
    """深度学习节点的参数面板上有「输入到达方式」，高级参数里有「权重共享组」、没有「超时（秒）」。"""
    from PySide6.QtCore import QPointF
    node = win.scene.add_node("dl.onnx_classifier", QPointF(50, 50))
    pump(app)
    win.scene.select_node(node.id); pump(app)
    panel = win.param_panel
    assert param_widget(panel, "输入到达方式").currentText() == "sync"
    button(panel.widget(), next(b.text() for b in panel.widget().findChildren(QPushButton)
                                if b.text().startswith("显示高级参数"))).click()
    pump(app)
    assert param_widget(panel, "权重共享组") is not None
    with pytest.raises(LookupError):
        param_widget(panel, "超时（秒）")

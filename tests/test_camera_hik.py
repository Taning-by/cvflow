"""海康 MVS 后端的 SDK 装载路径测试（找模块、挂运行库、报错怎么说）。

手上没有这台相机，但"装没装上"这条路最容易在现场出事，而且和硬件无关：
MvImport 是普通的 .py，可以造一个替身顶上去，连「模块找到了但运行库加载不了」
（打包版报 PyInstallerImportError 的那种）都能在这里复现。
"""
from __future__ import annotations

import sys

import pytest

from cvflow.camera import hik_cam
from cvflow.camera.base import CameraError


@pytest.fixture()
def clean_sdk_state(monkeypatch):
    """每个用例都从"还没装载过"开始，并且不把 sys.path / sys.modules 留给下一个用例。"""
    monkeypatch.setattr(hik_cam, "_mv", None)
    monkeypatch.setattr(hik_cam, "_runtime_tried", set())
    monkeypatch.setattr(hik_cam, "_SDK_ROOTS", [])
    monkeypatch.setattr(hik_cam, "_RUNTIME_ROOTS", [])
    path, mods = list(sys.path), dict(sys.modules)
    yield
    sys.path[:] = path
    for name in set(sys.modules) - set(mods):
        del sys.modules[name]


def _fake_mvimport(tmp_path, body: str):
    """造一个顶替 MvCameraControl_class 的目录，返回这个目录。"""
    d = tmp_path / "MvImport"
    d.mkdir()
    (d / "MvCameraControl_class.py").write_text(body, encoding="utf-8")
    return str(d)


def test_runtime_dir_found_under_the_given_root(tmp_path, clean_sdk_state):
    """运行库目录按架构子目录找，sdk_path 填到 Runtime 这一级也认。"""
    sub = hik_cam._runtime_subdirs()[0]
    d = tmp_path / "Runtime" / sub
    d.mkdir(parents=True)
    (d / hik_cam._LIB_NAME).write_bytes(b"")
    assert hik_cam._runtime_dirs(str(tmp_path / "Runtime")) == [str(d)]
    assert hik_cam._runtime_dirs(str(tmp_path / "没这个目录")) == []


def test_runtime_search_path_is_opened_before_the_module_is_imported(tmp_path, monkeypatch, clean_sdk_state):
    """顺序不能反：模块一导入就会去加载运行库，搜索路径必须在那之前挂好。"""
    log = tmp_path / "order.txt"
    monkeypatch.setenv("CVFLOW_TEST_ORDER", str(log))
    monkeypatch.setattr(hik_cam, "_open_runtime_search_path",
                        lambda extra="": log.open("a").write("runtime\n"))
    hik_cam._SDK_ROOTS.append(_fake_mvimport(
        tmp_path, "import os\nopen(os.environ['CVFLOW_TEST_ORDER'], 'a').write('import\\n')\n"))
    hik_cam.load_sdk()
    assert log.read_text().split() == ["runtime", "import"]


def test_dll_load_failure_says_it_is_the_runtime_not_a_missing_mvs(tmp_path, clean_sdk_state):
    """模块找到了、运行库没加载上（打包版的 PyInstallerImportError 就是这种）：

    报错要指向运行库目录和 MVCAM_COMMON_RUNENV，而不是再劝一遍「请安装 MVS」；
    而且这种 OSError 不能漏出去——sdk_available() 还得照常回 False。
    """
    rt = tmp_path / "Runtime" / hik_cam._runtime_subdirs()[0]
    rt.mkdir(parents=True)
    (rt / hik_cam._LIB_NAME).write_bytes(b"")
    hik_cam._RUNTIME_ROOTS.append(str(tmp_path / "Runtime"))
    hik_cam._SDK_ROOTS.append(_fake_mvimport(
        tmp_path, "raise OSError(\"Failed to load dynlib/dll 'MvCameraControl.dll'\")\n"))

    with pytest.raises(CameraError) as e:
        hik_cam.load_sdk()
    msg = str(e.value)
    assert hik_cam._LIB_NAME in msg and "MVCAM_COMMON_RUNENV" in msg and str(rt) in msg
    assert "请先安装 MVS" not in msg
    assert hik_cam.sdk_available() is False


def test_missing_module_explains_how_to_install(clean_sdk_state):
    """真的没装 MVS 时，报错要说清去哪儿找 MvImport。"""
    hik_cam._SDK_ROOTS.append("/不存在的目录")
    with pytest.raises(CameraError) as e:
        hik_cam.load_sdk()
    msg = str(e.value)
    assert "MvCameraControl_class" in msg and "MVCAM_SDK_PATH" in msg and "sdk_path" in msg


def test_sdk_module_is_cached_after_a_successful_load(tmp_path, clean_sdk_state):
    hik_cam._SDK_ROOTS.append(_fake_mvimport(tmp_path, "MV_GIGE_DEVICE = 1\n"))
    mv = hik_cam.load_sdk()
    assert mv.MV_GIGE_DEVICE == 1
    assert hik_cam.load_sdk() is mv
    assert hik_cam.sdk_available() is True

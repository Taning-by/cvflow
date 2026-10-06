"""安装环境用的工具脚本（tools/venv_clean.py）的黑盒测试。

这两个动作对应装 GPU 版时最隐蔽的两个坑，都是只看 pip 输出发现不了的，所以要有回归：
* 系统 site-packages 里的 CPU 版 onnxruntime 盖掉环境里的 GPU 版；
* 混装过的环境里"dist-info 还在、模块目录没了"，pip 以为装过直接跳过安装。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "venv_clean.py"


@pytest.fixture
def venv_clean():
    spec = importlib.util.spec_from_file_location("venv_clean", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_venv(tmp_path, monkeypatch, venv_clean):
    """把 tmp_path 伪装成一个虚拟环境：有 pyvenv.cfg，sys.prefix 指向它。"""
    site = tmp_path / "lib" / "site-packages"
    site.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setattr(venv_clean.sysconfig, "get_paths",
                        lambda *a, **k: {"purelib": str(site), "platlib": str(site)})
    return tmp_path, site


def test_isolate_turns_off_system_site_packages_and_keeps_a_backup(fake_venv, venv_clean, capsys):
    tmp_path, _ = fake_venv
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_text("home = /usr/bin\ninclude-system-site-packages = true\nversion = 3.10.12\n")

    assert venv_clean.main(["isolate"]) == 0
    assert "include-system-site-packages = false" in cfg.read_text()
    assert "home = /usr/bin" in cfg.read_text()                     # 其它行一个字没动
    assert "include-system-site-packages = true" in (tmp_path / "pyvenv.cfg.bak").read_text()
    assert "已关掉" in capsys.readouterr().out


def test_isolate_is_idempotent(fake_venv, venv_clean, capsys):
    tmp_path, _ = fake_venv
    cfg = tmp_path / "pyvenv.cfg"
    cfg.write_text("include-system-site-packages = false\n")

    assert venv_clean.main(["isolate"]) == 0
    assert cfg.read_text() == "include-system-site-packages = false\n"
    assert not (tmp_path / "pyvenv.cfg.bak").exists()               # 没改就不留备份
    assert "本来就是隔离的" in capsys.readouterr().out


def test_purge_removes_module_dir_and_stale_dist_info(fake_venv, venv_clean, capsys):
    """半残状态（dist-info 在、模块目录没了）必须被清掉，否则 pip 会跳过安装。"""
    _, site = fake_venv
    (site / "onnxruntime" / "capi").mkdir(parents=True)
    (site / "onnxruntime" / "__init__.py").write_text("")
    (site / "onnxruntime_gpu-1.23.2.dist-info").mkdir()
    (site / "onnxruntime-1.23.2.dist-info").mkdir()
    (site / "numpy").mkdir()                                        # 不相关的包不许动

    assert venv_clean.main(["purge"]) == 0
    assert not (site / "onnxruntime").exists()
    assert not (site / "onnxruntime_gpu-1.23.2.dist-info").exists()
    assert not (site / "onnxruntime-1.23.2.dist-info").exists()
    assert (site / "numpy").exists()
    assert "删掉残留" in capsys.readouterr().out


def test_purge_on_a_clean_environment_says_so(fake_venv, venv_clean, capsys):
    assert venv_clean.main(["purge"]) == 0
    assert "没有 onnxruntime 残留" in capsys.readouterr().out


def test_refuses_to_touch_files_outside_a_venv(tmp_path, monkeypatch, venv_clean, capsys):
    """在系统 python 下执行要直接拒绝，免得误删系统 site-packages。"""
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))           # prefix == base_prefix：不在虚拟环境里
    (tmp_path / "pyvenv.cfg").write_text("include-system-site-packages = true\n")

    assert venv_clean.main(["isolate"]) == 1
    assert venv_clean.main(["purge"]) == 1
    assert "include-system-site-packages = true" in (tmp_path / "pyvenv.cfg").read_text()
    assert "必须用虚拟环境里的 python" in capsys.readouterr().err


def test_unknown_action_prints_usage(venv_clean, capsys):
    assert venv_clean.main([]) == 1
    assert venv_clean.main(["isolate", "purge"]) == 1
    assert venv_clean.main(["rm -rf"]) == 1
    assert "venv_clean.py" in capsys.readouterr().out

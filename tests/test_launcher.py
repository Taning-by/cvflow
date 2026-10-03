import platform
from pathlib import Path

import pytest

from cvflow import launcher


def test_gui_executable_found():
    exe = launcher.gui_executable()
    assert exe.exists() and exe.name.startswith("cvflow-gui")


@pytest.mark.skipif(platform.system() != "Linux", reason="freedesktop entries only")
def test_create_desktop_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None if name == "xdg-user-dir" else launcher.shutil.which(name))
    entry = launcher.create_shortcut("examples/solutions/demo_holes.json", name="CVFlowTest")
    text = entry.read_text(encoding="utf-8")
    assert entry.name == "CVFlowTest.desktop" and "cvflow-gui" in text and "demo_holes.json" in text
    assert "Icon=" in text and Path(text.split("Icon=")[1].splitlines()[0]).exists()
    assert (tmp_path / ".local/share/applications/CVFlowTest.desktop").exists()

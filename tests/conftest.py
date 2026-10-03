import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cvflow.core import Image, registry  # noqa: E402

registry.load_builtins()


@pytest.fixture(scope="session")
def reg():
    return registry


@pytest.fixture(scope="session")
def images_dir():
    d = ROOT / "examples" / "images"
    if not any(d.glob("part_*.png")):
        import subprocess
        subprocess.check_call([sys.executable, str(ROOT / "examples" / "make_demo_images.py"), str(d)])
    return d


@pytest.fixture(scope="session")
def plugins_dir():
    return ROOT / "examples" / "plugins"


def make_blob_image(n_blobs: int = 3, size: int = 200) -> Image:
    """Dark background, n white squares (40x40) in a row."""
    data = np.zeros((size, size), dtype=np.uint8)
    for i in range(n_blobs):
        x = 20 + i * 60
        cv2.rectangle(data, (x, 80), (x + 40, 120), 255, -1)
    return Image(cv2.cvtColor(data, cv2.COLOR_GRAY2BGR))


@pytest.fixture
def blob_image():
    return make_blob_image()

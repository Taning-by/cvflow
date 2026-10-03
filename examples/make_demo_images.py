"""Generate synthetic 'part with three holes' images for the demo solution.

Images 0-6 are good (3 holes), 7 has a missing hole, 8 has an extra dark blemish,
9 is shifted and slightly darker. Edge caliper on the part width is ~400 px.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "images")


def make(idx: int, rng: np.random.Generator) -> np.ndarray:
    img = np.full((480, 640), 40, dtype=np.uint8)
    dx, dy = (int(rng.integers(-6, 7)), int(rng.integers(-6, 7))) if idx != 9 else (40, 20)
    part_val = 200 if idx != 9 else 170
    x0, y0, x1, y1 = 120 + dx, 100 + dy, 520 + dx, 380 + dy
    cv2.rectangle(img, (x0, y0), (x1, y1), part_val, -1)
    holes = [220, 320, 420]
    if idx == 7:
        holes = [220, 420]
    for hx in holes:
        cv2.circle(img, (hx + dx, 240 + dy), 25, 40, -1)
    if idx == 8:
        cv2.ellipse(img, (380 + dx, 320 + dy), (30, 12), 25, 0, 360, 45, -1)
    noise = rng.normal(0, 4, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(10):
        cv2.imwrite(str(OUT / f"part_{i:02d}.png"), make(i, rng))
    # a template of one hole for the template-matching demo
    tpl = make(0, np.random.default_rng(1))[215:265, 295:345]
    tdir = OUT.parent / "templates"
    tdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tdir / "hole_template.png"), tpl)
    print(f"wrote 10 images to {OUT} and hole_template.png to {tdir}")


if __name__ == "__main__":
    main()

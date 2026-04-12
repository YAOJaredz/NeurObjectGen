"""Crop Rust stimulus images to the circular aperture bounding box.

Raw stimuli are small circular patches embedded in a large black canvas.
We compute a single bbox from the first image (the aperture is screen-fixed)
and apply it to every frame, writing results to stimuli/rust_cropped/
with simple index-based filenames.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
import scipy

from config_const import STIMULI_ROOT, RUST_SRC_NAME, RUST_DST_NAME, RUST_BG_THRESHOLD

INDEX_RE = re.compile(r"index(\d+)\.png$")


def compute_bbox(img: Image.Image, pad: int = 2) -> tuple[int, int, int, int]:
    """Return bbox of the largest non-black connected component.

    Ignores small artifacts like the photodiode sync patch in the corner.
    """
    arr = np.asarray(img.convert("RGB"))
    mask = (arr > RUST_BG_THRESHOLD).any(axis=-1)
    labels, n = scipy.ndimage.label(mask) 
    if n == 0:
        raise ValueError("no foreground pixels found")
    h, w = mask.shape # type: ignore
    edge_labels: set[int] = set(labels[0].tolist()) | set(labels[-1].tolist())
    edge_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    for lbl in edge_labels:
        sizes[lbl] = 0  # drop photodiode / edge artifacts
    biggest = int(sizes.argmax())
    ys, xs = np.where(labels == biggest)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (
        max(x0 - pad, 0),
        max(y0 - pad, 0),
        min(x1 + pad, w),
        min(y1 + pad, h),
    )


def main() -> None:
    src = STIMULI_ROOT / RUST_SRC_NAME
    dst = STIMULI_ROOT / RUST_DST_NAME
    dst.mkdir(parents=True, exist_ok=True)

    def _index(p: Path) -> int:
        m = INDEX_RE.search(p.name)
        if m is None:
            raise ValueError(f"unexpected filename: {p.name}")
        return int(m.group(1))

    pngs = sorted(src.glob("*.png"), key=_index)
    if not pngs:
        raise SystemExit(f"no PNGs found in {src}")

    with Image.open(pngs[0]) as probe:
        bbox = compute_bbox(probe)
    print(f"bbox from {pngs[0].name}: {bbox} (w={bbox[2]-bbox[0]}, h={bbox[3]-bbox[1]})")

    for p in pngs[:-5]:
        idx = _index(p)
        with Image.open(p) as im:
            im.crop(bbox).save(dst / f"{idx:04d}.png")

    print(f"wrote {len(pngs)} images to {dst}")


if __name__ == "__main__":
    main()

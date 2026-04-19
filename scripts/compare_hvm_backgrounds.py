"""Find the most similar background pairs between two HVM categories.

Samples random pixels from each image and ranks all 45x45 pairs by MSE.

Usage:
  python scripts/compare_hvm_backgrounds.py --cat1 apple --cat2 bear
  python scripts/compare_hvm_backgrounds.py --cat1 apple --cat2 bear --top 10
"""
import argparse

import numpy as np
from PIL import Image

from config_const import HVM_STIM_DIR, HVM_CATEGORIES

_N_PTS = 500


def load_samples(cat: str, pts: np.ndarray) -> np.ndarray:
    """Return (45, N_PTS*3) array of sampled pixel values for all images in a category."""
    out = []
    for k in range(45):
        arr = np.array(Image.open(HVM_STIM_DIR / cat / f"{k:02d}.png").convert("RGB"))
        out.append(arr[pts[:, 0], pts[:, 1]].ravel().astype(np.float32))
    return np.stack(out)  # (45, N_PTS*3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat1", required=True, choices=HVM_CATEGORIES)
    parser.add_argument("--cat2", required=True, choices=HVM_CATEGORIES)
    parser.add_argument("--top", type=int, default=5, help="Number of most similar pairs to show.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # sample random pixel coordinates (shared across all images)
    h, w = np.array(Image.open(HVM_STIM_DIR / args.cat1 / "00.png")).shape[:2]
    pts = np.stack([rng.integers(0, h, _N_PTS), rng.integers(0, w, _N_PTS)], axis=1)

    s1 = load_samples(args.cat1, pts)  # (45, D)
    s2 = load_samples(args.cat2, pts)  # (45, D)

    # MSE for all 45x45 pairs
    mse = ((s1[:, None, :] - s2[None, :, :]) ** 2).mean(axis=-1)  # (45, 45)

    # find top-k most similar
    flat = mse.ravel()
    top_idx = np.argpartition(flat, args.top)[:args.top]
    top_idx = top_idx[np.argsort(flat[top_idx])]

    print(f"Top {args.top} most similar pairs ({args.cat1} vs {args.cat2}):")
    for idx in top_idx:
        k1, k2 = divmod(int(idx), 45)
        print(f"  {args.cat1}[{k1:02d}] ~ {args.cat2}[{k2:02d}]  MSE={mse[k1, k2]:.1f}")

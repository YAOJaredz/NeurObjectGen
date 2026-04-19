"""Copy and rename HVM stimulus images, saving raw and cropped versions separately.

Raw images are copied as-is; cropped versions use a single bbox computed from
the first image (the aperture is screen-fixed).

Source layout:
  E8/hvm10_{cat}_45_20240906/canvasvisible_..._index{k}.png

Destination layout:
  stimuli/hvm/{cat}/{k:02d}.png          (raw)
  stimuli/hvm_cropped/{cat}/{k:02d}.png  (cropped)

Index k (0–44) matches the within-category position in the neural response
arrays returned by get_hvm_responses — stim_id = k*2 in the h5 files.
"""
import re
import shutil
from pathlib import Path

from PIL import Image

from config_const import HVM_SRC_DIR, HVM_RAW_DIR, HVM_STIM_DIR, HVM_CATEGORIES
from data_utils.stim_preprocess import compute_bbox

_INDEX_RE = re.compile(r'_index(\d+)\.png$')


def copy_hvm_stimuli(overwrite: bool = False) -> None:
    bbox = None  # computed once from the first image encountered

    for cat in HVM_CATEGORIES:
        src_dirs = sorted(HVM_SRC_DIR.glob(f'hvm10_{cat}_45_*'))
        if not src_dirs:
            raise FileNotFoundError(f"No source folder for '{cat}' in {HVM_SRC_DIR}")
        src_dir = src_dirs[0]

        raw_dir    = HVM_RAW_DIR  / cat
        cropped_dir = HVM_STIM_DIR / cat
        raw_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)

        pngs = sorted(
            [p for p in src_dir.glob('*.png') if _INDEX_RE.search(p.name)],
            key=lambda p: int(_INDEX_RE.search(p.name).group(1)),
        )
        if len(pngs) != 45:
            raise RuntimeError(f"Expected 45 PNGs for {cat}, found {len(pngs)}")

        if bbox is None:
            with Image.open(pngs[0]) as probe:
                bbox = compute_bbox(probe)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            print(f"bbox: {bbox}  ({w}×{h} px)")

        for src in pngs:
            k = int(_INDEX_RE.search(src.name).group(1))
            raw_dst     = raw_dir    / f"{k:02d}.png"
            cropped_dst = cropped_dir / f"{k:02d}.png"

            if not raw_dst.exists() or overwrite:
                shutil.copy2(src, raw_dst)

            if not cropped_dst.exists() or overwrite:
                with Image.open(src) as im:
                    im.crop(bbox).save(cropped_dst)

        print(f"  {cat}: 45 raw → {raw_dir}")
        print(f"  {cat}: 45 cropped → {cropped_dir}")

    print(f"Done.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    copy_hvm_stimuli(overwrite=args.overwrite)

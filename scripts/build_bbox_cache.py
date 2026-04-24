"""Build per-stimulus bounding box cache for all 450 HVM stimuli.

Runs Grounding DINO on each stimulus at native 276×276 resolution, converts the
detected rectangle to a square, and falls back to the scale-based projection for
stimuli where detection fails.

Output: cache/hvm10_bboxes.json — list of 450 dicts, same order as hvm10_metadata.json.
  {idx, cx, cy, half, source, score}
  - cx, cy: square center in pixels (276×276 space)
  - half:   half-side-length in pixels
  - source: "gdino" | "projected"
  - score:  GDINO confidence (0.0 for projected fallback)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.append('.')
from config_const import CACHE_DIR, HVM_CATEGORIES, HVM_N_VAR, HVM_NOFIXATION_DIR
from generation.project_hvm import hvm_object_pixel_coords, load_hvm_metadata

IMAGE_SIZE    = 276
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
BASE_FRAC     = 0.5   # fallback bbox_frac at s=1

GDINO_MODEL = "IDEA-Research/grounding-dino-base"
OUT_PATH    = CACHE_DIR / "hvm10_bboxes.json"


def _to_square(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float]:
    """Convert a rectangle to a square (max side), return (cx, cy, half)."""
    w, h = x1 - x0, y1 - y0
    side = max(w, h)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return cx, cy, side / 2


def _fallback(m: dict) -> tuple[float, float, float, str, float]:
    cx, cy, _ = hvm_object_pixel_coords(m["ty"], m["tz"], m["s"], image_size=IMAGE_SIZE)
    half = IMAGE_SIZE * BASE_FRAC * m["s"] / 2
    return cx, cy, half, "projected", 0.0


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Grounding DINO ({GDINO_MODEL}) on {device}…")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device).eval()

    metadata = load_hvm_metadata()
    results: list[dict] = []

    n_gdino = 0
    n_fallback = 0

    for cat_i, cat in enumerate(HVM_CATEGORIES):
        cat_hits = 0
        for var in range(HVM_N_VAR):
            global_idx = cat_i * HVM_N_VAR + var
            m = metadata[global_idx]

            img_path = Path(HVM_NOFIXATION_DIR) / cat / f"{var:02d}.png"
            pil_img = Image.open(img_path).convert("RGB")

            text = cat + "."
            inputs = processor(images=pil_img, text=text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            det = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                target_sizes=[pil_img.size[::-1]],
            )[0]

            if len(det["boxes"]) > 0:
                best = det["scores"].argmax().item()
                box = det["boxes"][best].cpu().tolist()
                score = float(det["scores"][best])
                cx, cy, half = _to_square(*box)
                source = "gdino"
                n_gdino += 1
                cat_hits += 1
            else:
                cx, cy, half, source, score = _fallback(m)
                n_fallback += 1

            results.append({
                "idx":    global_idx,
                "cx":     round(cx, 2),
                "cy":     round(cy, 2),
                "half":   round(half, 2),
                "source": source,
                "score":  round(score, 4),
            })

        print(f"  {cat:10s}: {cat_hits}/{HVM_N_VAR} detected")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    total = len(results)
    print(f"\nSaved {total} entries → {OUT_PATH}")
    print(f"  gdino:     {n_gdino}/{total} ({100*n_gdino/total:.1f}%)")
    print(f"  projected: {n_fallback}/{total} ({100*n_fallback/total:.1f}%)")


if __name__ == "__main__":
    main()

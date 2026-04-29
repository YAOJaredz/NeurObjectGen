"""Embed bbox-cropped HVM stimuli with frozen SigLIP.

For each of the 450 HVM stimuli, crops the image to the GDINO bounding box
(from hvm10_bboxes.json, in 276×276 pixel space), resizes the crop to 384×384,
and embeds it with frozen SigLIP-SO400M.

Output: cache/hvm_obj_siglip_embeddings.pt — (450, 1152) L2-normalised.

Usage:
    python scripts/build_obj_siglip_cache.py
    python scripts/build_obj_siglip_cache.py --force
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append('.')

from config_const import (
    HVM_STIM_DIR, HVM_N_STIMULI, HVM_OBJ_SIGLIP_EMBEDDINGS_PATH,
)
from encoders.siglip_embed import load_siglip
from generation.project_hvm import load_hvm_bboxes

SRC_SIZE    = 276   # bbox coordinates are in this space
CROP_SIZE   = 384   # SigLIP input resolution
BATCH_SIZE  = 32
PAD_PX      = 5     # extra padding around bbox


def crop_to_bbox(pil_img: Image.Image, bbox: dict, src_size: int = SRC_SIZE,
                 pad: int = PAD_PX) -> Image.Image:
    """Crop PIL image to the square bbox region, with padding, then resize to CROP_SIZE."""
    w, h = pil_img.size
    scale = w / src_size  # pil_img may not be 276px
    cx  = bbox['cx'] * scale
    cy  = bbox['cy'] * scale
    half = bbox['half'] * scale + pad * scale

    x0 = max(0, int(cx - half))
    y0 = max(0, int(cy - half))
    x1 = min(w, int(cx + half))
    y1 = min(h, int(cy + half))

    crop = pil_img.crop((x0, y0, x1, y1))
    return crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    out_path = HVM_OBJ_SIGLIP_EMBEDDINGS_PATH
    if out_path.exists() and not args.force:
        print(f"{out_path} exists — skipping (use --force to recompute)")
        return

    bboxes = load_hvm_bboxes()

    all_stim_paths = sorted(
        p for cat in sorted(HVM_STIM_DIR.iterdir()) if cat.is_dir()
        for p in sorted(cat.iterdir())
    )
    assert len(all_stim_paths) == HVM_N_STIMULI, f"Expected {HVM_N_STIMULI} stimuli, got {len(all_stim_paths)}"

    print(f"Loading SigLIP on {args.device}...")
    processor, vision_model = load_siglip(args.device)

    crops = []
    for i in tqdm(range(HVM_N_STIMULI), desc='Cropping'):
        pil = Image.open(all_stim_paths[i]).convert('RGB')
        crops.append(crop_to_bbox(pil, bboxes[i]))

    all_embeds = []
    for start in tqdm(range(0, HVM_N_STIMULI, BATCH_SIZE), desc='Embedding'):
        batch = crops[start : start + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors='pt').to(args.device)
        with torch.no_grad():
            out = vision_model(**inputs)
        emb = F.normalize(out.pooler_output, dim=-1).cpu()
        all_embeds.append(emb)

    embeds = torch.cat(all_embeds, dim=0)  # (450, 1152)
    assert embeds.shape == (HVM_N_STIMULI, 1152), embeds.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeds, out_path)
    print(f"Saved {tuple(embeds.shape)} → {out_path}")


if __name__ == '__main__':
    main()

"""Build per-stimulus canonical object latent cache for HVM.

For each of the 450 HVM stimuli:
  1. Load the stimulus at IMAGE_SIZE × IMAGE_SIZE
  2. Encode with the FLUX VAE → (1, 16, H_lat, W_lat)
  3. Crop the GDINO bbox region and resize to CANONICAL_LAT × CANONICAL_LAT
  4. Store as float32

Output: cache/hvm10_canonical_latents.pt — (450, 16, CANONICAL_LAT, CANONICAL_LAT) tensor

Usage:
    python scripts/build_canonical_latent_cache.py
    python scripts/build_canonical_latent_cache.py --canonical-lat 32 --image-size 512
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.append(".")

from config_const import (
    CACHE_DIR, HVM_STIM_DIR, HVM_N_STIMULI, HVM_N_VAR, HVM_CATEGORIES,
    HVM_CANONICAL_LATENTS_PATH,
)
from generation.flux_instantx import load_pipeline, encode_image
from generation.project_hvm import load_hvm_bboxes

IMAGE_SIZE    = 512
CANONICAL_LAT = 16   # 16×16 latent patches = 128px object region
VAE_STRIDE    = 8
HVM_SRC_SIZE  = 276  # bbox cache is in 276×276 space


def _bbox_to_latent(bbox, image_size, vae_stride=VAE_STRIDE, src_size=HVM_SRC_SIZE):
    scale = image_size / src_size / vae_stride
    return bbox['cx'] * scale, bbox['cy'] * scale, bbox['half'] * scale


def extract_canonical(lat4d, cx_l, cy_l, half_l, canonical_lat):
    r    = round(half_l)
    cx_i = round(cx_l)
    cy_i = round(cy_l)
    h_lat, w_lat = lat4d.shape[2], lat4d.shape[3]
    y0, y1 = max(0, cy_i - r), min(h_lat, cy_i + r)
    x0, x1 = max(0, cx_i - r), min(w_lat, cx_i + r)
    crop = lat4d[:, :, y0:y1, x0:x1]
    if crop.shape[2] == 0 or crop.shape[3] == 0:
        return torch.zeros(1, lat4d.shape[1], canonical_lat, canonical_lat,
                           dtype=lat4d.dtype)
    return F.interpolate(crop.float(), size=(canonical_lat, canonical_lat),
                         mode='bilinear', align_corners=False)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  image_size={args.image_size}  canonical_lat={args.canonical_lat}")

    pipe, _ = load_pipeline(device=str(device), default_scale=1.0)
    pipe.vae.eval()

    bboxes = load_hvm_bboxes()  # list of 450 dicts with cx, cy, half in 276px space

    all_stim_paths = sorted(
        p for cat in sorted(Path(HVM_STIM_DIR).iterdir()) if cat.is_dir()
        for p in sorted(cat.iterdir())
    )
    assert len(all_stim_paths) == HVM_N_STIMULI, f"Expected 450 stimuli, got {len(all_stim_paths)}"

    canonical_latents = torch.zeros(
        HVM_N_STIMULI, 16, args.canonical_lat, args.canonical_lat, dtype=torch.float32
    )

    for idx in tqdm(range(HVM_N_STIMULI), desc="Encoding canonical latents"):
        pil = Image.open(all_stim_paths[idx]).convert('RGB').resize(
            (args.image_size, args.image_size), Image.LANCZOS
        )
        with torch.no_grad():
            lat = encode_image(pipe, pil)   # (1, 16, H_lat, W_lat)

        bbox = bboxes[idx]
        cx_l, cy_l, half_l = _bbox_to_latent(bbox, args.image_size)
        canonical = extract_canonical(lat.float().cpu(), cx_l, cy_l, half_l, args.canonical_lat)
        canonical_latents[idx] = canonical.squeeze(0)

        if (idx + 1) % 50 == 0:
            cat_name = HVM_CATEGORIES[idx // HVM_N_VAR]
            print(f"  [{idx+1}/{HVM_N_STIMULI}] {cat_name} — bbox half_l={half_l:.1f}")

    out_path = CACHE_DIR / f"hvm10_canonical_latents_s{args.image_size}_c{args.canonical_lat}.pt"
    torch.save({
        'latents':       canonical_latents,
        'image_size':    args.image_size,
        'canonical_lat': args.canonical_lat,
        'vae_stride':    VAE_STRIDE,
        'hvm_src_size':  HVM_SRC_SIZE,
    }, out_path)
    print(f"\nSaved {tuple(canonical_latents.shape)} → {out_path}")

    if args.canonical_lat == CANONICAL_LAT and args.image_size == IMAGE_SIZE:
        import shutil
        shutil.copy(out_path, HVM_CANONICAL_LATENTS_PATH)
        print(f"Also copied to {HVM_CANONICAL_LATENTS_PATH}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image-size',    type=int, default=512)
    p.add_argument('--canonical-lat', type=int, default=CANONICAL_LAT)
    main(p.parse_args())

"""Remove the central fixation square from HVM or Rust cropped stimuli.

Uses the FLUX img2img pipeline to inpaint only the fixation-square region,
guided by GT SigLIP + GT CLIP embeddings so the filled region matches the
surrounding image content.

Output layout:
  stimuli/hvm_nofixation/{cat}/{k:02d}.png   (HVM)
  stimuli/rust_nofixation/{i:04d}.png        (Rust)

Usage:
  python -m data_utils.remove_fixation --dataset hvm
  python -m data_utils.remove_fixation --dataset rust [--overwrite] [--steps 30]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from PIL import Image

from config_const import (
    SEED,
    # HVM
    HVM_CATEGORIES, HVM_N_VAR, HVM_STIM_DIR, HVM_NOFIXATION_DIR,
    HVM_SIGLIP_EMBEDDINGS_PATH, HVM_CLIP_EMBEDS_PATH, HVM_CENTER_FRAC,
    # Rust
    N_STIMULI, RUST_STIM_DIR, RUST_NOFIXATION_DIR,
    SIGLIP_EMBEDDINGS_PATH, CLIP_EMBEDS_PATH, APERTURE_CENTER_FRAC,
)
from generation.flux_instantx import load_pipeline, generate_img2img, encode_text_embeds
from get_device import get_device


IMAGE_SIZE      = 512
NUM_STEPS       = 20
STRENGTH        = 1
GUIDANCE        = 3.5
IP_SCALE        = 1.0
FIXATION_PROMPT = "a seamless, coherent image with no artifacts"

_DATASETS = {
    "hvm": dict(
        stim_dir=HVM_STIM_DIR,
        out_dir=HVM_NOFIXATION_DIR,
        siglip_path=HVM_SIGLIP_EMBEDDINGS_PATH,
        clip_path=HVM_CLIP_EMBEDS_PATH,
        center_frac=HVM_CENTER_FRAC,
    ),
    "rust": dict(
        stim_dir=RUST_STIM_DIR,
        out_dir=RUST_NOFIXATION_DIR,
        siglip_path=SIGLIP_EMBEDDINGS_PATH,
        clip_path=CLIP_EMBEDS_PATH,
        center_frac=APERTURE_CENTER_FRAC,
    ),
}


def build_fixation_mask(image_size: int, center_frac: float) -> torch.Tensor:
    """Return packed-latent mask covering exactly the fixation square.

    Shape: (1, packed_seq_len, 1) where packed_seq_len = (image_size//16)**2.
    """
    half = int(round(image_size * center_frac / 2))
    c = int(round((image_size - 1) / 2))
    lo = c - half - 5
    hi = c + half + 6

    pixel_mask = torch.zeros(image_size, image_size)
    pixel_mask[lo:hi, lo:hi] = 1.0

    packed_grid = image_size // 16
    mask = F.avg_pool2d(
        pixel_mask.view(1, 1, image_size, image_size), kernel_size=16
    ).view(1, packed_grid * packed_grid, 1)

    return (mask > 0.5).float()


def _hvm_items(stim_dir: Path, out_dir: Path, overwrite: bool) -> Iterator[tuple[int, Path, Path]]:
    """Yield (global_idx, src_path, dst_path) for all HVM stimuli."""
    gidx = 0
    for cat in HVM_CATEGORIES:
        out_cat = out_dir / cat
        out_cat.mkdir(parents=True, exist_ok=True)
        for k in range(HVM_N_VAR):
            src = stim_dir / cat / f"{k:02d}.png"
            dst = out_cat / f"{k:02d}.png"
            if not dst.exists() or overwrite:
                yield gidx, src, dst
            gidx += 1


def _rust_items(stim_dir: Path, out_dir: Path, overwrite: bool) -> Iterator[tuple[int, Path, Path]]:
    """Yield (global_idx, src_path, dst_path) for all Rust stimuli."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_STIMULI):
        src = stim_dir / f"{i:04d}.png"
        dst = out_dir / f"{i:04d}.png"
        if not dst.exists() or overwrite:
            yield i, src, dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(_DATASETS), required=True,
                        help="Which stimulus dataset to process")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output images")
    parser.add_argument("--steps", type=int, default=NUM_STEPS,
                        help=f"FLUX inference steps (default: {NUM_STEPS})")
    parser.add_argument("--strength", type=float, default=STRENGTH)
    parser.add_argument("--ip-scale", type=float, default=IP_SCALE)
    args = parser.parse_args()

    cfg = _DATASETS[args.dataset]
    stim_dir   = cfg["stim_dir"]
    out_dir    = cfg["out_dir"]
    center_frac = cfg["center_frac"]

    device = get_device()
    dtype  = torch.bfloat16

    print(f"Dataset: {args.dataset}")
    print("Loading GT embeddings …")
    siglip_gt = torch.load(cfg["siglip_path"], weights_only=True)
    clip_gt   = torch.load(cfg["clip_path"],   weights_only=True)

    print("Loading FLUX pipeline …")
    pipe, image_proj = load_pipeline(device=device, dtype=dtype, default_scale=args.ip_scale)

    print("Encoding inpainting prompt …")
    prompt_clip, prompt_t5 = encode_text_embeds(pipe, [FIXATION_PROMPT])

    print("Building fixation-square mask …")
    fix_mask = build_fixation_mask(IMAGE_SIZE, center_frac).to(dtype=dtype)

    items = (
        _hvm_items(stim_dir, out_dir, args.overwrite)
        if args.dataset == "hvm"
        else _rust_items(stim_dir, out_dir, args.overwrite)
    )

    for gidx, src, dst in items:
        orig = Image.open(src).convert("RGB")
        orig_size = orig.size  # (W, H)
        init_img = orig.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

        result = generate_img2img(
            pipe,
            image_proj,
            init_img,
            siglip_gt[gidx],
            strength=args.strength,
            prompt_embeds=prompt_t5,
            pooled_prompt_embeds=prompt_clip,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=args.steps,
            guidance_scale=GUIDANCE,
            ip_adapter_scale=args.ip_scale,
            seed=SEED,
            aperture_mask=fix_mask,
            aperture_composite=True,
            show_progress=False,
        )

        result.resize(orig_size, Image.Resampling.LANCZOS).save(dst)
        print(f"  saved {dst.relative_to(out_dir.parent)}")

    print(f"\nDone. Output in {out_dir}")


if __name__ == "__main__":
    main()

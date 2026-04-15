"""Embed the 300 stimuli with frozen SigLIP and cache global + patch features.

Writes four cache files:
  cache/siglip_embeddings.pt  — (300, 1152)        global CLS, L2-normalised (raw aperture)
  cache/siglip_stripped.pt    — (300, 1152)        global CLS, aperture mask removed
  cache/siglip_patch14.pt     — (300, 196, 1152)   14x14 average-pooled patch grid
  cache/siglip_patch8.pt      — (300,  64, 1152)    8x8 average-pooled patch grid
"""

import argparse
from PIL import Image

import torch

from config_const import (
    N_STIMULI,
    RUST_STIM_DIR,
    SIGLIP_EMBEDDINGS_PATH,
    SIGLIP_STRIPPED_PATH,
    SIGLIP_PATCH8_PATH,
    SIGLIP_PATCH14_PATH,
)
from data_utils.stimuli import load_rust_stimuli
from encoders.siglip_embed import embed_images, embed_images_patches, embed_images_stripped, load_siglip


def load_rust_pils() -> list[Image.Image]:
    """Load raw RGBA stimulus PIL images (needed for aperture stripping)."""
    return [
        Image.open(RUST_STIM_DIR / f"{i:04d}.png")
        for i in range(N_STIMULI)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    SIGLIP_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    processor, vision_model = load_siglip(args.device)

    # --- tensor-based targets (global raw, patches) ---
    tensor_targets = [
        ("global", SIGLIP_EMBEDDINGS_PATH, None),
        ("patch14", SIGLIP_PATCH14_PATH, 14),
        ("patch8",  SIGLIP_PATCH8_PATH,   8),
    ]
    images = None
    for name, path, grid in tensor_targets:
        if path.exists() and not args.force:
            print(f"{path} exists; skipping {name}")
            continue
        if images is None:
            images = load_rust_stimuli()
        if grid is None:
            embeds = embed_images(processor, vision_model, images)
        else:
            embeds = embed_images_patches(processor, vision_model, images, grid_size=grid)
        torch.save(embeds, path)
        print(f"saved {tuple(embeds.shape)} -> {path}")

    # --- stripped variant (needs raw RGBA PIL images) ---
    if SIGLIP_STRIPPED_PATH.exists() and not args.force:
        print(f"{SIGLIP_STRIPPED_PATH} exists; skipping stripped")
    else:
        pils = load_rust_pils()
        embeds = embed_images_stripped(processor, vision_model, pils)
        torch.save(embeds, SIGLIP_STRIPPED_PATH)
        print(f"saved {tuple(embeds.shape)} -> {SIGLIP_STRIPPED_PATH}")


if __name__ == "__main__":
    main()

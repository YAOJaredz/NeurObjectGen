"""Embed the 300 stimuli with frozen SigLIP and cache global + patch features.

Writes three cache files:
  cache/siglip_embeddings.pt  — (300, 1152)        global CLS, L2-normalised
  cache/siglip_patch14.pt     — (300, 196, 1152)   14x14 average-pooled patch grid
  cache/siglip_patch8.pt      — (300,  64, 1152)    8x8 average-pooled patch grid
"""

import argparse

import torch

from config_const import (
    SIGLIP_EMBEDDINGS_PATH,
    SIGLIP_PATCH8_PATH,
    SIGLIP_PATCH14_PATH,
)
from data_utils.stimuli import load_rust_stimuli
from encoders.siglip_embed import embed_images, embed_images_patches, load_siglip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    targets = [
        ("global", SIGLIP_EMBEDDINGS_PATH, None),
        ("patch14", SIGLIP_PATCH14_PATH, 14),
        ("patch8",  SIGLIP_PATCH8_PATH,   8),
    ]

    if not args.force and all(p.exists() for _, p, _ in targets):
        print("All caches exist; use --force to overwrite")
        return

    SIGLIP_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    images = load_rust_stimuli()
    processor, vision_model = load_siglip(args.device)

    for name, path, grid in targets:
        if path.exists() and not args.force:
            print(f"{path} exists; skipping {name}")
            continue
        if grid is None:
            embeds = embed_images(processor, vision_model, images)
        else:
            embeds = embed_images_patches(processor, vision_model, images, grid_size=grid)
        torch.save(embeds, path)
        print(f"saved {tuple(embeds.shape)} -> {path}")


if __name__ == "__main__":
    main()

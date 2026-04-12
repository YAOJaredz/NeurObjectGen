"""Embed the 300 stimuli with frozen SigLIP and cache to cache/siglip_embeddings.pt."""

import argparse
from pathlib import Path

import torch

from config_const import SIGLIP_EMBEDDINGS_PATH
from data_utils.stimuli import load_rust_stimuli
from encoders.siglip_embed import embed_images, load_siglip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(SIGLIP_EMBEDDINGS_PATH))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; use --force to overwrite")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = load_rust_stimuli()
    processor, vision_model = load_siglip(args.device)
    embeddings = embed_images(processor, vision_model, images)
    torch.save(embeddings, out_path)
    print(f"saved {tuple(embeddings.shape)} -> {out_path}")


if __name__ == "__main__":
    main()

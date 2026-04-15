"""Encode the 300 BLIP2 captions with FLUX's CLIP and T5 text encoders and cache them.

Writes two cache files:
  cache/clip_embeds.pt  — (300, 768)        CLIP pooled embeddings, float32
  cache/t5_embeds.pt    — (300, 512, 4096)  T5 sequence embeddings, float32

Requires the FLUX pipeline weights (used to load the text encoders).
Both caches are stored on CPU so they can be loaded without a GPU.
"""

import argparse
import json

import torch

from config_const import BLIP2_CAPTIONS_PATH, CLIP_EMBEDS_PATH, T5_EMBEDS_PATH
from generation.flux_instantx import encode_text_embeds, load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if CLIP_EMBEDS_PATH.exists() and T5_EMBEDS_PATH.exists() and not args.force:
        print("Both caches exist; use --force to overwrite")
        return

    with open(BLIP2_CAPTIONS_PATH) as f:
        captions = json.load(f)

    prompts = [captions[str(i)] for i in range(len(captions))]
    print(f"Encoding {len(prompts)} captions …")

    pipe, _ = load_pipeline(device=args.device)

    clip_embeds, t5_embeds = encode_text_embeds(pipe, prompts)

    CLIP_EMBEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clip_embeds, CLIP_EMBEDS_PATH)
    torch.save(t5_embeds,   T5_EMBEDS_PATH)
    print(f"saved CLIP {tuple(clip_embeds.shape)} -> {CLIP_EMBEDS_PATH}")
    print(f"saved T5   {tuple(t5_embeds.shape)}   -> {T5_EMBEDS_PATH}")


if __name__ == "__main__":
    main()

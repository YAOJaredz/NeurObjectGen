"""Encode InstructBLIP captions with FLUX's CLIP and T5 text encoders and cache them.

Writes four cache files:
  cache/clip_embeds.pt          — (300, 768)       CLIP pooled, short captions
  cache/t5_embeds.pt            — (300, 512, 4096)  T5 sequence, short captions
  cache/clip_detailed_embeds.pt — (300, 768)       CLIP pooled, detailed captions
  cache/t5_detailed_embeds.pt   — (300, 512, 4096)  T5 sequence, detailed captions

Mean embeddings are computed in-place where needed (trivial operation, not worth caching).

Requires the FLUX pipeline weights (used to load the text encoders).
All caches are stored on CPU so they can be loaded without a GPU.
"""

import argparse
import json

import torch

from config_const import (
    BLIP2_CAPTIONS_PATH, BLIP2_DETAILED_CAPTIONS_PATH,
    CLIP_EMBEDS_PATH, T5_EMBEDS_PATH,
    CLIP_DETAILED_EMBEDS_PATH, T5_DETAILED_EMBEDS_PATH,
)
from generation.flux_instantx import encode_text_embeds, load_pipeline

_ALL_PATHS = [
    CLIP_EMBEDS_PATH, T5_EMBEDS_PATH,
    CLIP_DETAILED_EMBEDS_PATH, T5_DETAILED_EMBEDS_PATH,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if all(p.exists() for p in _ALL_PATHS) and not args.force:
        print("All caches exist; use --force to overwrite")
        return

    with open(BLIP2_CAPTIONS_PATH) as f:
        captions_short = json.load(f)
    with open(BLIP2_DETAILED_CAPTIONS_PATH) as f:
        captions_detailed = json.load(f)

    n = len(captions_short)
    short_prompts    = [captions_short[str(i)]    for i in range(n)]
    detailed_prompts = [captions_detailed[str(i)] for i in range(n)]

    pipe, _ = load_pipeline(device=args.device)

    print(f"Encoding {n} short captions …")
    clip_short, t5_short = encode_text_embeds(pipe, short_prompts)

    print(f"Encoding {n} detailed captions …")
    clip_detailed, t5_detailed = encode_text_embeds(pipe, detailed_prompts)

    CLIP_EMBEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clip_short,    CLIP_EMBEDS_PATH)
    torch.save(t5_short,      T5_EMBEDS_PATH)
    torch.save(clip_detailed, CLIP_DETAILED_EMBEDS_PATH)
    torch.save(t5_detailed,   T5_DETAILED_EMBEDS_PATH)

    for name, clip, t5 in [
        ("short",    clip_short,    t5_short),
        ("detailed", clip_detailed, t5_detailed),
    ]:
        print(f"  {name:8s}  CLIP {tuple(clip.shape)}  T5 {tuple(t5.shape)}")


if __name__ == "__main__":
    main()

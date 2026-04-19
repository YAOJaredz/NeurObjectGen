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
    HVM_N_STIMULI,
    HVM_BLIP2_CAPTIONS_PATH, HVM_BLIP2_DETAILED_CAPTIONS_PATH,
    HVM_CLIP_EMBEDS_PATH, HVM_CLIP_DETAILED_EMBEDS_PATH,
    HVM_T5_EMBEDS_PATH, HVM_T5_DETAILED_EMBEDS_PATH,
)
from generation.flux_instantx import encode_text_embeds, load_pipeline


def _encode_and_save(pipe, cap_short_path, cap_detailed_path, n,
                     clip_short_path, t5_short_path,
                     clip_detailed_path, t5_detailed_path, force):
    all_paths = [clip_short_path, t5_short_path, clip_detailed_path, t5_detailed_path]
    if all(p.exists() for p in all_paths) and not force:
        print("All caches exist; use --force to overwrite")
        return

    for p in [cap_short_path, cap_detailed_path]:
        if not p.exists():
            raise FileNotFoundError(f"Caption file not found: {p}. Run captions.py first.")

    with open(cap_short_path) as f:
        captions_short = json.load(f)
    with open(cap_detailed_path) as f:
        captions_detailed = json.load(f)

    short_prompts    = [captions_short[str(i)]    for i in range(n)]
    detailed_prompts = [captions_detailed[str(i)] for i in range(n)]

    print(f"Encoding {n} short captions ...")
    clip_short, t5_short = encode_text_embeds(pipe, short_prompts)

    print(f"Encoding {n} detailed captions ...")
    clip_detailed, t5_detailed = encode_text_embeds(pipe, detailed_prompts)

    clip_short_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clip_short,    clip_short_path)
    torch.save(t5_short,      t5_short_path)
    torch.save(clip_detailed, clip_detailed_path)
    torch.save(t5_detailed,   t5_detailed_path)

    for name, clip, t5 in [("short", clip_short, t5_short), ("detailed", clip_detailed, t5_detailed)]:
        print(f"  {name:8s}  CLIP {tuple(clip.shape)}  T5 {tuple(t5.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", choices=["rust", "hvm", "both"], default="rust")
    args = parser.parse_args()

    datasets = ["rust", "hvm"] if args.dataset == "both" else [args.dataset]
    pipe, _ = load_pipeline(device=args.device)

    if "rust" in datasets:
        from config_const import N_STIMULI
        _encode_and_save(
            pipe,
            BLIP2_CAPTIONS_PATH, BLIP2_DETAILED_CAPTIONS_PATH, N_STIMULI,
            CLIP_EMBEDS_PATH, T5_EMBEDS_PATH,
            CLIP_DETAILED_EMBEDS_PATH, T5_DETAILED_EMBEDS_PATH,
            args.force,
        )

    if "hvm" in datasets:
        _encode_and_save(
            pipe,
            HVM_BLIP2_CAPTIONS_PATH, HVM_BLIP2_DETAILED_CAPTIONS_PATH, HVM_N_STIMULI,
            HVM_CLIP_EMBEDS_PATH, HVM_T5_EMBEDS_PATH,
            HVM_CLIP_DETAILED_EMBEDS_PATH, HVM_T5_DETAILED_EMBEDS_PATH,
            args.force,
        )


if __name__ == "__main__":
    main()

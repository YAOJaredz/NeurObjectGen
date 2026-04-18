"""Encode detailed captions with standalone google/t5-v1_1-xxl and cache mean-pooled embeddings.

Output: cache/t5_xxl_pooled_detailed.pt — (300, 4096) float32, mean-pooled over real tokens.

Usage:
    python scripts/cache_t5_xxl_pooled.py [--device cuda] [--batch-size 8] [--force]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import BLIP2_DETAILED_CAPTIONS_PATH, N_STIMULI, T5_XXL_POOLED_PATH

MODEL = "google/t5-v1_1-xxl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",     default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--force",      action="store_true")
    args = p.parse_args()

    if T5_XXL_POOLED_PATH.exists() and not args.force:
        print(f"Cache already exists at {T5_XXL_POOLED_PATH}; use --force to overwrite")
        return

    with open(BLIP2_DETAILED_CAPTIONS_PATH) as f:
        caps = json.load(f)
    texts = [caps[str(i)] for i in range(N_STIMULI)]

    print(f"Loading {MODEL} ...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    encoder   = T5EncoderModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    print(f"  hidden dim: {encoder.config.d_model}")

    all_pooled = []
    for start in tqdm(range(0, N_STIMULI, args.batch_size), desc="Encoding"):
        batch  = texts[start : start + args.batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(args.device)

        with torch.no_grad():
            hidden = encoder(**tokens).last_hidden_state.float()  # (B, T, 4096)

        mask   = tokens["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)             # (B, 4096) masked mean pool
        all_pooled.append(pooled.cpu())

    out = torch.cat(all_pooled)  # (300, 4096)
    T5_XXL_POOLED_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, T5_XXL_POOLED_PATH)
    print(f"Saved {tuple(out.shape)} → {T5_XXL_POOLED_PATH}")


if __name__ == "__main__":
    main()

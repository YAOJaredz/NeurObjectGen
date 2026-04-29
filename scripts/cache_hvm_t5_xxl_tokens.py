"""Encode HVM short and detailed captions with google/t5-v1_1-xxl and cache per-token embeddings.

Saves per-caption token hidden states as padded tensor + length tensor.

Outputs (in cache/):
  hvm_t5_xxl_tokens_short.pt    — dict with keys:
      "hidden":  (450, max_seq_short,    4096) float32
      "lengths": (450,)                        int32
  hvm_t5_xxl_tokens_detailed.pt — same for detailed captions

Usage:
    python scripts/cache_hvm_t5_xxl_tokens.py [--device cuda] [--batch-size 4] [--force]
    python scripts/cache_hvm_t5_xxl_tokens.py --captions short
    python scripts/cache_hvm_t5_xxl_tokens.py --captions detailed
    python scripts/cache_hvm_t5_xxl_tokens.py --captions both  (default)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

sys.path.append('.')

from config_const import (
    HVM_BLIP2_CAPTIONS_PATH,
    HVM_BLIP2_DETAILED_CAPTIONS_PATH,
    CACHE_DIR,
    HVM_N_STIMULI,
    T5_XXL_MODEL_ID,
    HVM_T5_TOKENS_SHORT_PATH, HVM_T5_TOKENS_DETAILED_PATH,
)


def encode_and_save(
    texts: list[str],
    out_path: Path,
    device: str,
    batch_size: int,
    tokenizer: T5Tokenizer,
    encoder: T5EncoderModel,
):
    n_stimuli = len(texts)
    all_hidden  = []
    all_lengths = []

    for start in tqdm(range(0, n_stimuli, batch_size), desc=f"Encoding → {out_path.name}"):
        batch  = texts[start : start + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)

        with torch.no_grad():
            hidden = encoder(**tokens).last_hidden_state.float()  # (B, T, 4096)

        mask = tokens["attention_mask"]  # (B, T)
        for b in range(len(batch)):
            n   = int(mask[b].sum().item())
            emb = hidden[b, :n].cpu()               # (n, 4096)
            all_hidden.append(emb)
            all_lengths.append(n)

    max_len = max(all_lengths)
    D       = all_hidden[0].shape[-1]
    padded  = torch.zeros(n_stimuli, max_len, D, dtype=torch.float32)
    for i, (emb, n) in enumerate(zip(all_hidden, all_lengths)):
        padded[i, :n] = emb

    lengths_t = torch.tensor(all_lengths, dtype=torch.int32)

    out = {"hidden": padded, "lengths": lengths_t}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved {tuple(padded.shape)} + lengths → {out_path}")
    print(f"  real tokens: min={lengths_t.min()} max={lengths_t.max()} mean={lengths_t.float().mean():.1f}")
    print(f"  total token embeddings: {int(lengths_t.sum())}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",     default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--force",      action="store_true")
    p.add_argument("--captions",   choices=["short", "detailed", "both"], default="both")
    args = p.parse_args()

    sources = []
    if args.captions in ("short", "both"):
        sources.append((HVM_BLIP2_CAPTIONS_PATH, HVM_T5_TOKENS_SHORT_PATH, "short"))
    if args.captions in ("detailed", "both"):
        sources.append((HVM_BLIP2_DETAILED_CAPTIONS_PATH, HVM_T5_TOKENS_DETAILED_PATH, "detailed"))

    to_encode = []
    for cap_path, out_path, label in sources:
        if out_path.exists() and not args.force:
            print(f"Cache already exists at {out_path}; use --force to overwrite")
        else:
            to_encode.append((cap_path, out_path, label))

    if not to_encode:
        return

    print(f"Loading {T5_XXL_MODEL_ID} ...")
    tokenizer = T5Tokenizer.from_pretrained(T5_XXL_MODEL_ID)
    encoder   = T5EncoderModel.from_pretrained(T5_XXL_MODEL_ID, torch_dtype=torch.bfloat16).to(args.device).eval()

    for cap_path, out_path, label in to_encode:
        with open(cap_path) as f:
            caps = json.load(f)
        texts = [caps[str(i)] for i in range(HVM_N_STIMULI)]
        print(f"\n=== HVM {label} captions ({len(texts)} stimuli) ===")
        encode_and_save(texts, out_path, args.device, args.batch_size, tokenizer, encoder)

    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

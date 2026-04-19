"""Encode short and detailed captions with google/t5-v1_1-xxl and cache per-token embeddings.

Saves per-caption token hidden states and attention masks (real tokens only, no padding)
as ragged lists packed into a padded tensor + length tensor.

Outputs (in cache/):
  t5_xxl_tokens_short.pt    — dict with keys:
      "hidden":  (300, max_seq_short,    4096) float32
      "lengths": (300,)                        int32  — number of real tokens per stimulus
  t5_xxl_tokens_detailed.pt — same for detailed captions

Usage:
    python scripts/cache_t5_xxl_tokens.py [--device cuda] [--batch-size 4] [--force]
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

from config_const import (
    BLIP2_CAPTIONS_PATH,
    BLIP2_DETAILED_CAPTIONS_PATH,
    CACHE_DIR,
    N_STIMULI,
)

MODEL = "google/t5-v1_1-xxl"

T5_TOKENS_SHORT_PATH    = CACHE_DIR / "t5_xxl_tokens_short.pt"
T5_TOKENS_DETAILED_PATH = CACHE_DIR / "t5_xxl_tokens_detailed.pt"


def encode_and_save(texts: list[str], out_path: Path, device: str, batch_size: int):
    print(f"Loading {MODEL} ...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    encoder   = T5EncoderModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()

    all_hidden  = []   # list of (seq_i, 4096) float32 tensors (real tokens only)
    all_lengths = []   # list of ints

    for start in tqdm(range(0, len(texts), batch_size), desc=f"Encoding → {out_path.name}"):
        batch  = texts[start : start + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)

        with torch.no_grad():
            hidden = encoder(**tokens).last_hidden_state.float()  # (B, T, 4096)

        mask = tokens["attention_mask"]  # (B, T)
        for b in range(len(batch)):
            n   = int(mask[b].sum().item())         # real token count (incl EOS)
            emb = hidden[b, :n].cpu()               # (n, 4096)
            all_hidden.append(emb)
            all_lengths.append(n)

    # Pack into a padded tensor
    max_len = max(all_lengths)
    D       = all_hidden[0].shape[-1]
    padded  = torch.zeros(N_STIMULI, max_len, D, dtype=torch.float32)
    for i, (emb, n) in enumerate(zip(all_hidden, all_lengths)):
        padded[i, :n] = emb

    lengths_t = torch.tensor(all_lengths, dtype=torch.int32)

    out = {"hidden": padded, "lengths": lengths_t}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved {tuple(padded.shape)} + lengths → {out_path}")
    print(f"  real tokens: min={lengths_t.min()} max={lengths_t.max()} mean={lengths_t.float().mean():.1f}")
    print(f"  total token embeddings: {int(lengths_t.sum())}")

    del encoder
    torch.cuda.empty_cache() if device.startswith("cuda") else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",     default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--force",      action="store_true")
    args = p.parse_args()

    sources = [
        (BLIP2_CAPTIONS_PATH,          T5_TOKENS_SHORT_PATH,    "short"),
        (BLIP2_DETAILED_CAPTIONS_PATH,  T5_TOKENS_DETAILED_PATH, "detailed"),
    ]

    for cap_path, out_path, label in sources:
        if out_path.exists() and not args.force:
            print(f"Cache already exists at {out_path}; use --force to overwrite")
            continue
        with open(cap_path) as f:
            caps = json.load(f)
        texts = [caps[str(i)] for i in range(N_STIMULI)]
        print(f"\n=== {label} captions ({len(texts)} stimuli) ===")
        encode_and_save(texts, out_path, args.device, args.batch_size)


if __name__ == "__main__":
    main()

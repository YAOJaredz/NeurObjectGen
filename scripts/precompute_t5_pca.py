"""Offline PCA compression of T5 sequence embeddings.

Mean-pools T5 tokens (512, 4096) → 4096-d, then computes a truncated SVD to
obtain a K-dimensional coordinate basis. Saves three files to cache/:
  t5_pca_basis_kK.pt  — (K, 4096) right singular vectors (the PCA basis)
  t5_pca_coords_kK.pt — (300, K)  projection of all stimuli onto the basis
  t5_pca_mean_kK.pt   — (4096,)   per-dimension mean (subtract before projecting)

Usage:
    python scripts/precompute_t5_pca.py [--k 64] [--source short|detailed]
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import (
    T5_EMBEDS_PATH, T5_DETAILED_EMBEDS_PATH,
    T5_PCA_K, T5_PCA_BASIS_PATH, T5_PCA_COORDS_PATH, T5_PCA_MEAN_PATH,
    CACHE_DIR,
)


def precompute_t5_pca(k: int, source: str) -> None:
    src_path = T5_DETAILED_EMBEDS_PATH if source == "detailed" else T5_EMBEDS_PATH
    print(f"Loading T5 embeddings from {src_path} ...")
    t5 = torch.load(src_path, weights_only=True).float()   # (300, 512, 4096)
    print(f"  Loaded: {tuple(t5.shape)}")

    # Mean-pool over token dimension
    x = t5.mean(dim=1)   # (300, 4096)
    print(f"  Mean-pooled: {tuple(x.shape)}")

    # Center
    mean = x.mean(dim=0)  # (4096,)
    x_centered = x - mean

    # Truncated SVD (full_matrices=False → U:(300,300), S:(300,), Vh:(300,4096))
    print("Computing SVD ...")
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)

    basis  = Vh[:k]                       # (K, 4096)
    coords = x_centered @ basis.T         # (300, K)

    # Variance explained
    var_total    = S.pow(2).sum().item()
    var_top_k    = S[:k].pow(2).sum().item()
    pct_explained = 100.0 * var_top_k / var_total
    print(f"  Top-{k} components: {pct_explained:.1f}% variance explained")

    # Round-trip sanity check
    recon = coords @ basis
    err   = (x_centered - recon).norm(dim=-1).mean().item()
    print(f"  Mean reconstruction error: {err:.4f}")

    # Derive per-K paths in case K differs from config default
    basis_path  = CACHE_DIR / f"t5_pca_basis_k{k}.pt"
    coords_path = CACHE_DIR / f"t5_pca_coords_k{k}.pt"
    mean_path   = CACHE_DIR / f"t5_pca_mean_k{k}.pt"

    torch.save(basis,  basis_path)
    torch.save(coords, coords_path)
    torch.save(mean,   mean_path)
    print(f"Saved:\n  {basis_path}\n  {coords_path}\n  {mean_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k",      type=int, default=T5_PCA_K,
                   help=f"Number of PCA components (default: {T5_PCA_K})")
    p.add_argument("--source", choices=["short", "detailed"], default="detailed",
                   help="Which T5 embed cache to use (default: detailed captions)")
    args = p.parse_args()
    precompute_t5_pca(args.k, args.source)

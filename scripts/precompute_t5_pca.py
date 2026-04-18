"""Offline PCA compression of mean-pooled T5-xxl embeddings.

Reads the pre-computed standalone T5-xxl pooled cache (300, 4096) and fits a
truncated SVD to obtain a K-dimensional coordinate basis. Saves three files:

  cache/t5_pca_basis_kK.pt  — (K, 4096)  right singular vectors
  cache/t5_pca_coords_kK.pt — (300, K)   projection of all stimuli
  cache/t5_pca_mean_kK.pt   — (4096,)    per-dimension mean (subtract before projecting)

Requires cache_t5_xxl_pooled.py to have been run first.

Usage:
    python scripts/precompute_t5_pca.py [--k 128]
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import (
    T5_XXL_POOLED_PATH,
    T5_PCA_K, T5_PCA_BASIS_PATH, T5_PCA_COORDS_PATH, T5_PCA_MEAN_PATH,
    CACHE_DIR,
)


def precompute_t5_pca(k: int) -> None:
    if not T5_XXL_POOLED_PATH.exists():
        raise FileNotFoundError(
            f"Pooled cache not found at {T5_XXL_POOLED_PATH}. "
            "Run: python scripts/cache_t5_xxl_pooled.py"
        )

    print(f"Loading pooled T5-xxl embeddings from {T5_XXL_POOLED_PATH} ...")
    x = torch.load(T5_XXL_POOLED_PATH, weights_only=True).float()  # (300, 4096)
    print(f"  Loaded pooled: {tuple(x.shape)}")

    # Center
    mean = x.mean(dim=0)       # (4096,)
    x_centered = x - mean

    # Truncated SVD
    print("Computing SVD ...")
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)

    basis  = Vh[:k]                  # (K, 4096)
    coords = x_centered @ basis.T    # (300, K)

    var_total     = S.pow(2).sum().item()
    var_top_k     = S[:k].pow(2).sum().item()
    pct_explained = 100.0 * var_top_k / var_total
    print(f"  Top-{k} components: {pct_explained:.1f}% variance explained")

    recon = coords @ basis
    err   = (x_centered - recon).norm(dim=-1).mean().item()
    print(f"  Mean reconstruction error: {err:.4f}")

    basis_path  = CACHE_DIR / f"t5_pca_basis_k{k}.pt"
    coords_path = CACHE_DIR / f"t5_pca_coords_k{k}.pt"
    mean_path   = CACHE_DIR / f"t5_pca_mean_k{k}.pt"

    torch.save(basis,  basis_path)
    torch.save(coords, coords_path)
    torch.save(mean,   mean_path)
    print(f"Saved:\n  {basis_path}\n  {coords_path}\n  {mean_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=T5_PCA_K,
                   help=f"Number of PCA components (default: {T5_PCA_K})")
    args = p.parse_args()
    precompute_t5_pca(args.k)

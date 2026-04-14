"""Train a ridge regression decoder mapping IT responses -> spatial latent targets.

The spatial target is a PCA-compressed representation of the FLUX VAE packed latent
of each stimulus.  At inference the predicted PCA codes are projected back into latent
space and used as the init_image for generate_img2img(..., ip_adapter_scale=<semantic>).

Pipeline:
  1. Encode every training stimulus with the FLUX VAE -> packed latent (1, L, C).
     (cached to cache/spatial_latents_<size>.pt via generation/latent_cache.py)
  2. Flatten to (N, L*C) and fit PCA on the training split.
  3. Project all splits to n_components PCA codes.
  4. Fit closed-form ridge regression: X_neural (N, neurons*time) -> Z_pca (N, K).
  5. Evaluate on val/test: report per-component R² and 2AFC identification in PCA space.

Usage:
    python scripts/train_spatial_ridge.py
    python scripts/train_spatial_ridge.py --n-components 64 --alpha 1e4
    python scripts/train_spatial_ridge.py --image-size 256 --no-cache
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '.')

from config_const import CHECKPOINT_DIR, N_STIMULI, N_TRAIN, N_VAL, SEED
from data_utils.rust_loader import make_rust_loader
from eval.metrics import r2_per_component, two_afc_identification
from generation.latent_cache import load_latent_cache
from spatial_decoder.ridge import fit_ridge, decode_spatial_ridge  # noqa: F401  (re-exported for callers)


def train(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- neural data ---
    train_loader, val_loader, test_loader = make_rust_loader(
        batch_size=256, use_embeddings=False,
    )

    def collect_neural(loader) -> np.ndarray:
        return torch.cat([neural for neural, _ in loader], dim=0).numpy()

    X_train = collect_neural(train_loader).reshape(N_TRAIN, -1)
    X_val   = collect_neural(val_loader).reshape(N_VAL, -1)
    n_test  = N_STIMULI - N_TRAIN - N_VAL
    X_test  = collect_neural(test_loader).reshape(n_test, -1)

    print(f"Neural features: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # --- spatial latent targets ---
    latents_flat = load_latent_cache(args.image_size, args.device, force=args.no_cache)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_STIMULI)
    train_idx = perm[:N_TRAIN]
    val_idx   = perm[N_TRAIN:N_TRAIN + N_VAL]
    test_idx  = perm[N_TRAIN + N_VAL:]

    Z_all   = latents_flat.numpy()
    Z_train = Z_all[train_idx]
    Z_val   = Z_all[val_idx]
    Z_test  = Z_all[test_idx]

    # --- PCA on training split only ---
    print(f"Fitting PCA: {Z_train.shape[1]} -> {args.n_components} components...")
    Z_mean = Z_train.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Z_train - Z_mean, full_matrices=False)
    V = Vt[:args.n_components].T   # (D_latent, K)

    explained = (S[:args.n_components] ** 2).sum() / (S ** 2).sum()
    print(f"PCA explained variance: {explained*100:.1f}% with {args.n_components} components")

    Y_train = (Z_train - Z_mean) @ V
    Y_val   = (Z_val   - Z_mean) @ V
    Y_test  = (Z_test  - Z_mean) @ V

    # --- ridge regression ---
    print(f"Fitting ridge (alpha={args.alpha:.1e})...")
    W = fit_ridge(X_train, Y_train, alpha=args.alpha)

    # --- evaluate ---
    Y_val_pred  = X_val  @ W
    Y_test_pred = X_test @ W

    r2_val  = r2_per_component(Y_val_pred,  Y_val)
    r2_test = r2_per_component(Y_test_pred, Y_test)

    print(f"\nVal  R²: mean={r2_val.mean():.4f}  median={np.median(r2_val):.4f}  "
          f"frac>0={(r2_val > 0).mean()*100:.1f}%")
    print(f"Test R²: mean={r2_test.mean():.4f}  median={np.median(r2_test):.4f}  "
          f"frac>0={(r2_test > 0).mean()*100:.1f}%")

    afc_val  = two_afc_identification(
        torch.from_numpy(Y_val_pred).float(),
        torch.from_numpy(Y_val).float(),
    )
    afc_test = two_afc_identification(
        torch.from_numpy(Y_test_pred).float(),
        torch.from_numpy(Y_test).float(),
    )
    print(f"Val  2AFC: {afc_val:.3f}  |  Test 2AFC: {afc_test:.3f}")

    # --- save ---
    ckpt_dir = CHECKPOINT_DIR / "spatial_ridge"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    save_path = ckpt_dir / f"ridge_K{args.n_components}_a{args.alpha:.0e}.npz"
    np.savez(
        save_path,
        W=W,
        pca_mean=Z_mean,
        pca_V=V,
        image_size=np.array([args.image_size]),
        alpha=np.array([args.alpha]),
        n_components=np.array([args.n_components]),
        val_r2_mean=np.array([r2_val.mean()]),
        test_r2_mean=np.array([r2_test.mean()]),
        val_2afc=np.array([afc_val]),
        test_2afc=np.array([afc_test]),
    )
    print(f"\nSaved -> {save_path}")
    return save_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-components", type=int, default=128)
    p.add_argument("--alpha", type=float, default=1e4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-cache", action="store_true",
                   help="Re-encode stimuli even if a latent cache already exists")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

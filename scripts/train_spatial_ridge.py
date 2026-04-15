"""Train a ridge regression decoder mapping IT responses -> spatial targets.

Two target modes:

  pixel  (default)
    Downsample each stimulus to `--image-size` grayscale pixels, flatten to
    (N, H*W), fit PCA, then ridge. At inference, predicted PCA codes are
    reconstructed to a grayscale image and upsampled to the full resolution
    before being passed as init_image to generate_img2img.

  latent
    Use the FLUX VAE packed latent as the target (original behaviour).
    (cached to cache/spatial_latents_<size>.pt via generation/latent_cache.py)

Pipeline:
  1. Build targets: pixel array (N, H*W) or VAE latents (N, L*C).
  2. Fit PCA on the training split -> n_components codes.
  3. Fit closed-form ridge regression: X_neural -> Z_pca.
  4. Evaluate: per-component R² and 2AFC on train/val/test.

Usage:
    python scripts/train_spatial_ridge.py
    python scripts/train_spatial_ridge.py --target pixel --image-size 32 --n-components 64
    python scripts/train_spatial_ridge.py --target latent --image-size 512 --alpha 1e8
"""

import argparse
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '.')

from config_const import CHECKPOINT_DIR, N_STIMULI, N_TRAIN, N_VAL, SEED
from data_utils.rust_loader import make_rust_loader
from eval.metrics import r2_per_component, two_afc_identification
from generation.latent_cache import build_pixel_targets, load_latent_cache
from spatial_decoder.ridge import fit_ridge


def _load_targets(args) -> torch.Tensor:
    if args.target == "pixel":
        return build_pixel_targets(args.image_size)
    else:
        return load_latent_cache(args.image_size, args.device, force=args.no_cache)


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

    # --- optional neural PCA ---
    if args.n_neural_pcs > 0:
        print(f"Fitting neural PCA: {X_train.shape[1]} -> {args.n_neural_pcs} components...")
        X_mean = X_train.mean(axis=0, keepdims=True)
        _, _, Vt_x = np.linalg.svd(X_train - X_mean, full_matrices=False)
        V_x = Vt_x[:args.n_neural_pcs].T   # (d_in, n_neural_pcs)
        X_train = (X_train - X_mean) @ V_x
        X_val   = (X_val   - X_mean) @ V_x
        X_test  = (X_test  - X_mean) @ V_x
        print(f"Neural features after PCA: {X_train.shape}")

    # --- targets ---
    targets_flat = _load_targets(args)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_STIMULI)
    train_idx = perm[:N_TRAIN]
    val_idx   = perm[N_TRAIN:N_TRAIN + N_VAL]
    test_idx  = perm[N_TRAIN + N_VAL:]

    Z_all   = targets_flat.numpy()
    Z_train = Z_all[train_idx]
    Z_val   = Z_all[val_idx]
    Z_test  = Z_all[test_idx]

    # --- PCA on training split only ---
    print(f"Fitting PCA: {Z_train.shape[1]} -> {args.n_components} components...")
    Z_mean = Z_train.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Z_train - Z_mean, full_matrices=False)
    V = Vt[:args.n_components].T   # (D, K)

    explained = (S[:args.n_components] ** 2).sum() / (S ** 2).sum()
    print(f"PCA explained variance: {explained*100:.1f}% with {args.n_components} components")

    Y_train = (Z_train - Z_mean) @ V
    Y_val   = (Z_val   - Z_mean) @ V
    Y_test  = (Z_test  - Z_mean) @ V

    # --- ridge regression ---
    print(f"Fitting ridge (alpha={args.alpha:.1e})...")
    W = fit_ridge(X_train, Y_train, alpha=args.alpha)

    # --- evaluate ---
    Y_train_pred = X_train @ W
    Y_val_pred   = X_val   @ W
    Y_test_pred  = X_test  @ W

    r2_train = r2_per_component(Y_train_pred, Y_train)
    r2_val   = r2_per_component(Y_val_pred,   Y_val)
    r2_test  = r2_per_component(Y_test_pred,  Y_test)

    print(f"\nTrain R²: mean={r2_train.mean():.4f}  median={np.median(r2_train):.4f}  "
          f"frac>0={(r2_train > 0).mean()*100:.1f}%")
    print(f"Val   R²: mean={r2_val.mean():.4f}  median={np.median(r2_val):.4f}  "
          f"frac>0={(r2_val > 0).mean()*100:.1f}%")
    print(f"Test  R²: mean={r2_test.mean():.4f}  median={np.median(r2_test):.4f}  "
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

    save_path = ckpt_dir / f"ridge_{args.target}_K{args.n_components}_a{args.alpha:.0e}.npz"
    extras = {}
    if args.n_neural_pcs > 0:
        extras["neural_pca_mean"] = X_mean
        extras["neural_pca_V"]    = V_x

    np.savez(
        save_path,
        W=W,
        pca_mean=Z_mean,
        pca_V=V,
        target=np.array([args.target]),
        image_size=np.array([args.image_size]),
        alpha=np.array([args.alpha]),
        n_components=np.array([args.n_components]),
        n_neural_pcs=np.array([args.n_neural_pcs]),
        **extras,
        train_r2_mean=np.array([r2_train.mean()]),
        val_r2_mean=np.array([r2_val.mean()]),
        test_r2_mean=np.array([r2_test.mean()]),
        val_2afc=np.array([afc_val]),
        test_2afc=np.array([afc_test]),
    )
    print(f"\nSaved -> {save_path}")
    return save_path


def project_neural(neural_flat: np.ndarray, ckpt: dict) -> np.ndarray:
    """Apply neural PCA projection from checkpoint if present."""
    if "neural_pca_mean" in ckpt:
        neural_flat = (neural_flat - ckpt["neural_pca_mean"]) @ ckpt["neural_pca_V"]
    return neural_flat


def decode_to_pil(pca_codes: np.ndarray, ckpt: dict, output_size: int = 512) -> list[Image.Image]:
    """Reconstruct predicted pixel images from PCA codes and upsample.

    Args:
        pca_codes:   (N, K) predicted PCA codes from ridge
        ckpt:        loaded .npz checkpoint dict
        output_size: final PIL image size (upsampled from image_size)

    Returns:
        List of N PIL RGB images ready for generate_img2img as init_image.
    """
    Z_mean = ckpt["pca_mean"]   # (1, H*W)
    V      = ckpt["pca_V"]      # (H*W, K)
    hw     = int(ckpt["image_size"][0])

    pixels = pca_codes @ V.T + Z_mean   # (N, H*W)
    pixels = np.clip(pixels, 0.0, 1.0)

    images = []
    for px in pixels:
        gray = Image.fromarray((px.reshape(hw, hw) * 255).astype(np.uint8), mode="L")
        images.append(gray.convert("RGB").resize((output_size, output_size), Image.LANCZOS))
    return images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["pixel", "latent"], default="pixel")
    p.add_argument("--n-neural-pcs", type=int, default=100,
                   help="PCA on neural features before ridge. 0 = disabled.")
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1e4)
    p.add_argument("--image-size", type=int, default=32,
                   help="Pixel target: downsample size. Latent target: VAE encode size.")
    p.add_argument("--device", default="cuda",
                   help="Used only for latent target VAE encoding.")
    p.add_argument("--no-cache", action="store_true",
                   help="Re-encode stimuli even if a latent cache already exists (latent target only).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

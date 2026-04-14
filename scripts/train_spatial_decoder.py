"""Train a nonlinear spatial decoder (MLP or Transformer) mapping IT responses -> PCA latent codes.

Shares the VAE latent cache built by generation/latent_cache.py — run
train_spatial_ridge.py first (or pass --build-cache) to populate it.

The model outputs K-dimensional PCA codes with MSE loss. No L2 normalisation
on the output (unlike the semantic encoder) since PCA codes are regression targets.

Architecture choices:
  --model mlp         SpatialMLP: flatten (neurons*time) -> bottleneck -> K
  --model transformer SpatialTransformer: Pre-LN CLS transformer over time axis -> K

Usage:
    python scripts/train_spatial_decoder.py --model mlp --bottleneck 256
    python scripts/train_spatial_decoder.py --model transformer --d-model 64 --n-heads 4 --n-layers 2
    python scripts/train_spatial_decoder.py --model mlp --n-components 64 --alpha-l2 1e-3 --dropout 0.2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '.')

from config_const import CHECKPOINT_DIR, N_STIMULI, N_TRAIN, N_VAL, SEED
from data_utils.rust_loader import make_rust_loader
from eval.metrics import r2_per_component, two_afc_identification
from generation.latent_cache import build_latent_cache, latent_cache_path
from get_device import get_device
from spatial_decoder import SpatialMLP, SpatialTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(args, n_neurons: int, n_time: int, out_dim: int):
    if args.model == "mlp":
        return SpatialMLP(
            in_dim=n_neurons * n_time,
            bottleneck=args.bottleneck,
            out_dim=out_dim,
            dropout=args.dropout,
        )
    else:
        return SpatialTransformer(
            n_neurons=n_neurons,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            out_dim=out_dim,
            dropout=args.dropout,
        )


def prepare_input(neural: torch.Tensor, model_name: str) -> torch.Tensor:
    """(B, N, T) -> model-specific shape."""
    if model_name == "mlp":
        return neural.flatten(1)        # (B, N*T)
    else:
        return neural.permute(0, 2, 1)  # (B, T, N)


def run_name(args) -> str:
    shared = f"K{args.n_components}_do{args.dropout}_lr{args.lr}_wd{args.alpha_l2}_bs{args.batch_size}"
    if args.model == "mlp":
        return f"mlp_bn{args.bottleneck}_{shared}"
    else:
        return f"transformer_d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}_{shared}"


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args, pca_mean, pca_V):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "pca_mean": pca_mean,
        "pca_V": pca_V,
    }, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()

    # --- neural data ---
    train_loader, val_loader, test_loader = make_rust_loader(
        batch_size=args.batch_size, use_embeddings=False,
    )

    sample_neural, _ = next(iter(train_loader))
    _, n_neurons, n_time = sample_neural.shape

    def collect_neural(loader) -> np.ndarray:
        return torch.cat([n for n, _ in loader], dim=0).numpy()

    X_train_np = collect_neural(train_loader)
    X_val_np   = collect_neural(val_loader)
    X_test_np  = collect_neural(test_loader)

    # --- spatial latent targets ---
    path = latent_cache_path(args.image_size)
    if path.exists() and not args.build_cache:
        latents_flat = torch.load(path, weights_only=True).numpy()
        print(f"Loaded latent cache: {latents_flat.shape} from {path}")
    else:
        latents_flat = build_latent_cache(args.image_size, args.device).numpy()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_STIMULI)
    train_idx = perm[:N_TRAIN]
    val_idx   = perm[N_TRAIN:N_TRAIN + N_VAL]
    test_idx  = perm[N_TRAIN + N_VAL:]

    Z_train = latents_flat[train_idx]
    Z_val   = latents_flat[val_idx]
    Z_test  = latents_flat[test_idx]

    # PCA on training split only
    print(f"Fitting PCA: {Z_train.shape[1]} -> {args.n_components} components...")
    Z_mean = Z_train.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(Z_train - Z_mean, full_matrices=False)
    pca_V = Vt[:args.n_components].T
    explained = (S[:args.n_components] ** 2).sum() / (S ** 2).sum()
    print(f"PCA explained variance: {explained*100:.1f}%")

    Y_train_np = (Z_train - Z_mean) @ pca_V
    Y_val_np   = (Z_val   - Z_mean) @ pca_V
    Y_test_np  = (Z_test  - Z_mean) @ pca_V

    def to_loader(X_np, Y_np, shuffle):
        X = torch.from_numpy(X_np).float()
        Y = torch.from_numpy(Y_np).float()
        return DataLoader(TensorDataset(X, Y), batch_size=args.batch_size, shuffle=shuffle)

    tr_dl = to_loader(X_train_np, Y_train_np, shuffle=True)
    va_dl = to_loader(X_val_np,   Y_val_np,   shuffle=False)
    te_dl = to_loader(X_test_np,  Y_test_np,  shuffle=False)

    # --- model ---
    model = build_model(args, n_neurons, n_time, out_dim=args.n_components).to(device)
    print(f"Model: {model.__class__.__name__} | params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.alpha_l2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = CHECKPOINT_DIR / "spatial_decoder" / args.model / run_name(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    pca_mean_t = torch.from_numpy(Z_mean).float()
    pca_V_t    = torch.from_numpy(pca_V).float()
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for neural, pca_target in tr_dl:
            neural     = neural.to(device)
            pca_target = pca_target.to(device)
            pred = model(prepare_input(neural, args.model))
            loss = F.mse_loss(pred, pca_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(tr_dl)
        scheduler.step()

        # --- val ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for neural, pca_target in va_dl:
                neural     = neural.to(device)
                pca_target = pca_target.to(device)
                pred = model(prepare_input(neural, args.model))
                val_loss += F.mse_loss(pred, pca_target).item()
                all_preds.append(pred.cpu())
                all_targets.append(pca_target.cpu())
        val_loss /= len(va_dl)

        Y_val_pred = torch.cat(all_preds).numpy()
        Y_val_true = torch.cat(all_targets).numpy()
        r2_val  = r2_per_component(Y_val_pred, Y_val_true)
        afc_val = two_afc_identification(
            torch.from_numpy(Y_val_pred).float(),
            torch.from_numpy(Y_val_true).float(),
        )

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"R²={r2_val.mean():.4f}  2AFC={afc_val:.3f}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_r2_mean": float(r2_val.mean()),
            "val_2afc": afc_val,
        }
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args, pca_mean_t, pca_V_t)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args, pca_mean_t, pca_V_t)

    # --- test eval ---
    best = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(best["model_state"])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for neural, pca_target in te_dl:
            pred = model(prepare_input(neural.to(device), args.model))
            all_preds.append(pred.cpu())
            all_targets.append(pca_target)

    Y_test_pred = torch.cat(all_preds).numpy()
    Y_test_true = torch.cat(all_targets).numpy()
    r2_test  = r2_per_component(Y_test_pred, Y_test_true)
    afc_test = two_afc_identification(
        torch.from_numpy(Y_test_pred).float(),
        torch.from_numpy(Y_test_true).float(),
    )
    print(
        f"\nTest R²: mean={r2_test.mean():.4f}  median={np.median(r2_test):.4f}  "
        f"frac>0={(r2_test > 0).mean()*100:.1f}%"
    )
    print(f"Test 2AFC: {afc_test:.3f}")
    print(f"\nBest checkpoint -> {ckpt_dir / 'best.pt'}")


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def decode_spatial_nn(
    neural: torch.Tensor,
    ckpt_path: str | Path,
    device: str = "cpu",
) -> np.ndarray:
    """Map neural response tensor(s) to flat FLUX packed latent(s).

    Args:
        neural:    (neurons, time) or (N, neurons, time) float32 tensor
        ckpt_path: path to a best.pt saved by train()
        device:    inference device

    Returns:
        (N, D_latent) float32 numpy array; reshape to (1, L, C_packed) before
        passing to generate_img2img as an init latent.
    """
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    saved_args = argparse.Namespace(**ckpt["args"])
    pca_mean = ckpt["pca_mean"].to(device)
    pca_V    = ckpt["pca_V"].to(device)

    if neural.ndim == 2:
        neural = neural.unsqueeze(0)
    neural = neural.float().to(device)

    _, n_neurons, n_time = neural.shape
    K = pca_V.shape[1]
    model = build_model(saved_args, n_neurons, n_time, out_dim=K).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        pca_codes = model(prepare_input(neural, saved_args.model))

    return (pca_codes @ pca_V.T + pca_mean).cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-components", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha-l2", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--build-cache", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

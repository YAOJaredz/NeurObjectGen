"""Train a neural encoder to predict canonical object latents from HVM IT responses.

The prediction target is a CANONICAL_LAT × CANONICAL_LAT VAE latent patch cropped
from the object's GDINO bounding box and resized to a fixed spatial resolution.
This gives a consistent shape across all stimuli regardless of object scale.

At inference:
  1. Model predicts canonical patch (16, CANONICAL_LAT, CANONICAL_LAT)
  2. Resize to bbox size in latent space
  3. Paste into full init latent at (cx_l, cy_l)
  4. Use bbox guidance_mask to restrict RF correction to the object region

Usage:
    python train/train_canonical_latent.py
    python train/train_canonical_latent.py --d-model 256 --n-layers 4 --use-category
    python train/train_canonical_latent.py --pca-dim 64 --input-noise 0.05 --neuron-dropout 0.1
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(".")

from config_const import (
    SEED, CHECKPOINT_DIR, CACHE_DIR,
    HVM_N_CAT, HVM_N_STIMULI, HVM_N_VAR,
    HVM_TIME_WINDOW,
    HVM_CANONICAL_LATENTS_PATH,
)
from encoders import TemporalTransformer
from get_device import get_device
from train.train_latent import _load_hvm_neural, _hvm_stratified_split, prepare_input


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def info_nce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style)."""
    pred_n = F.normalize(pred, dim=-1)
    tgt_n  = F.normalize(target, dim=-1)
    logits = (pred_n @ tgt_n.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_canonical_latents(canonical_lat: int, image_size: int) -> torch.Tensor:
    specific = CACHE_DIR / f"hvm10_canonical_latents_s{image_size}_c{canonical_lat}.pt"
    path = specific if specific.exists() else HVM_CANONICAL_LATENTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Canonical latent cache not found. Run:\n"
            f"  python scripts/build_canonical_latent_cache.py "
            f"--image-size {image_size} --canonical-lat {canonical_lat}"
        )
    blob = torch.load(path, weights_only=True)
    latents = blob['latents'] if isinstance(blob, dict) else blob
    assert latents.shape[0] == HVM_N_STIMULI
    stored_lat = blob.get('canonical_lat', latents.shape[2]) if isinstance(blob, dict) else latents.shape[2]
    if stored_lat != canonical_lat:
        latents = F.interpolate(latents.float(), size=(canonical_lat, canonical_lat),
                                mode='bilinear', align_corners=False)
    return latents.float()


def fit_pca(lat_train_flat: torch.Tensor, pca_dim: int):
    """Fit PCA on (N, D) training latents. Returns (components, mean) on CPU."""
    mean = lat_train_flat.mean(dim=0)          # (D,)
    X    = lat_train_flat - mean               # centered
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    components = Vh[:pca_dim]                  # (K, D)
    var_explained = (torch.linalg.svd(X, full_matrices=False)[1][:pca_dim] ** 2).sum() / \
                    (torch.linalg.svd(X, full_matrices=False)[1] ** 2).sum()
    print(f"PCA: keeping {pca_dim} components, "
          f"variance explained = {var_explained.item():.4f}")
    return components.cpu(), mean.cpu()


def compute_cat_means(latents: torch.Tensor, train_idx: torch.Tensor,
                      cat_indices: torch.Tensor) -> torch.Tensor:
    """Category-mean latents (10, C, S, S) computed from training split only."""
    n_cat = cat_indices.max().item() + 1
    cat_means = torch.zeros(n_cat, *latents.shape[1:])
    cat_train = cat_indices[train_idx]
    for c in range(n_cat):
        mask = cat_train == c
        cat_means[c] = latents[train_idx][mask].mean(0)
    return cat_means


def make_loaders(args):
    rsp = _load_hvm_neural()
    neural = torch.from_numpy(rsp).float()                    # (450, N, T)
    latents = load_canonical_latents(args.canonical_lat, args.image_size)  # (450, 16, S, S)

    lat_shape = tuple(latents.shape[1:])   # (16, S, S)
    out_dim   = int(np.prod(lat_shape))    # 16 * S * S

    cat_indices = torch.arange(HVM_N_STIMULI) // HVM_N_VAR
    train_idx_np, val_idx_np, test_idx_np = _hvm_stratified_split(args.seed)
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx   = torch.from_numpy(val_idx_np).long()
    test_idx  = torch.from_numpy(test_idx_np).long()

    # category-mean residual: subtract per-category mean (train-computed) from all latents
    cat_means = None
    if args.use_residual:
        cat_means = compute_cat_means(latents, train_idx, cat_indices)  # (10, 16, S, S)
        baseline  = cat_means[cat_indices]                               # (450, 16, S, S)
        latents   = latents - baseline
        resid_var = latents.var().item()
        print(f"Residual mode: var={resid_var:.4f} "
              f"({resid_var / (resid_var + cat_means.var().item() * 10) * 100:.1f}% of original)")

    # z-score neural on training split
    neural_train = neural[train_idx]
    neural_mean = neural_train.mean(dim=(0, 2), keepdim=True)
    neural_std  = neural_train.std(dim=(0, 2), keepdim=True).clamp(min=1e-6)

    # per-channel normalization on training split (on residuals if use_residual)
    lat_train = latents[train_idx]
    lat_mean  = lat_train.mean(dim=(0, 2, 3))                   # (16,)
    lat_std   = lat_train.std(dim=(0, 2, 3)).clamp(min=1e-6)    # (16,)

    # PCA on whitened training latents (on residuals if use_residual)
    pca_components = pca_mean = None
    if args.pca_dim > 0:
        lat_train_norm = ((lat_train - lat_mean[None,:,None,None])
                          / lat_std[None,:,None,None]).flatten(1)   # (270, D)
        pca_components, pca_mean = fit_pca(lat_train_norm, args.pca_dim)
        out_dim = args.pca_dim
        print(f"Training in PCA space: out_dim={out_dim} (was {int(np.prod(lat_shape))})")

    def make(idx, shuffle):
        if args.use_category:
            ds = TensorDataset(neural[idx], latents[idx], cat_indices[idx])
        else:
            ds = TensorDataset(neural[idx], latents[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    n_neurons, n_time = rsp.shape[1], rsp.shape[2]
    print(
        f"HVM canonical latent loaders: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
        f"neurons={n_neurons}, time={n_time}, "
        f"lat_shape={lat_shape}, out_dim={out_dim}"
    )
    return (train_loader, val_loader, test_loader,
            n_neurons, n_time, lat_shape, out_dim,
            neural_mean, neural_std, lat_mean, lat_std,
            pca_components, pca_mean, cat_means, cat_indices)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args, n_neurons, n_time, out_dim):
    n_cat = HVM_N_CAT if args.use_category else 0
    return TemporalTransformer(
        n_neurons=n_neurons, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, out_dim=out_dim, dropout=args.dropout,
        n_categories=n_cat, normalize_output=False,
    )


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_lat(lat, lat_mean, lat_std):
    return (lat - lat_mean[None, :, None, None]) / lat_std[None, :, None, None]


def to_pca(flat_norm: torch.Tensor, components: torch.Tensor, pca_mean: torch.Tensor) -> torch.Tensor:
    """Project whitened flat latent (B, D) → PCA space (B, K)."""
    return (flat_norm - pca_mean) @ components.T


def from_pca(pca_codes: torch.Tensor, components: torch.Tensor, pca_mean: torch.Tensor) -> torch.Tensor:
    """Invert PCA codes (B, K) → whitened flat latent (B, D)."""
    return pca_codes @ components + pca_mean


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_name(args):
    cat_tag = "_cat" if args.use_category else ""
    res_tag = "_res" if args.use_residual else ""
    pca_tag = f"_pca{args.pca_dim}" if args.pca_dim > 0 else ""
    in_tag  = f"_in{args.input_noise}" if args.input_noise > 0 else ""
    nd_tag  = f"_nd{args.neuron_dropout}" if args.neuron_dropout > 0 else ""
    nce_tag = f"_nw{args.nce_weight}" if args.nce_weight > 0 else ""
    return (f"d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}"
            f"_c{args.canonical_lat}_do{args.dropout}"
            f"_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}"
            f"{cat_tag}{res_tag}{pca_tag}{in_tag}{nd_tag}{nce_tag}")


def run_dir(args):
    return CHECKPOINT_DIR / "hvm" / "transformer" / "canonical_latent" / run_name(args)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args,
                    lat_shape=None, neural_mean=None, neural_std=None,
                    lat_mean=None, lat_std=None, pca_components=None, pca_mean=None,
                    cat_means=None):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics":         metrics,
        "args":            vars(args),
        "lat_shape":       lat_shape,
        "neural_mean":     neural_mean,
        "neural_std":      neural_std,
        "lat_mean":        lat_mean,
        "lat_std":         lat_std,
        "pca_components":  pca_components,
        "pca_mean":        pca_mean,
        "cat_means":       cat_means,
    }, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    device = get_device()

    (train_loader, val_loader, test_loader,
     n_neurons, n_time, lat_shape, out_dim,
     neural_mean, neural_std, lat_mean, lat_std,
     pca_components, pca_mean, cat_means, cat_indices) = make_loaders(args)

    neural_mean = neural_mean.to(device)
    neural_std  = neural_std.to(device)
    lat_mean    = lat_mean.to(device)
    lat_std     = lat_std.to(device)
    use_pca     = pca_components is not None
    use_res     = cat_means is not None
    if use_pca:
        pca_comp_d = pca_components.to(device)
        pca_mean_d = pca_mean.to(device)
    if use_res:
        cat_means_d = cat_means.to(device)   # (10, 16, S, S)

    model = build_model(args, n_neurons, n_time, out_dim).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, int(args.epochs * args.warmup_frac))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    ckpt_dir = run_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    ckpt_kwargs = dict(
        lat_shape=lat_shape,
        neural_mean=neural_mean.cpu(), neural_std=neural_std.cpu(),
        lat_mean=lat_mean.cpu(),       lat_std=lat_std.cpu(),
        pca_components=pca_components, pca_mean=pca_mean,
        cat_means=cat_means,
    )
    best_val_cos = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            neural = batch[0].to(device)
            lat    = batch[1].to(device).float()
            cat    = batch[2].to(device) if len(batch) == 3 else None

            neural = (neural - neural_mean) / neural_std

            # input noise + neuron dropout
            if args.input_noise > 0.0:
                neural = neural + torch.randn_like(neural) * args.input_noise
            if args.neuron_dropout > 0.0:
                mask = (torch.rand(neural.shape[0], 1, neural.shape[2], device=device)
                        > args.neuron_dropout).float()
                neural = neural * mask

            flat_norm = normalize_lat(lat, lat_mean, lat_std).flatten(1)
            target = to_pca(flat_norm, pca_comp_d, pca_mean_d) if use_pca else flat_norm

            x    = prepare_input(neural, 'transformer')
            pred = model(x, cat)

            loss = F.mse_loss(pred, target)
            if args.nce_weight > 0.0 and pred.size(0) > 1:
                loss = (1.0 - args.nce_weight) * loss + \
                       args.nce_weight * info_nce_loss(pred, target, args.nce_temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                neural = batch[0].to(device)
                lat    = batch[1].to(device).float()
                cat    = batch[2].to(device) if len(batch) == 3 else None
                neural = (neural - neural_mean) / neural_std
                flat_norm = normalize_lat(lat, lat_mean, lat_std).flatten(1)
                target = to_pca(flat_norm, pca_comp_d, pca_mean_d) if use_pca else flat_norm
                x    = prepare_input(neural, 'transformer')
                pred = model(x, cat)
                val_loss += F.mse_loss(pred, target).item()
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        val_loss /= len(val_loader)
        preds_t   = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        val_cos = F.cosine_similarity(preds_t, targets_t, dim=-1).mean().item()

        # report MSE in original whitened space (invert PCA and/or add back category baseline)
        if use_pca:
            preds_full   = from_pca(preds_t,   pca_components, pca_mean)
            targets_full = from_pca(targets_t, pca_components, pca_mean)
        else:
            preds_full, targets_full = preds_t, targets_t
        val_mse_full = F.mse_loss(preds_full, targets_full).item()
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"val_mse={val_mse_full:.6f}  val_cos={val_cos:.4f}")

        metrics = {"epoch": epoch, "train_loss": train_loss,
                   "val_loss": val_loss, "val_mse": val_mse_full, "val_cos_sim": val_cos}
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler,
                        epoch, metrics, args, **ckpt_kwargs)
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler,
                            epoch, metrics, args, **ckpt_kwargs)

    # Test evaluation
    ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            neural = batch[0].to(device)
            lat    = batch[1].to(device).float()
            cat    = batch[2].to(device) if len(batch) == 3 else None
            neural = (neural - neural_mean) / neural_std
            flat_norm = normalize_lat(lat, lat_mean, lat_std).flatten(1)
            target = to_pca(flat_norm, pca_comp_d, pca_mean_d) if use_pca else flat_norm
            x    = prepare_input(neural, 'transformer')
            pred = model(x, cat)
            test_preds.append(pred.cpu())
            test_targets.append(target.cpu())

    tp = torch.cat(test_preds)
    tt = torch.cat(test_targets)
    test_cos = F.cosine_similarity(tp, tt, dim=-1).mean().item()
    if use_pca:
        tp_full = from_pca(tp, pca_components, pca_mean)
        tt_full = from_pca(tt, pca_components, pca_mean)
    else:
        tp_full, tt_full = tp, tt
    test_mse = F.mse_loss(tp_full, tt_full).item()
    print(f"\nTest (N={len(tp)}):  mse={test_mse:.6f}  cos={test_cos:.4f}")

    ckpt["test_metrics"] = {"test_mse": test_mse, "test_cos_sim": test_cos}
    torch.save(ckpt, ckpt_dir / "best.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use-category",    action="store_true", default=False)
    p.add_argument("--canonical-lat",   type=int,   default=16)
    p.add_argument("--image-size",      type=int,   default=512)
    p.add_argument("--d-model",         type=int,   default=128)
    p.add_argument("--n-heads",         type=int,   default=4)
    p.add_argument("--n-layers",        type=int,   default=2)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--epochs",          type=int,   default=200)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight-decay",    type=float, default=1e-2)
    p.add_argument("--warmup-frac",     type=float, default=0.1)
    p.add_argument("--seed",            type=int,   default=SEED)
    p.add_argument("--use-residual",    action="store_true", default=False,
                   help="Predict residual from category-mean latent instead of full latent.")
    p.add_argument("--pca-dim",         type=int,   default=0,
                   help="Project targets to top-K PCA components. 0 = disabled.")
    p.add_argument("--input-noise",     type=float, default=0.0,
                   help="Gaussian noise std added to neural inputs during training.")
    p.add_argument("--neuron-dropout",  type=float, default=0.0,
                   help="Fraction of neurons zeroed per batch during training.")
    p.add_argument("--nce-weight",      type=float, default=0.0,
                   help="Weight of InfoNCE loss mixed with MSE. 0 = MSE only.")
    p.add_argument("--nce-temperature", type=float, default=0.07)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

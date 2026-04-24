"""Train a conditioned latent predictor from HVM IT neural responses.

Architecture (two modules, trained jointly):
  NeuralEmbedder  : TemporalTransformer(n_neurons, T) → embed (embed_dim,)
  LatentRefiner   : MLP([embed; cat_mean_proj]) → predicted canonical latent

The category-mean latent (computed from training stimuli) is projected to
embed_dim and concatenated with the neural embedding before the MLP head.
This lets the model learn nonlinear interactions between the object prior
(cat_mean) and the stimulus-specific neural signal (embed).

Loss = (1 - nce_weight) * MSE + nce_weight * InfoNCE  (in whitened latent space)
Augmentation: input_noise on neural rates, neuron_dropout per batch.

Inference:
  1. Encode neural activity → embed
  2. Look up cat_mean for the known category
  3. Project cat_mean, concat with embed, run MLP → predicted latent
  4. Resize to bbox in latent space and paste into full init latent

Usage:
    python train/train_latent_conditioned.py
    python train/train_latent_conditioned.py --embed-dim 128 --refiner-hidden 512 --n-layers 2
"""

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(".")

from config_const import (
    SEED, CHECKPOINT_DIR, CACHE_DIR,
    HVM_N_CAT, HVM_N_STIMULI, HVM_N_VAR,
    HVM_CANONICAL_LATENTS_PATH,
)
from encoders import TemporalTransformer
from get_device import get_device
from train.train_latent import _load_hvm_neural, _hvm_stratified_split, prepare_input
from train.train_canonical_latent import (
    load_canonical_latents, normalize_lat, compute_cat_means, info_nce_loss,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LatentRefiner(nn.Module):
    """MLP that maps [neural_embed; cat_mean_proj] → flat latent.

    cat_mean_flat (D,) is projected to embed_dim, then concatenated with the
    neural embedding. Three linear layers with GELU activations predict the
    whitened flat latent.
    """

    def __init__(self, embed_dim: int, lat_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cat_proj = nn.Linear(lat_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lat_dim),
        )

    def forward(self, embed: torch.Tensor, cat_mean_flat: torch.Tensor) -> torch.Tensor:
        cat_feat = self.cat_proj(cat_mean_flat)              # (B, embed_dim)
        fused    = torch.cat([embed, cat_feat], dim=-1)      # (B, 2*embed_dim)
        return self.mlp(fused)                               # (B, lat_dim)


class ConditionedLatentModel(nn.Module):
    def __init__(self, n_neurons: int, n_time: int, embed_dim: int, lat_dim: int,
                 hidden_dim: int, n_heads: int, n_layers: int, dropout: float,
                 n_categories: int):
        super().__init__()
        self.embedder = TemporalTransformer(
            n_neurons=n_neurons, d_model=embed_dim, n_heads=n_heads,
            n_layers=n_layers, out_dim=embed_dim, dropout=dropout,
            n_categories=n_categories, normalize_output=False,
        )
        self.refiner = LatentRefiner(embed_dim, lat_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor, cat_mean_flat: torch.Tensor,
                cat: torch.Tensor | None = None) -> torch.Tensor:
        embed = self.embedder(x, cat)           # (B, embed_dim)
        return self.refiner(embed, cat_mean_flat)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_loaders(args):
    rsp     = _load_hvm_neural()
    neural  = torch.from_numpy(rsp).float()                    # (450, N, T)
    latents = load_canonical_latents(args.canonical_lat, args.image_size)  # (450, 16, S, S)

    lat_shape = tuple(latents.shape[1:])
    lat_dim   = int(np.prod(lat_shape))

    cat_indices = torch.arange(HVM_N_STIMULI) // HVM_N_VAR
    train_idx_np, val_idx_np, test_idx_np = _hvm_stratified_split(args.seed)
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx   = torch.from_numpy(val_idx_np).long()
    test_idx  = torch.from_numpy(test_idx_np).long()

    # category means from training split only
    cat_means = compute_cat_means(latents, train_idx, cat_indices)  # (10, 16, S, S)

    # z-score neural on training split
    neural_train = neural[train_idx]
    neural_mean  = neural_train.mean(dim=(0, 2), keepdim=True)
    neural_std   = neural_train.std(dim=(0, 2), keepdim=True).clamp(min=1e-6)

    # per-channel latent normalization on training split
    lat_train = latents[train_idx]
    lat_mean  = lat_train.mean(dim=(0, 2, 3))
    lat_std   = lat_train.std(dim=(0, 2, 3)).clamp(min=1e-6)

    # cat_mean_flat per stimulus (whitened), stored in dataset for convenience
    cat_mean_per_stim = cat_means[cat_indices]   # (450, 16, S, S)
    cat_mean_norm = normalize_lat(cat_mean_per_stim, lat_mean, lat_std).flatten(1)  # (450, D)

    def make(idx, shuffle):
        ds = TensorDataset(neural[idx], latents[idx], cat_indices[idx], cat_mean_norm[idx])
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    n_neurons, n_time = rsp.shape[1], rsp.shape[2]
    print(
        f"Conditioned latent loaders: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
        f"neurons={n_neurons}, time={n_time}, lat_shape={lat_shape}, lat_dim={lat_dim}"
    )
    return (train_loader, val_loader, test_loader,
            n_neurons, n_time, lat_shape, lat_dim,
            neural_mean, neural_std, lat_mean, lat_std, cat_means)


# ---------------------------------------------------------------------------
# Naming / checkpointing
# ---------------------------------------------------------------------------

def run_name(args):
    return (
        f"ed{args.embed_dim}_nh{args.n_heads}_nl{args.n_layers}"
        f"_rh{args.refiner_hidden}_c{args.canonical_lat}_do{args.dropout}"
        f"_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}"
        f"_in{args.input_noise}_nd{args.neuron_dropout}_nw{args.nce_weight}_cat"
    )


def run_dir(args):
    return CHECKPOINT_DIR / "hvm" / "transformer" / "latent_conditioned" / run_name(args)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args,
                    lat_shape, neural_mean, neural_std, lat_mean, lat_std, cat_means):
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
        "cat_means":       cat_means,
    }, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    device = get_device()

    (train_loader, val_loader, test_loader,
     n_neurons, n_time, lat_shape, lat_dim,
     neural_mean, neural_std, lat_mean, lat_std, cat_means) = make_loaders(args)

    neural_mean = neural_mean.to(device)
    neural_std  = neural_std.to(device)
    lat_mean    = lat_mean.to(device)
    lat_std     = lat_std.to(device)

    n_cat = HVM_N_CAT  # always category-conditioned
    model = ConditionedLatentModel(
        n_neurons=n_neurons, n_time=n_time,
        embed_dim=args.embed_dim, lat_dim=lat_dim,
        hidden_dim=args.refiner_hidden,
        n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, n_categories=n_cat,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer     = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        cat_means=cat_means,
    )
    best_val_cos = -float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            neural       = batch[0].to(device)
            lat          = batch[1].to(device).float()
            cat          = batch[2].to(device)
            cat_mean_flat = batch[3].to(device)   # already whitened

            neural = (neural - neural_mean) / neural_std
            if args.input_noise > 0.0:
                neural = neural + torch.randn_like(neural) * args.input_noise
            if args.neuron_dropout > 0.0:
                mask   = (torch.rand(neural.shape[0], 1, neural.shape[2], device=device)
                          > args.neuron_dropout).float()
                neural = neural * mask

            target = normalize_lat(lat, lat_mean, lat_std).flatten(1)
            x      = prepare_input(neural, 'transformer')
            pred   = model(x, cat_mean_flat, cat)

            loss = F.mse_loss(pred, target)
            if args.nce_weight > 0.0 and pred.size(0) > 1:
                loss = ((1.0 - args.nce_weight) * loss
                        + args.nce_weight * info_nce_loss(pred, target, args.nce_temperature))

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
                neural        = batch[0].to(device)
                lat           = batch[1].to(device).float()
                cat           = batch[2].to(device)
                cat_mean_flat = batch[3].to(device)
                neural = (neural - neural_mean) / neural_std
                target = normalize_lat(lat, lat_mean, lat_std).flatten(1)
                x      = prepare_input(neural, 'transformer')
                pred   = model(x, cat_mean_flat, cat)
                val_loss += F.mse_loss(pred, target).item()
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        val_loss  /= len(val_loader)
        preds_t    = torch.cat(all_preds)
        targets_t  = torch.cat(all_targets)
        val_cos    = F.cosine_similarity(preds_t, targets_t, dim=-1).mean().item()

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  val_mse={val_loss:.6f}  val_cos={val_cos:.4f}")

        metrics = {"epoch": epoch, "train_loss": train_loss,
                   "val_mse": val_loss, "val_cos_sim": val_cos}
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
            neural        = batch[0].to(device)
            lat           = batch[1].to(device).float()
            cat           = batch[2].to(device)
            cat_mean_flat = batch[3].to(device)
            neural = (neural - neural_mean) / neural_std
            target = normalize_lat(lat, lat_mean, lat_std).flatten(1)
            x      = prepare_input(neural, 'transformer')
            pred   = model(x, cat_mean_flat, cat)
            test_preds.append(pred.cpu())
            test_targets.append(target.cpu())

    tp       = torch.cat(test_preds)
    tt       = torch.cat(test_targets)
    test_mse = F.mse_loss(tp, tt).item()
    test_cos = F.cosine_similarity(tp, tt, dim=-1).mean().item()
    print(f"\nTest (N={len(tp)}):  mse={test_mse:.6f}  cos={test_cos:.4f}")

    ckpt["test_metrics"] = {"test_mse": test_mse, "test_cos_sim": test_cos}
    torch.save(ckpt, ckpt_dir / "best.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-lat",   type=int,   default=16)
    p.add_argument("--image-size",      type=int,   default=512)
    p.add_argument("--embed-dim",       type=int,   default=128,
                   help="Neural encoder output dim and transformer width.")
    p.add_argument("--n-heads",         type=int,   default=4)
    p.add_argument("--n-layers",        type=int,   default=2)
    p.add_argument("--refiner-hidden",  type=int,   default=512,
                   help="Hidden dim of the LatentRefiner MLP.")
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--epochs",          type=int,   default=200)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-2)
    p.add_argument("--warmup-frac",     type=float, default=0.1)
    p.add_argument("--seed",            type=int,   default=SEED)
    p.add_argument("--input-noise",     type=float, default=0.05)
    p.add_argument("--neuron-dropout",  type=float, default=0.1)
    p.add_argument("--nce-weight",      type=float, default=0.1)
    p.add_argument("--nce-temperature", type=float, default=0.07)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

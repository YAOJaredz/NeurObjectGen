"""Train a neural encoder (MLP, LSTM, or Transformer) to map IT responses -> FLUX VAE latents.

The target is the spatial VAE latent (16, 28, 28) flattened to 12544-d. Loss is MSE.
Validation reports MSE and cosine similarity on the flat latent vectors.

Usage:
    python train/train_latent.py --model transformer --dataset rust
    python train/train_latent.py --model transformer --dataset hvm --use-category
    python train/train_latent.py --model mlp --bottleneck 512 --dataset hvm
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import (
    SEED,
    CHECKPOINT_DIR,
    HVM_N_CAT, HVM_N_STIMULI, HVM_N_VAL, HVM_N_VAR,
    N_STIMULI, N_TRAIN, N_VAL,
    RUST_TIME_WINDOW, HVM_TIME_WINDOW,
    RUST_VAE_LATENTS_PATH, HVM_VAE_LATENTS_PATH,
)
from encoders import BottleneckMLP, TemporalLSTM, TemporalTransformer
from get_device import get_device

LATENT_SHAPE = (16, 28, 28)
OUT_DIM = 16 * 28 * 28  # 12544

MLP_TIME_SLICE = slice(5, 20)
MLP_N_BINS = MLP_TIME_SLICE.stop - MLP_TIME_SLICE.start


# ---------------------------------------------------------------------------
# Neural data loading (inlined to avoid modifying existing loaders)
# ---------------------------------------------------------------------------

def _load_rust_neural() -> np.ndarray:
    from HexPred.object_response.get_rust_response import get_rust_responses
    from HexPred.constants import ALL_MONKEYS

    responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_rust_responses(mode='area', area='all', monkey=monkey,
                                    time_window=RUST_TIME_WINDOW)
        responses.append(rsp)
    rsp = np.concatenate(responses, axis=1)
    dead = (
        np.all((rsp == 0) | np.isnan(rsp), axis=(0, 2))
        | np.any(np.isnan(rsp), axis=(0, 2))
    )
    rsp = rsp[:, ~dead, :]
    assert not np.isnan(rsp).any()
    assert rsp.shape[0] == N_STIMULI
    return rsp


def _load_hvm_neural() -> np.ndarray:
    from HexPred.object_response.get_hvm_response import get_hvm_responses
    from HexPred.constants import ALL_MONKEYS

    responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_hvm_responses(mode='area', monkey=monkey, area='all',
                                   time_window=HVM_TIME_WINDOW)
        responses.append(rsp)
    rsp = np.concatenate(responses, axis=1)
    dead = (
        np.all((rsp == 0) | np.isnan(rsp), axis=(0, 2))
        | np.any(np.isnan(rsp), axis=(0, 2))
    )
    rsp = rsp[:, ~dead, :]
    assert not np.isnan(rsp).any()
    assert rsp.shape[0] == HVM_N_STIMULI
    return rsp


def _hvm_stratified_split(seed: int):
    from HexPred.object_response.get_hvm_response import get_hvm_category_vector
    cats = get_hvm_category_vector()
    unique_cats = np.unique(cats)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    n_val = n_test = 9
    for cat in unique_cats:
        idx = np.where(cats == cat)[0]
        perm = rng.permutation(idx)
        train_idx.append(perm[n_val + n_test:])
        val_idx.append(perm[:n_val])
        test_idx.append(perm[n_val:n_val + n_test])
    return (np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx))


def make_loaders(args) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """Return (train, val, test, n_neurons, n_time)."""
    if args.dataset == "rust":
        if not RUST_VAE_LATENTS_PATH.exists():
            raise FileNotFoundError(
                f"RUST VAE latent cache not found. Run: python scripts/cache_vae_latents.py --dataset rust"
            )
        rsp = _load_rust_neural()
        neural = torch.from_numpy(rsp).float()
        latents = torch.load(RUST_VAE_LATENTS_PATH, weights_only=True).flatten(1)  # (300, 12544)

        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(N_STIMULI)
        train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
        val_idx   = torch.from_numpy(perm[N_TRAIN:N_TRAIN + N_VAL]).long()
        test_idx  = torch.from_numpy(perm[N_TRAIN + N_VAL:]).long()

        def make(idx, shuffle):
            return DataLoader(
                TensorDataset(neural[idx], latents[idx]),
                batch_size=args.batch_size, shuffle=shuffle,
            )

    else:  # hvm
        if not HVM_VAE_LATENTS_PATH.exists():
            raise FileNotFoundError(
                f"HVM VAE latent cache not found. Run: python scripts/cache_vae_latents.py --dataset hvm"
            )
        rsp = _load_hvm_neural()
        neural = torch.from_numpy(rsp).float()
        latents = torch.load(HVM_VAE_LATENTS_PATH, weights_only=True).flatten(1)  # (450, 12544)
        cat_indices = torch.arange(HVM_N_STIMULI) // HVM_N_VAR  # (450,) int64

        train_idx_np, val_idx_np, test_idx_np = _hvm_stratified_split(args.seed)
        train_idx = torch.from_numpy(train_idx_np).long()
        val_idx   = torch.from_numpy(val_idx_np).long()
        test_idx  = torch.from_numpy(test_idx_np).long()

        def make(idx, shuffle):
            if args.use_category:
                ds = TensorDataset(neural[idx], latents[idx], cat_indices[idx])
            else:
                ds = TensorDataset(neural[idx], latents[idx])
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    n_neurons = rsp.shape[1]
    n_time    = rsp.shape[2]
    print(
        f"{args.dataset.upper()} loaders: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
        f"neurons={n_neurons}, time={n_time}, target=vae_latent{LATENT_SHAPE}"
    )
    return train_loader, val_loader, test_loader, n_neurons, n_time


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args, n_neurons: int, n_time: int, n_categories: int = 0):
    if args.model == "mlp":
        in_dim = n_neurons * MLP_N_BINS
        return BottleneckMLP(in_dim=in_dim, bottleneck=args.bottleneck, out_dim=OUT_DIM,
                             dropout=args.dropout, n_categories=n_categories)
    elif args.model == "lstm":
        return TemporalLSTM(n_neurons=n_neurons, hidden=args.hidden, out_dim=OUT_DIM,
                            n_layers=args.n_layers, dropout=args.dropout, n_categories=n_categories)
    else:
        return TemporalTransformer(
            n_neurons=n_neurons, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, out_dim=OUT_DIM, dropout=args.dropout,
            n_categories=n_categories,
        )


def prepare_input(neural: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name == "mlp":
        return neural[:, :, MLP_TIME_SLICE].flatten(1)
    return neural.permute(0, 2, 1)  # (B, T, N)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_name(args) -> str:
    cat_tag = "_cat" if getattr(args, "use_category", False) else ""
    shared = f"do{args.dropout}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}_uw{args.uniformity_weight}"
    if args.model == "mlp":
        return f"bn{args.bottleneck}_{shared}{cat_tag}"
    elif args.model == "lstm":
        return f"h{args.hidden}_nl{args.n_layers}_{shared}{cat_tag}"
    else:
        return f"d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}_{shared}{cat_tag}"


def run_dir(args) -> Path:
    return CHECKPOINT_DIR / args.dataset / args.model / "vae_latent" / run_name(args)


def save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, args):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }, path)


def load_checkpoint(path, model):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    device = get_device()

    train_loader, val_loader, test_loader, n_neurons, n_time = make_loaders(args)

    n_cat = HVM_N_CAT if (args.dataset == "hvm" and args.use_category) else 0
    model = build_model(args, n_neurons, n_time, n_categories=n_cat).to(device)
    print(f"Model: {model.__class__.__name__} | params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, int(args.epochs * args.warmup_frac)) if args.warmup_frac > 0 else 0

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    ckpt_dir = run_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            neural  = batch[0].to(device)
            target  = batch[1].to(device)
            cat     = batch[2].to(device) if len(batch) == 3 else None

            x    = prepare_input(neural, args.model)
            pred = model(x, cat)

            loss = F.mse_loss(pred, target)

            if args.uniformity_weight > 0.0:
                pred_norm = F.normalize(pred, dim=-1)
                loss = loss + args.uniformity_weight * uniformity_loss(pred_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_mse = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                neural = batch[0].to(device)
                target = batch[1].to(device)
                cat    = batch[2].to(device) if len(batch) == 3 else None
                x      = prepare_input(neural, args.model)
                pred   = model(x, cat)
                val_mse += F.mse_loss(pred, target).item()
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        val_mse /= len(val_loader)
        all_preds_t   = torch.cat(all_preds)
        all_targets_t = torch.cat(all_targets)
        cos_sim = F.cosine_similarity(all_preds_t, all_targets_t, dim=-1).mean().item()

        print(f"epoch {epoch:3d}/{args.epochs}  train_mse={train_loss:.6f}  val_mse={val_mse:.6f}  cos={cos_sim:.4f}")

        metrics = {"epoch": epoch, "train_mse": train_loss, "val_mse": val_mse, "val_cos_sim": cos_sim}
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args)

    load_checkpoint(ckpt_dir / "best.pt", model)

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            neural = batch[0].to(device)
            cat    = batch[2].to(device) if len(batch) == 3 else None
            x      = prepare_input(neural, args.model)
            pred   = model(x, cat)
            test_preds.append(pred.cpu())
            test_targets.append(batch[1])

    test_preds_t   = torch.cat(test_preds)
    test_targets_t = torch.cat(test_targets)
    test_mse = F.mse_loss(test_preds_t, test_targets_t).item()
    test_cos = F.cosine_similarity(test_preds_t, test_targets_t, dim=-1).mean().item()

    print(f"\nTest set (N={len(test_preds_t)}):  mse={test_mse:.6f}  cos={test_cos:.4f}")

    test_metrics = {"test_mse": test_mse, "test_cos_sim": test_cos}
    best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    best_ckpt["test_metrics"] = test_metrics
    torch.save(best_ckpt, ckpt_dir / "best.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "lstm", "transformer"], default="transformer")
    p.add_argument("--dataset", choices=["rust", "hvm"], default="rust")
    p.add_argument("--use-category", action="store_true", default=False,
                   help="Condition on category label (HVM only)")
    p.add_argument("--seed", type=int, default=SEED)

    # MLP
    p.add_argument("--bottleneck", type=int, default=256)
    # LSTM
    p.add_argument("--hidden", type=int, default=128)
    # Transformer
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=1)

    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--uniformity-weight", type=float, default=0.0,
                   help="Weight of uniformity loss on predicted embeddings (applied after L2-norm). "
                        "Set >0 to spread predictions across the latent space.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

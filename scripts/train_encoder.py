"""Train a neural encoder (MLP, LSTM, or Transformer) to map IT responses -> SigLIP.

Loss: InfoNCE (NT-Xent) contrastive loss aligning predicted embeddings with
      frozen SigLIP image embeddings of the presented stimuli.
Regularisation: L2 weight decay via AdamW (set with --weight-decay).

Usage examples:
    python scripts/train_encoder.py --model mlp --bottleneck 256 --dropout 0.1
    python scripts/train_encoder.py --model lstm --hidden 128 --dropout 0.1
    python scripts/train_encoder.py --model transformer --d-model 128 --n-heads 4 --n-layers 1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# make repo root importable when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import SEED, SIGLIP_DIM, CHECKPOINT_DIR
from data_utils.rust_loader import make_rust_loader
from encoders import BottleneckMLP, TemporalLSTM, TemporalTransformer
from eval.metrics import two_afc_identification
from get_device import get_device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_name(args) -> str:
    """Deterministic directory name encoding all hyperparameters for this run.

    Structure: checkpoints/<model>/<run_name>/
    """
    shared = f"do{args.dropout}_lr{args.lr}_wd{args.weight_decay}_tn{args.target_noise}_bs{args.batch_size}_t{args.temperature}"
    if args.model == "mlp":
        return f"bn{args.bottleneck}_{shared}"
    elif args.model == "lstm":
        return f"h{args.hidden}_{shared}"
    else:
        return f"d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}_{shared}"


def run_dir(args) -> Path:
    """checkpoints/<model>/<run_name>/"""
    return CHECKPOINT_DIR / args.model / run_name(args)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, args):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["metrics"], ckpt["args"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def infonce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE (NT-Xent) between predicted and target embeddings.

    Both pred and target are assumed L2-normalised (shape B x D).
    Each sample is its own positive; all others in the batch are negatives.
    """
    logits = pred @ target.T / temperature  # (B, B)
    labels = torch.arange(len(pred), device=pred.device)
    loss_p = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_p + loss_t) / 2


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args, n_neurons: int, n_time: int) -> torch.nn.Module:
    if args.model == "mlp":
        in_dim = n_neurons * n_time
        return BottleneckMLP(in_dim=in_dim, bottleneck=args.bottleneck, out_dim=SIGLIP_DIM, dropout=args.dropout)
    elif args.model == "lstm":
        return TemporalLSTM(n_neurons=n_neurons, hidden=args.hidden, out_dim=SIGLIP_DIM, dropout=args.dropout)
    elif args.model == "transformer":
        return TemporalTransformer(
            n_neurons=n_neurons, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, out_dim=SIGLIP_DIM, dropout=args.dropout,
        )
    else:
        raise ValueError(f"unknown model: {args.model}")


def prepare_input(neural: torch.Tensor, model_name: str) -> torch.Tensor:
    """neural: (B, N_neurons, T) -> model-specific shape."""
    if model_name == "mlp":
        return neural.flatten(1)        # (B, N*T)
    else:
        return neural.permute(0, 2, 1)  # (B, T, N) for LSTM / Transformer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_with_embeddings(args):
    torch.manual_seed(SEED)
    device = get_device()

    train_loader, val_loader, test_loader = make_rust_loader(
        batch_size=args.batch_size, use_embeddings=True,
    )

    sample_neural, _ = next(iter(train_loader))
    _, n_neurons, n_time = sample_neural.shape

    # --- sanity checks before training ---
    # 1. NaN in neural data
    all_neural = sample_neural
    if torch.isnan(all_neural).any():
        raise ValueError("NaN values detected in neural input — check dead-neuron filtering in rust_loader.")

    # 2. batch too small for InfoNCE
    if args.batch_size < 2:
        raise ValueError("batch_size must be >= 2 for InfoNCE loss.")

    model = build_model(args, n_neurons, n_time).to(device)
    print(f"Model: {model.__class__.__name__} | params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = run_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for neural, siglip in train_loader:
            neural  = neural.to(device)
            siglip  = siglip.to(device)

            x    = prepare_input(neural, args.model)
            pred = model(x)
            if args.target_noise > 0.0:
                siglip = F.normalize(siglip + torch.randn_like(siglip) * args.target_noise, dim=-1)
            loss = infonce_loss(pred, siglip, temperature=args.temperature)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch {epoch}. "
                    "Likely causes: NaN in inputs, embeddings not L2-normalised, "
                    "or temperature too low. Try --temperature 0.1 or higher."
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- val ---
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for neural, siglip in val_loader:
                neural = neural.to(device)
                siglip = siglip.to(device)
                x      = prepare_input(neural, args.model)
                pred   = model(x)
                val_loss += infonce_loss(pred, siglip, temperature=args.temperature).item()
                all_preds.append(pred.cpu())
                all_targets.append(siglip.cpu())

        val_loss /= len(val_loader)
        afc = two_afc_identification(torch.cat(all_preds), torch.cat(all_targets))

        print(f"epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  2AFC={afc:.3f}")

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_2afc": afc}

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args)

    # restore best weights for the returned model
    load_checkpoint(ckpt_dir / "best.pt", model)

    return model, test_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "lstm", "transformer"], default="mlp")

    # MLP
    p.add_argument("--bottleneck", type=int, default=256)
    # LSTM
    p.add_argument("--hidden", type=int, default=128)

    # Transformer
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=1)

    # shared
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--target-noise", type=float, default=0.0,
                   help="Std of Gaussian noise added to SigLIP targets during training (re-normalised after)")

    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="L2 regularisation via AdamW weight decay")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, test_loader = train_with_embeddings(args)

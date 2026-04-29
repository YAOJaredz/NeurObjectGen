"""Train a neural encoder (MLP, LSTM, or Transformer) to map IT responses -> SigLIP.

Loss (--loss):
  infonce (default): Symmetric InfoNCE (NT-Xent) contrastive loss. Negatives are
      augmented with a full-dataset memory bank so every batch sees all training
      embeddings as negatives, not just intra-batch pairs.
  cosine: Mean cosine distance (1 - cos_sim) + uniformity regularisation to
      prevent collapse. Directly optimises embedding proximity, which better
      matches the downstream image-generation objective.
Regularisation:
  - L2 weight decay via AdamW (--weight-decay).
  - Uniformity loss on predicted embeddings (--uniformity-weight) to prevent
    collapse onto a low-dimensional submanifold of the hypersphere.

Usage examples:
    python scripts/train_encoder.py --model mlp --bottleneck 256 --dropout 0.1
    python scripts/train_encoder.py --model lstm --hidden 128 --dropout 0.1
    python scripts/train_encoder.py --model transformer --d-model 128 --n-heads 4 --n-layers 1
    python scripts/train_encoder.py --model transformer --loss cosine --uniformity-weight 0.1
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('.')

from config_const import SEED, SIGLIP_DIM, CLIP_DIM, CHECKPOINT_DIR, HVM_N_CAT

TARGET_DIMS = {"siglip": SIGLIP_DIM, "clip": CLIP_DIM}
from data_utils.rust_loader import make_rust_loader
from data_utils.hvm_loader import make_hvm_loader
from encoders import BottleneckMLP, TemporalLSTM, TemporalTransformer
from eval.metrics import two_afc_identification, retrieval_accuracy
from train.losses import cosine_loss, info_nce_loss_with_bank, uniformity_loss
from get_device import get_device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_name(args) -> str:
    """Deterministic directory name encoding all hyperparameters for this run.

    Structure: checkpoints/<model>/<run_name>/
    """
    loss_tag = f"loss{args.loss}_" if args.loss != "cosine" else ""
    cat_tag  = "_cat" if getattr(args, "use_category", False) else ""
    shared = f"{loss_tag}do{args.dropout}_lr{args.lr}_wd{args.weight_decay}_tn{args.target_noise}_bs{args.batch_size}_t{args.temperature}_uw{args.uniformity_weight}"
    if args.model == "mlp":
        return f"bn{args.bottleneck}_{shared}{cat_tag}"
    elif args.model == "lstm":
        return f"h{args.hidden}_nl{args.n_layers}_{shared}{cat_tag}"
    else:
        return f"d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}_{shared}{cat_tag}"


def run_dir(args) -> Path:
    """checkpoints/<dataset>/<model>/<target>/<run_name>/"""
    dataset = getattr(args, "dataset", "rust")
    if dataset == "hvm":
        return CHECKPOINT_DIR / "hvm" / args.model / args.target / run_name(args)
    return CHECKPOINT_DIR / args.model / args.target / run_name(args)


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
# Model factory
# ---------------------------------------------------------------------------

MLP_TIME_SLICE = slice(5, 20)  # 50-200 ms window, 15 bins
MLP_N_BINS = MLP_TIME_SLICE.stop - MLP_TIME_SLICE.start  # 15


def build_model(args, n_neurons: int, n_time: int, out_dim: int, n_categories: int = 0) -> torch.nn.Module:
    if args.model == "mlp":
        in_dim = n_neurons * MLP_N_BINS  # 15 bins, not full n_time
        return BottleneckMLP(in_dim=in_dim, bottleneck=args.bottleneck, out_dim=out_dim, dropout=args.dropout, n_categories=n_categories)
    elif args.model == "lstm":
        return TemporalLSTM(n_neurons=n_neurons, hidden=args.hidden, out_dim=out_dim, n_layers=args.n_layers, dropout=args.dropout, n_categories=n_categories)
    elif args.model == "transformer":
        return TemporalTransformer(
            n_neurons=n_neurons, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, out_dim=out_dim, dropout=args.dropout,
            n_categories=n_categories,
        )
    else:
        raise ValueError(f"unknown model: {args.model}")


def prepare_input(neural: torch.Tensor, model_name: str) -> torch.Tensor:
    """neural: (B, N_neurons, T) -> model-specific shape.

    For MLP, only the 50-200 ms window (bins 5:20) is used to reduce the
    input dimensionality from N*25 to N*15.
    """
    if model_name == "mlp":
        return neural[:, :, MLP_TIME_SLICE].flatten(1)  # (B, N*15)
    else:
        return neural.permute(0, 2, 1)        # (B, T, N) for LSTM / Transformer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_with_embeddings(args):
    torch.manual_seed(SEED)
    device = get_device()

    out_dim = TARGET_DIMS[args.target]

    if args.dataset == "hvm":
        train_loader, val_loader, test_loader = make_hvm_loader(
            batch_size=args.batch_size, use_embeddings=True, target=args.target,
            use_categories=args.use_category,
        )
    else:
        train_loader, val_loader, test_loader = make_rust_loader(
            batch_size=args.batch_size, use_embeddings=True, target=args.target,
        )

    sample_batch = next(iter(train_loader))
    sample_neural = sample_batch[0]
    _, n_neurons, n_time = sample_neural.shape

    # --- sanity checks before training ---
    # 1. NaN in neural data
    all_neural = sample_neural
    if torch.isnan(all_neural).any():
        raise ValueError("NaN values detected in neural input — check dead-neuron filtering in rust_loader.")

    # 2. batch too small for InfoNCE
    if args.loss == "infonce" and args.batch_size < 2:
        raise ValueError("batch_size must be >= 2 for InfoNCE loss.")

    n_cat = HVM_N_CAT if (args.dataset == "hvm" and args.use_category) else 0
    model = build_model(args, n_neurons, n_time, out_dim, n_categories=n_cat).to(device)
    print(f"Model: {model.__class__.__name__} | params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Linear warmup for the first `warmup_epochs` epochs, then cosine decay to 0.
    warmup_epochs = max(1, int(args.epochs * args.warmup_frac)) if args.warmup_frac > 0 else 0

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear ramp from ~0 -> 1 across warmup_epochs
            return (epoch + 1) / warmup_epochs
        # Cosine decay from 1 -> 0 over the remaining epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    ckpt_dir = run_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    # --- Build static memory bank from all training target embeddings ---
    # Collect all training targets once (they're frozen SigLIP embeds, no grad needed).
    all_train_targets = torch.cat([batch[1] for batch in train_loader], dim=0).to(device)  # (N_train, D)

    best_val_2afc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            neural  = batch[0].to(device)
            siglip  = batch[1].to(device)
            cat     = batch[2].to(device) if len(batch) == 3 else None

            x    = prepare_input(neural, args.model)
            pred = model(x, cat)

            if args.target_noise > 0.0:
                siglip = F.normalize(siglip + torch.randn_like(siglip) * args.target_noise, dim=-1)

            if args.loss == "cosine":
                loss = cosine_loss(pred, siglip)
            else:
                bank = all_train_targets.detach()
                loss = info_nce_loss_with_bank(pred, siglip, temperature=args.temperature, memory_bank=bank)

            if args.uniformity_weight > 0.0:
                loss = loss + args.uniformity_weight * uniformity_loss(pred)

            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch {epoch}. "
                    "Likely causes: NaN in inputs, embeddings not L2-normalised, "
                    "or temperature too low. Try --temp-max 0.1 or higher."
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
            for batch in val_loader:
                neural = batch[0].to(device)
                siglip = batch[1].to(device)
                cat    = batch[2].to(device) if len(batch) == 3 else None
                x      = prepare_input(neural, args.model)
                pred   = model(x, cat)
                if args.loss == "cosine":
                    val_loss += cosine_loss(pred, siglip).item()
                else:
                    val_loss += info_nce_loss_with_bank(pred, siglip, temperature=args.temperature).item()
                all_preds.append(pred.cpu())
                all_targets.append(siglip.cpu())

        val_loss /= len(val_loader)
        all_preds_t  = torch.cat(all_preds)
        all_targets_t = torch.cat(all_targets)
        afc     = two_afc_identification(all_preds_t, all_targets_t)
        cos_sim = F.cosine_similarity(all_preds_t, all_targets_t, dim=-1).mean().item()

        print(f"epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  2AFC={afc:.3f}  cos={cos_sim:.4f}")

        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_2afc": afc, "val_cos_sim": cos_sim}

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args)
        if afc > best_val_2afc:
            best_val_2afc = afc
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args)

    # restore best weights for the returned model
    load_checkpoint(ckpt_dir / "best.pt", model)

    # --- test-set retrieval evaluation ---
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

    test_2afc = two_afc_identification(test_preds_t, test_targets_t)
    test_topk = retrieval_accuracy(test_preds_t, test_targets_t, k=[1, 5, 10])
    test_cos  = F.cosine_similarity(test_preds_t, test_targets_t, dim=-1).mean().item()

    print(
        f"\nTest set (N={len(test_preds_t)}):"
        f"  2AFC={test_2afc:.3f}"
        f"  top-1={test_topk[1]:.3f}"
        f"  top-5={test_topk[5]:.3f}"
        f"  top-10={test_topk[10]:.3f}"
        f"  cos={test_cos:.4f}"
    )

    test_metrics = {
        "test_2afc": test_2afc,
        "test_top1": test_topk[1],
        "test_top5": test_topk[5],
        "test_top10": test_topk[10],
        "test_cos_sim": test_cos,
    }
    best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    best_ckpt["test_metrics"] = test_metrics
    torch.save(best_ckpt, ckpt_dir / "best.pt")

    return model, test_loader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "lstm", "transformer"], default="mlp")
    p.add_argument("--dataset", choices=["rust", "hvm"], default="rust",
                   help="Neural dataset to train on")
    p.add_argument("--use-category", action="store_true", default=False,
                   help="Condition on category label (HVM only; adds learned Embedding(10, dim) at bottleneck)")
    p.add_argument("--target", choices=["siglip", "clip"], default="siglip",
                   help="Embedding space to predict: siglip (1152-d) or clip (768-d, detailed captions)")
    p.add_argument("--loss", choices=["infonce", "cosine"], default="cosine",
                   help="infonce: contrastive (default); cosine: cosine distance + uniformity, "
                        "optimises direct embedding proximity for generation")

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
    p.add_argument("--target-noise", type=float, default=0.02,
                   help="Std of Gaussian noise added to SigLIP targets during training (re-normalised after)")

    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="L2 regularisation via AdamW weight decay")
    p.add_argument("--warmup-frac", type=float, default=0.1,
                   help="Fraction of epochs used for linear LR warmup (0 to disable)")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature")
    p.add_argument("--uniformity-weight", type=float, default=0.1,
                   help="Weight of uniformity loss on predicted embeddings (0 to disable)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, test_loader = train_with_embeddings(args)

"""Train a TemporalTransformer to predict T5-XXL PCA coordinates from IT neural responses.

Loss: InfoNCE + cosine mixture (head_loss), same as each head in train_multihead.py.
      Optional uniformity regularisation on predicted embeddings.

Checkpoint criterion: best val cosine similarity on T5 PCA targets.

Datasets:
  --dataset rust (default): 300 RUST stimuli, random 200/50/50 split.
  --dataset hvm:            450 HVM stimuli, category-stratified 270/90/90 split.
                            Use --use-category to add learned category conditioning.

Modes:
  default (--per-token not set):
      One datapoint per stimulus. Target = mean-pooled T5 PCA vector.
  --per-token:
      One datapoint per real T5 token × stimulus.
      --captions controls which caption set: short, detailed, or both.
      RUST:  Requires cache/t5_xxl_tokens_{short,detailed}.pt
      HVM:   Requires cache/hvm_t5_xxl_tokens_{short,detailed}.pt
             Run scripts/cache_hvm_t5_xxl_tokens.py first.

Usage:
    python scripts/train_t5.py
    python scripts/train_t5.py --n-layers 2 --nce-weight 0.5
    python scripts/train_t5.py --per-token --captions both
    python scripts/train_t5.py --dataset hvm --captions short
    python scripts/train_t5.py --dataset hvm --captions short --use-category
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import SEED, CHECKPOINT_DIR, T5_PCA_K, HVM_N_CAT
from data_utils.rust_loader import make_multihead_loader
from data_utils.t5_token_loader import make_t5_token_loader
from data_utils.hvm_t5_token_loader import make_hvm_t5_token_loader
from encoders import TemporalTransformer
from eval.metrics import two_afc_identification, retrieval_accuracy
from get_device import get_device


# ---------------------------------------------------------------------------
# Loss functions (mirrored from train_multihead.py)
# ---------------------------------------------------------------------------

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def info_nce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    tgt_n  = F.normalize(target, dim=-1)
    logits = (pred_n @ tgt_n.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def head_loss(pred: torch.Tensor, target: torch.Tensor, nce_weight: float, temperature: float) -> torch.Tensor:
    l_cos = cosine_loss(pred, target)
    if nce_weight > 0.0 and pred.size(0) > 1:
        l_nce = info_nce_loss(pred, target, temperature)
        return nce_weight * l_nce + (1.0 - nce_weight) * l_cos
    return l_cos


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_name(args) -> str:
    mode = f"_tok{args.captions}" if args.per_token else ""
    cat  = "_cat" if getattr(args, "use_category", False) else ""
    return (
        f"d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}"
        f"_k{args.t5_pca_k}_do{args.dropout}"
        f"_lr{args.lr}_wd{args.weight_decay}"
        f"_tn{args.target_noise}_in{args.input_noise}_nd{args.neuron_dropout}"
        f"_bs{args.batch_size}_nw{args.nce_weight}_nt{args.nce_temperature}"
        f"_uw{args.uniformity_weight}{mode}{cat}"
    )


def run_dir(args) -> Path:
    dataset = getattr(args, "dataset", "rust")
    return CHECKPOINT_DIR / "transformer" / "t5" / dataset / run_name(args)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, args):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _extract_target(tgt) -> torch.Tensor:
    """Handle both multihead dict targets and plain token-loader tensors."""
    if isinstance(tgt, dict):
        return tgt["t5_pca"]
    return tgt


def _is_hvm_batch(batch) -> bool:
    """HVM T5 token batches are 3-tuples (neural, pca_coords, cat_idx)."""
    return len(batch) == 3


def _pool_by_stimulus(preds: torch.Tensor, stim_index: torch.Tensor) -> torch.Tensor:
    """Average per-token predictions to one L2-normalised vector per stimulus."""
    n_stim = int(stim_index.max().item()) + 1
    pooled = torch.zeros(n_stim, preds.shape[1])
    counts = torch.zeros(n_stim, 1)
    pooled.scatter_add_(0, stim_index.unsqueeze(1).expand_as(preds), preds)
    counts.scatter_add_(0, stim_index.unsqueeze(1), torch.ones(len(stim_index), 1))
    return F.normalize(pooled / counts.clamp(min=1), dim=-1)


def train(args):
    torch.manual_seed(SEED)
    device = get_device()

    dataset = getattr(args, "dataset", "rust")
    use_category = getattr(args, "use_category", False)

    if dataset == "hvm":
        train_loader, val_loader, test_loader = make_hvm_t5_token_loader(
            batch_size=args.batch_size, t5_pca_k=args.t5_pca_k, captions=args.captions,
        )
    elif args.per_token:
        train_loader, val_loader, test_loader = make_t5_token_loader(
            batch_size=args.batch_size, t5_pca_k=args.t5_pca_k, captions=args.captions,
        )
    else:
        train_loader, val_loader, test_loader = make_multihead_loader(
            batch_size=args.batch_size, t5_pca_k=args.t5_pca_k,
        )

    sample_batch = next(iter(train_loader))
    sample_neural = sample_batch[0]
    _, n_neurons, n_time = sample_neural.shape

    n_categories = HVM_N_CAT if (dataset == "hvm" and use_category) else 0

    model = TemporalTransformer(
        n_neurons=n_neurons,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        out_dim=args.t5_pca_k,
        dropout=args.dropout,
        n_categories=n_categories,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"TemporalTransformer (T5) | dataset: {dataset} | params: {n_params:,} | "
        f"out_dim: {args.t5_pca_k} | n_categories: {n_categories}"
    )

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

    best_val_cos = -1.0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            neural, tgt = batch[0], batch[1]
            cat = batch[2].to(device) if (_is_hvm_batch(batch) and use_category) else None

            x      = neural.permute(0, 2, 1).to(device)   # (B, T, N)
            t5_tgt = _extract_target(tgt).to(device)

            if args.target_noise > 0.0:
                t5_tgt = F.normalize(t5_tgt + torch.randn_like(t5_tgt) * args.target_noise, dim=-1)

            # input augmentation: additive gaussian noise + neuron dropout
            if args.input_noise > 0.0:
                x = x + torch.randn_like(x) * args.input_noise
            if args.neuron_dropout > 0.0:
                mask = (torch.rand(x.shape[0], 1, x.shape[2], device=device) > args.neuron_dropout).float()
                x = x * mask

            pred = model(x, cat)
            loss = head_loss(pred, t5_tgt, args.nce_weight, args.nce_temperature)

            if args.uniformity_weight > 0.0:
                loss = loss + args.uniformity_weight * uniformity_loss(pred)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch}.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- train cos / 2AFC (no-grad pass over train set) ---
        model.eval()
        train_preds, train_tgts = [], []
        with torch.no_grad():
            for batch in train_loader:
                neural, tgt = batch[0], batch[1]
                cat = batch[2].to(device) if (_is_hvm_batch(batch) and use_category) else None
                x = neural.permute(0, 2, 1).to(device)
                train_preds.append(model(x, cat).cpu())
                train_tgts.append(_extract_target(tgt))
        train_preds_t = torch.cat(train_preds)
        train_tgts_t  = torch.cat(train_tgts)
        if hasattr(train_loader.dataset, "stim_index"):
            sidx = train_loader.dataset.stim_index
            train_preds_stim = _pool_by_stimulus(train_preds_t, sidx)
            train_tgts_stim  = _pool_by_stimulus(train_tgts_t,  sidx)
        else:
            train_preds_stim, train_tgts_stim = train_preds_t, train_tgts_t
        train_cos  = F.cosine_similarity(train_preds_stim, train_tgts_stim, dim=-1).mean().item()
        train_2afc = two_afc_identification(train_preds_stim, train_tgts_stim)

        # --- val ---
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                neural, tgt = batch[0], batch[1]
                cat  = batch[2].to(device) if (_is_hvm_batch(batch) and use_category) else None
                x    = neural.permute(0, 2, 1).to(device)
                pred = model(x, cat)
                all_preds.append(pred.cpu())
                all_targets.append(_extract_target(tgt))

        preds_t   = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        if hasattr(val_loader.dataset, "stim_index"):
            sidx = val_loader.dataset.stim_index
            preds_stim   = _pool_by_stimulus(preds_t,   sidx)
            targets_stim = _pool_by_stimulus(targets_t, sidx)
        else:
            preds_stim, targets_stim = preds_t, targets_t
        val_cos  = F.cosine_similarity(preds_stim, targets_stim, dim=-1).mean().item()
        val_2afc = two_afc_identification(preds_stim, targets_stim)

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_cos={train_cos:.4f}  train_2afc={train_2afc:.3f}  "
            f"val_cos={val_cos:.4f}  val_2afc={val_2afc:.3f}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cos_t5": train_cos,
            "train_2afc_t5": train_2afc,
            "val_cos_t5": val_cos,
            "val_2afc_t5": val_2afc,
        }

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args)
        if epoch > warmup_epochs and val_cos > best_val_cos:
            best_val_cos = val_cos
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args)

    # --- test ---
    best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            neural, tgt = batch[0], batch[1]
            cat  = batch[2].to(device) if (_is_hvm_batch(batch) and use_category) else None
            x    = neural.permute(0, 2, 1).to(device)
            pred = model(x, cat)
            test_preds.append(pred.cpu())
            test_targets.append(_extract_target(tgt))

    tp = torch.cat(test_preds)
    tt = torch.cat(test_targets)

    if hasattr(test_loader.dataset, "stim_index"):
        sidx = test_loader.dataset.stim_index
        tp_stim = _pool_by_stimulus(tp, sidx)
        tt_stim = _pool_by_stimulus(tt, sidx)
    else:
        tp_stim, tt_stim = tp, tt

    test_cos  = F.cosine_similarity(tp_stim, tt_stim, dim=-1).mean().item()
    test_2afc = two_afc_identification(tp_stim, tt_stim)
    test_topk = retrieval_accuracy(tp_stim, tt_stim, k=[1, 5, 10])

    print(
        f"\nTest (N={len(tp_stim)}):"
        f"  cos={test_cos:.4f}"
        f"  2AFC={test_2afc:.3f}"
        f"  top-1={test_topk[1]:.3f}"
        f"  top-5={test_topk[5]:.3f}"
        f"  top-10={test_topk[10]:.3f}"
    )

    test_metrics = {
        "test_cos_t5": test_cos,
        "test_2afc_t5": test_2afc,
        "test_top1": test_topk[1],
        "test_top5": test_topk[5],
        "test_top10": test_topk[10],
    }
    best_ckpt["test_metrics"] = test_metrics
    torch.save(best_ckpt, ckpt_dir / "best.pt")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Dataset
    p.add_argument("--dataset",      choices=["rust", "hvm"], default="rust",
                   help="Which dataset to train on (default: rust)")
    p.add_argument("--use-category", action="store_true",
                   help="Add learned category conditioning (HVM only, requires --dataset hvm)")

    # Transformer backbone
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=1)
    p.add_argument("--t5-pca-k",   type=int,   default=T5_PCA_K,
                   help="Number of T5 PCA components to predict")
    p.add_argument("--dropout",    type=float, default=0.1)

    # Input augmentation
    p.add_argument("--input-noise",     type=float, default=0.05,
                   help="Std of Gaussian noise added to neural firing rates during training")
    p.add_argument("--neuron-dropout",  type=float, default=0.1,
                   help="Fraction of neurons randomly zeroed per sample during training")

    # Loss
    p.add_argument("--nce-weight",      type=float, default=0.8,
                   help="Mix: 1.0=pure InfoNCE, 0.0=pure cosine")
    p.add_argument("--nce-temperature", type=float, default=0.07)
    p.add_argument("--uniformity-weight", type=float, default=0.1)
    p.add_argument("--target-noise",    type=float, default=0.0,
                   help="Gaussian noise std added to T5 PCA targets (re-normalised after)")

    # Per-token mode
    p.add_argument("--per-token",  action="store_true",
                   help="Train on individual T5 token embeddings instead of mean-pooled vector")
    p.add_argument("--captions",   choices=["short", "detailed", "both"], default="both",
                   help="Caption set for per-token mode (default: both)")

    # Optimisation
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac",  type=float, default=0.1)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

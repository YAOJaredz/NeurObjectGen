"""Train a MultiHeadTransformer: shared backbone → SigLIP, CLIP, and T5-PCA heads.

Loss:
  L = w_siglip * cosine_loss(pred_siglip, tgt_siglip)
    + w_clip   * cosine_loss(pred_clip,   tgt_clip)
    + w_t5     * mse_loss(pred_t5_pca,    tgt_t5_pca)
    + w_unif   * uniformity_loss(shared_latent)

Checkpoint criterion: best mean 2-AFC across SigLIP and CLIP heads.

Usage:
    python scripts/train_multihead.py
    python scripts/train_multihead.py --loss-weight-t5 0.1  # reduce T5 influence
    python scripts/train_multihead.py --loss-weight-t5 0.0  # 2-head ablation
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

from config_const import SEED, CHECKPOINT_DIR
from data_utils.rust_loader import make_multihead_loader
from encoders import MultiHeadTransformer
from eval.metrics import two_afc_identification, retrieval_accuracy
from get_device import get_device


# ---------------------------------------------------------------------------
# Loss functions (reused from train_encoder.py)
# ---------------------------------------------------------------------------

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_name(args) -> str:
    return (
        f"d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}"
        f"_sd{args.shared_dim}_k{args.t5_pca_k}"
        f"_do{args.dropout}_lr{args.lr}_wd{args.weight_decay}"
        f"_tn{args.target_noise}_bs{args.batch_size}"
        f"_ws{args.loss_weight_siglip}_wc{args.loss_weight_clip}"
        f"_wt{args.loss_weight_t5}_uw{args.uniformity_weight}"
    )


def run_dir(args) -> Path:
    return CHECKPOINT_DIR / "multihead" / run_name(args)


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

def train(args):
    torch.manual_seed(SEED)
    device = get_device()

    train_loader, val_loader, test_loader = make_multihead_loader(batch_size=args.batch_size)

    sample_neural, _ = next(iter(train_loader))
    _, n_neurons, n_time = sample_neural.shape

    model = MultiHeadTransformer(
        n_neurons=n_neurons,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        shared_dim=args.shared_dim,
        t5_pca_k=args.t5_pca_k,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultiHeadTransformer | params: {n_params:,}")

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

    best_mean_cos = -1.0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = train_l_sig = train_l_clip = train_l_t5 = 0.0

        for neural, tgt in train_loader:
            x = neural.permute(0, 2, 1).to(device)  # (B, T, N)
            pred = model(x)

            sig_tgt  = tgt["siglip"].to(device)
            clip_tgt = tgt["clip"].to(device)
            t5_tgt   = tgt["t5_pca"].to(device)

            if args.target_noise > 0.0:
                sig_tgt  = F.normalize(sig_tgt  + torch.randn_like(sig_tgt)  * args.target_noise, dim=-1)
                clip_tgt = F.normalize(clip_tgt + torch.randn_like(clip_tgt) * args.target_noise, dim=-1)

            l_sig  = cosine_loss(pred["siglip"], sig_tgt)
            l_clip = cosine_loss(pred["clip"],   clip_tgt)
            l_t5   = F.mse_loss(pred["t5_pca"],  t5_tgt)
            l_unif = uniformity_loss(pred["shared"]) if args.uniformity_weight > 0.0 else torch.tensor(0.0)

            loss = (args.loss_weight_siglip * l_sig
                  + args.loss_weight_clip   * l_clip
                  + args.loss_weight_t5     * l_t5
                  + args.uniformity_weight  * l_unif)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch}.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss   += loss.item()
            train_l_sig  += l_sig.item()
            train_l_clip += l_clip.item()
            train_l_t5   += l_t5.item()

        n_batches = len(train_loader)
        train_loss   /= n_batches
        train_l_sig  /= n_batches
        train_l_clip /= n_batches
        train_l_t5   /= n_batches
        scheduler.step()

        # --- val ---
        model.eval()
        preds_sig, preds_clip, gt_sig, gt_clip = [], [], [], []
        val_mse_t5 = 0.0

        with torch.no_grad():
            for neural, tgt in val_loader:
                x = neural.permute(0, 2, 1).to(device)
                pred = model(x)
                preds_sig.append(pred["siglip"].cpu())
                preds_clip.append(pred["clip"].cpu())
                gt_sig.append(tgt["siglip"])
                gt_clip.append(tgt["clip"])
                val_mse_t5 += F.mse_loss(pred["t5_pca"], tgt["t5_pca"].to(device)).item()

        val_mse_t5 /= len(val_loader)
        preds_sig_t  = torch.cat(preds_sig)
        preds_clip_t = torch.cat(preds_clip)
        gt_sig_t     = torch.cat(gt_sig)
        gt_clip_t    = torch.cat(gt_clip)

        afc_sig  = two_afc_identification(preds_sig_t,  gt_sig_t)
        afc_clip = two_afc_identification(preds_clip_t, gt_clip_t)
        cos_sig  = F.cosine_similarity(preds_sig_t,  gt_sig_t,  dim=-1).mean().item()
        cos_clip = F.cosine_similarity(preds_clip_t, gt_clip_t, dim=-1).mean().item()
        mean_2afc = (afc_sig + afc_clip) / 2.0
        mean_cos  = (cos_sig + cos_clip) / 2.0

        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f} (sig={train_l_sig:.3f} clip={train_l_clip:.3f} t5={train_l_t5:.3f})  "
            f"val: sig_cos={cos_sig:.3f} clip_cos={cos_clip:.3f} "
            f"sig_2afc={afc_sig:.3f} clip_2afc={afc_clip:.3f} t5_mse={val_mse_t5:.4f}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss, "train_l_siglip": train_l_sig,
            "train_l_clip": train_l_clip, "train_l_t5": train_l_t5,
            "val_2afc_siglip": afc_sig, "val_2afc_clip": afc_clip,
            "val_mean_2afc": mean_2afc,
            "val_cos_siglip": cos_sig, "val_cos_clip": cos_clip,
            "val_mean_cos": mean_cos,
            "val_mse_t5": val_mse_t5,
        }

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args)
        # Guard against noisy early-epoch cos_sim: only checkpoint after warmup
        if epoch > warmup_epochs and mean_cos > best_mean_cos:
            best_mean_cos = mean_cos
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args)

    # --- test ---
    best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    test_sig, test_clip, test_gt_sig, test_gt_clip = [], [], [], []
    with torch.no_grad():
        for neural, tgt in test_loader:
            x = neural.permute(0, 2, 1).to(device)
            pred = model(x)
            test_sig.append(pred["siglip"].cpu())
            test_clip.append(pred["clip"].cpu())
            test_gt_sig.append(tgt["siglip"])
            test_gt_clip.append(tgt["clip"])

    ts  = torch.cat(test_sig);  tgs = torch.cat(test_gt_sig)
    tc  = torch.cat(test_clip); tgc = torch.cat(test_gt_clip)

    test_afc_sig  = two_afc_identification(ts, tgs)
    test_afc_clip = two_afc_identification(tc, tgc)
    test_topk_sig  = retrieval_accuracy(ts, tgs, k=[1, 5, 10])
    test_topk_clip = retrieval_accuracy(tc, tgc, k=[1, 5, 10])
    test_cos_sig   = F.cosine_similarity(ts, tgs, dim=-1).mean().item()
    test_cos_clip  = F.cosine_similarity(tc, tgc, dim=-1).mean().item()

    print(
        f"\nTest (N={len(ts)}):\n"
        f"  SigLIP: cos={test_cos_sig:.4f}  2AFC={test_afc_sig:.3f}"
        f"  top-1={test_topk_sig[1]:.3f}  top-5={test_topk_sig[5]:.3f}  top-10={test_topk_sig[10]:.3f}\n"
        f"  CLIP:   cos={test_cos_clip:.4f}  2AFC={test_afc_clip:.3f}"
        f"  top-1={test_topk_clip[1]:.3f}  top-5={test_topk_clip[5]:.3f}  top-10={test_topk_clip[10]:.3f}"
    )

    test_metrics = {
        "test_2afc_siglip": test_afc_sig, "test_2afc_clip": test_afc_clip,
        "test_top1_siglip": test_topk_sig[1], "test_top5_siglip": test_topk_sig[5],
        "test_top10_siglip": test_topk_sig[10],
        "test_top1_clip": test_topk_clip[1], "test_top5_clip": test_topk_clip[5],
        "test_top10_clip": test_topk_clip[10],
        "test_cos_siglip": test_cos_sig, "test_cos_clip": test_cos_clip,
    }
    best_ckpt["test_metrics"] = test_metrics
    torch.save(best_ckpt, ckpt_dir / "best.pt")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Transformer backbone
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=1)
    p.add_argument("--shared-dim", type=int,   default=512,
                   help="Shared latent dimension before the three heads")
    p.add_argument("--t5-pca-k",   type=int,   default=64,
                   help="Number of T5 PCA components to predict")
    p.add_argument("--dropout",    type=float, default=0.1)

    # Loss weights
    p.add_argument("--loss-weight-siglip", type=float, default=1.0)
    p.add_argument("--loss-weight-clip",   type=float, default=1.0)
    p.add_argument("--loss-weight-t5",     type=float, default=0.5,
                   help="Weight for T5-PCA MSE loss. Set 0.0 for 2-head ablation.")
    p.add_argument("--uniformity-weight",  type=float, default=0.1)
    p.add_argument("--target-noise",       type=float, default=0.02,
                   help="Gaussian noise std added to SigLIP/CLIP targets (re-normalised after)")

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

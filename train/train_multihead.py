"""Train a MultiHeadTransformer: shared backbone → SigLIP and CLIP heads.

Loss:
  L = w_siglip * head_loss(pred_siglip, tgt_siglip)
    + w_clip   * head_loss(pred_clip,   tgt_clip)
    + w_unif   * uniformity_loss(shared_latent)

Checkpoint criterion: best mean 2-AFC across SigLIP and CLIP heads.

Usage:
    python scripts/train_multihead.py
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

sys.path.append('.')

from config_const import SEED, CHECKPOINT_DIR, HVM_N_CAT
from data_utils.rust_loader import make_multihead_loader
from data_utils.hvm_loader import make_hvm_multihead_loader
from encoders import MultiHeadTransformer
from eval.metrics import two_afc_identification, retrieval_accuracy
from train.losses import cosine_loss, info_nce_loss, head_loss, uniformity_loss
from get_device import get_device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_name(args) -> str:
    base = (
        f"d{args.d_model}_nh{args.n_heads}_nl{args.n_layers}"
        f"_sd{args.shared_dim}"
        f"_do{args.dropout}_lr{args.lr}_wd{args.weight_decay}"
        f"_tn{args.target_noise}_in{args.input_noise}_nd{args.neuron_dropout}"
        f"_bs{args.batch_size}"
        f"_ws{args.loss_weight_siglip}_wc{args.loss_weight_clip}"
        f"_uw{args.uniformity_weight}"
        f"_nw{args.nce_weight}_nt{args.nce_temperature}"
    )
    if args.n_neurons is not None:
        base += f"_nn{args.n_neurons}_ns{args.neuron_seed}"
    return base


def run_dir(args) -> Path:
    tag = args.dataset + ("_cat" if getattr(args, "use_category", False) else "")
    if getattr(args, "area", None):
        tag += f"_{args.area.upper()}"
    return CHECKPOINT_DIR / "multihead" / tag / run_name(args)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, args,
                    neuron_indices: np.ndarray | None = None):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "neuron_indices": neuron_indices,
    }, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    device = get_device()

    neuron_indices = None
    if args.dataset == "hvm":
        area = getattr(args, "area", None) or 'all'
        if area == 'all' and args.n_neurons is not None:
            from data_utils.hvm_loader import _load_hvm_neural
            full_rsp, _, _ = _load_hvm_neural()
            n_total = full_rsp.shape[1]
            if args.n_neurons >= n_total:
                print(f"--n-neurons {args.n_neurons} >= full count {n_total}; using all neurons.")
            else:
                rng = np.random.default_rng(args.neuron_seed)
                neuron_indices = rng.choice(n_total, size=args.n_neurons, replace=False)
                neuron_indices.sort()
                print(f"Neuron subset: {args.n_neurons}/{n_total} (seed={args.neuron_seed})")
        train_loader, val_loader, test_loader = make_hvm_multihead_loader(
            batch_size=args.batch_size, area=area, neuron_indices=neuron_indices,
        )
        clip_key = "clip_short"
    else:
        train_loader, val_loader, test_loader = make_multihead_loader(batch_size=args.batch_size)
        clip_key = "clip"

    sample_batch = next(iter(train_loader))
    sample_neural = sample_batch[0]
    _, n_neurons, n_time = sample_neural.shape

    n_cat = HVM_N_CAT if (args.dataset == "hvm" and args.use_category) else 0
    model = MultiHeadTransformer(
        n_neurons=n_neurons,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        shared_dim=args.shared_dim,
        dropout=args.dropout,
        n_categories=n_cat,
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
        train_loss = train_l_sig = train_l_clip = 0.0

        for batch in train_loader:
            neural, tgt = batch[0], batch[1]
            cat = batch[2].to(device) if len(batch) >= 3 else None

            x = neural.permute(0, 2, 1).to(device)  # (B, T, N)

            sig_tgt  = tgt["siglip"].to(device)
            clip_tgt = tgt[clip_key].to(device)

            if args.target_noise > 0.0:
                sig_tgt  = F.normalize(sig_tgt  + torch.randn_like(sig_tgt)  * args.target_noise, dim=-1)
                clip_tgt = F.normalize(clip_tgt + torch.randn_like(clip_tgt) * args.target_noise, dim=-1)

            if args.input_noise > 0.0:
                x = x + torch.randn_like(x) * args.input_noise
            if args.neuron_dropout > 0.0:
                mask = (torch.rand(x.shape[0], 1, x.shape[2], device=device) > args.neuron_dropout).float()
                x = x * mask

            pred = model(x, cat)

            l_sig  = head_loss(pred["siglip"], sig_tgt,  args.nce_weight, args.nce_temperature)
            l_clip = head_loss(pred["clip"],   clip_tgt, args.nce_weight, args.nce_temperature)
            if args.uniformity_weight > 0.0:
                l_unif = (uniformity_loss(pred["shared"])
                          + uniformity_loss(F.normalize(pred["siglip"], dim=-1))
                          + uniformity_loss(F.normalize(pred["clip"], dim=-1))) / 3.0
            else:
                l_unif = torch.tensor(0.0)

            loss = (args.loss_weight_siglip * l_sig
                  + args.loss_weight_clip   * l_clip
                  + args.uniformity_weight  * l_unif)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch}.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss   += loss.item()
            train_l_sig  += l_sig.item()
            train_l_clip += l_clip.item()

        n_batches = len(train_loader)
        train_loss   /= n_batches
        train_l_sig  /= n_batches
        train_l_clip /= n_batches
        scheduler.step()

        # --- val ---
        model.eval()
        preds_sig, preds_clip, gt_sig, gt_clip = [], [], [], []

        with torch.no_grad():
            for batch in val_loader:
                neural, tgt = batch[0], batch[1]
                cat = batch[2].to(device) if len(batch) >= 3 else None
                x = neural.permute(0, 2, 1).to(device)
                pred = model(x, cat)
                preds_sig.append(pred["siglip"].cpu())
                preds_clip.append(pred["clip"].cpu())
                gt_sig.append(tgt["siglip"])
                gt_clip.append(tgt[clip_key])

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
            f"loss={train_loss:.4f} (sig={train_l_sig:.3f} clip={train_l_clip:.3f})  "
            f"val: sig_cos={cos_sig:.3f} clip_cos={cos_clip:.3f} "
            f"sig_2afc={afc_sig:.3f} clip_2afc={afc_clip:.3f}  crit(cos)={mean_cos:.4f}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss, "train_l_siglip": train_l_sig,
            "train_l_clip": train_l_clip,
            "val_2afc_siglip": afc_sig, "val_2afc_clip": afc_clip,
            "val_mean_2afc": mean_2afc,
            "val_cos_siglip": cos_sig, "val_cos_clip": cos_clip,
            "val_mean_cos": mean_cos,
        }

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args, neuron_indices)
        # Guard against noisy early-epoch cos_sim: only checkpoint after warmup
        if epoch > warmup_epochs and mean_cos > best_mean_cos:
            best_mean_cos = mean_cos
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args, neuron_indices)

    # --- test ---
    best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    test_sig, test_clip, test_gt_sig, test_gt_clip = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            neural, tgt = batch[0], batch[1]
            cat = batch[2].to(device) if len(batch) >= 3 else None
            x = neural.permute(0, 2, 1).to(device)
            pred = model(x, cat)
            test_sig.append(pred["siglip"].cpu())
            test_clip.append(pred["clip"].cpu())
            test_gt_sig.append(tgt["siglip"])
            test_gt_clip.append(tgt[clip_key])

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
    best_ckpt["neuron_indices"] = neuron_indices
    torch.save(best_ckpt, ckpt_dir / "best.pt")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Dataset + conditioning
    p.add_argument("--dataset", choices=["rust", "hvm"], default="rust",
                   help="Neural dataset to train on.")
    p.add_argument("--use-category", action="store_true", default=False,
                   help="Condition on category label (HVM only; adds learned "
                        "Embedding at shared latent).")

    # Transformer backbone
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=1)
    p.add_argument("--shared-dim", type=int,   default=512,
                   help="Shared latent dimension before the two heads")
    p.add_argument("--dropout",    type=float, default=0.1)

    # Loss weights
    p.add_argument("--loss-weight-siglip", type=float, default=1.0)
    p.add_argument("--loss-weight-clip",   type=float, default=1.0)
    p.add_argument("--uniformity-weight",  type=float, default=0.1)
    p.add_argument("--target-noise",       type=float, default=0.02,
                   help="Gaussian noise std added to SigLIP/CLIP targets (re-normalised after)")
    p.add_argument("--input-noise",        type=float, default=0.05,
                   help="Std of Gaussian noise added to neural firing rates during training")
    p.add_argument("--neuron-dropout",     type=float, default=0.1,
                   help="Fraction of neurons randomly zeroed per sample during training")
    p.add_argument("--nce-weight",         type=float, default=0.8,
                   help="Mix between InfoNCE (primary) and cosine (regulariser) in each head loss. "
                        "0.0 = pure cosine (old behaviour), 1.0 = pure InfoNCE.")
    p.add_argument("--nce-temperature",    type=float, default=0.07,
                   help="InfoNCE softmax temperature. Lower → sharper discrimination.")

    # Optimisation
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac",  type=float, default=0.1)

    # Neuron subset ablation
    p.add_argument("--n-neurons",    type=int,   default=None,
                   help="Number of neurons to randomly sample. None = use all neurons.")
    p.add_argument("--neuron-seed",  type=int,   default=0,
                   help="RNG seed for neuron subset sampling.")

    # Brain-area subset
    p.add_argument("--area",         type=str,   default=None,
                   help="Restrict to neurons from one brain area, e.g. 'TE2', 'PRH'. "
                        "Mutually exclusive with --n-neurons.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

"""Train MultiHeadTransformerV2: shared backbone → global SigLIP, obj SigLIP, and CLIP heads.

Loss:
  L = w_sg  * head_loss(pred_siglip_global, tgt_sg)    # InfoNCE + cosine mix
    + w_so  * head_loss(pred_siglip_obj,    tgt_so)    # InfoNCE + cosine mix
    + w_clip * cosine_loss(pred_clip,       tgt_clip)  # cosine only (no InfoNCE)
    + w_unif * mean(uniformity across all four outputs)

In predict mode, CategoryClassifier is pre-trained first (--clf-pretrain-epochs), then
the main backbone trains with the warm classifier (separate optimizer, decoupled gradients).

Checkpoint criterion: best mean cosine similarity across all three heads (after warmup).

Usage:
    python train/train_multihead_v2.py --cat-mode given
    python train/train_multihead_v2.py --cat-mode predict --clf-pretrain-epochs 50
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
from data_utils.hvm_loader import make_hvm_multihead_loader
from encoders import MultiHeadTransformerV2
from eval.metrics import two_afc_identification, retrieval_accuracy
from train.losses import cosine_loss, head_loss, uniformity_loss
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
        f"_wsg{args.loss_weight_siglip}_wso{args.loss_weight_siglip_obj}_wc{args.loss_weight_clip}"
        f"_uw{args.uniformity_weight}"
        f"_nw{args.nce_weight}_nt{args.nce_temperature}"
        f"_cm{args.cat_mode}"
    )

    if args.n_neurons is not None:
        base += f"_nn{args.n_neurons}_ns{args.neuron_seed}"
    return base


def run_dir(args) -> Path:
    tag = "hvm_v2"
    if getattr(args, "area", None):
        tag += f"_{args.area.upper()}"
    return CHECKPOINT_DIR / "multihead_v2" / tag / run_name(args)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, args,
                    neuron_indices: np.ndarray | None = None,
                    clf_optimizer=None, clf_scheduler=None):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "clf_optimizer_state": clf_optimizer.state_dict() if clf_optimizer is not None else None,
        "clf_scheduler_state": clf_scheduler.state_dict() if clf_scheduler is not None else None,
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

    area = getattr(args, "area", None) or 'all'
    neuron_indices = None
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

    sample_batch = next(iter(train_loader))
    sample_neural = sample_batch[0]
    _, n_neurons, n_time = sample_neural.shape

    model = MultiHeadTransformerV2(
        n_neurons=n_neurons,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        shared_dim=args.shared_dim,
        dropout=args.dropout,
        n_categories=HVM_N_CAT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultiHeadTransformerV2 | params: {n_params:,} | cat_mode: {args.cat_mode}")

    clf_params  = set(model.cat_clf.parameters())
    main_params = [p for p in model.parameters() if p not in clf_params]

    # Dedicated optimizer for the independent CategoryClassifier
    clf_optimizer = AdamW(model.cat_clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_dir = run_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    # -----------------------------------------------------------------------
    # Phase 1 (predict mode only): pre-train CategoryClassifier until
    # val accuracy converges, then load best classifier weights.
    # -----------------------------------------------------------------------
    if args.cat_mode == "predict":
        print("\n=== Phase 1: CategoryClassifier pre-training (patience=15, min_delta=0.001) ===")

        clf_scheduler_pre = LambdaLR(clf_optimizer, lr_lambda=lambda e: 1.0)  # constant LR for pre-train

        best_clf_acc = -1.0
        best_clf_state = None
        patience, min_delta, no_improve = 15, 0.001, 0
        min_pretrain_epochs = 50

        for epoch in range(1, 501):  # hard cap 500 epochs
            model.train()
            total_ce = correct = total = 0
            for batch in train_loader:
                neural, _, cat = batch
                cat = cat.to(device)
                x = neural.permute(0, 2, 1).to(device)
                if args.input_noise > 0.0:
                    x = x + torch.randn_like(x) * args.input_noise
                if args.neuron_dropout > 0.0:
                    mask = (torch.rand(x.shape[0], 1, x.shape[2], device=device) > args.neuron_dropout).float()
                    x = x * mask
                logits = model.cat_clf(x)
                l_ce = F.cross_entropy(logits, cat)
                clf_optimizer.zero_grad()
                l_ce.backward()
                clf_optimizer.step()
                total_ce += l_ce.item()
                correct  += (logits.argmax(-1) == cat).sum().item()
                total    += cat.size(0)

            train_acc = correct / total

            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    neural, _, cat = batch
                    cat = cat.to(device)
                    x = neural.permute(0, 2, 1).to(device)
                    logits = model.cat_clf(x)
                    val_correct += (logits.argmax(-1) == cat).sum().item()
                    val_total   += cat.size(0)
            val_acc = val_correct / val_total

            print(f"  clf epoch {epoch:3d}  ce={total_ce/len(train_loader):.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

            if val_acc > best_clf_acc + min_delta:
                best_clf_acc = val_acc
                best_clf_state = {k: v.clone() for k, v in model.cat_clf.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience and epoch >= min_pretrain_epochs:
                    print(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs).")
                    break

        model.cat_clf.load_state_dict(best_clf_state)
        print(f"=== Classifier pre-training done. Best val_acc={best_clf_acc:.3f} ===\n")

        # Reset clf_optimizer for Phase 2
        clf_optimizer = AdamW(model.cat_clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------------------------------------------------
    # Phase 2: train main backbone (+ continue classifier)
    # -----------------------------------------------------------------------
    optimizer = AdamW(main_params, lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, int(args.epochs * args.warmup_frac)) if args.warmup_frac > 0 else 0

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    clf_scheduler = LambdaLR(clf_optimizer, lr_lambda=lr_lambda)

    best_mean_cos = -1.0

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = train_l_sg = train_l_so = train_l_clip = train_l_ce = 0.0

        for batch in train_loader:
            neural, tgt, cat = batch
            cat = cat.to(device)
            x = neural.permute(0, 2, 1).to(device)  # (B, T, N)

            sg_tgt   = tgt["siglip"].to(device)
            so_tgt   = tgt["siglip_obj"].to(device)
            clip_tgt = tgt["clip_short"].to(device)

            if args.target_noise > 0.0:
                sg_tgt   = F.normalize(sg_tgt   + torch.randn_like(sg_tgt)   * args.target_noise, dim=-1)
                so_tgt   = F.normalize(so_tgt   + torch.randn_like(so_tgt)   * args.target_noise, dim=-1)
                clip_tgt = F.normalize(clip_tgt + torch.randn_like(clip_tgt) * args.target_noise, dim=-1)

            if args.input_noise > 0.0:
                x = x + torch.randn_like(x) * args.input_noise
            if args.neuron_dropout > 0.0:
                mask = (torch.rand(x.shape[0], 1, x.shape[2], device=device) > args.neuron_dropout).float()
                x = x * mask

            pred = model(x, cat, cat_mode=args.cat_mode)

            l_sg   = head_loss(pred["siglip_global"], sg_tgt,   args.nce_weight, args.nce_temperature)
            l_so   = head_loss(pred["siglip_obj"],    so_tgt,   args.nce_weight, args.nce_temperature)
            l_clip = cosine_loss(pred["clip"],        clip_tgt)

            if args.uniformity_weight > 0.0:
                l_unif = (
                    uniformity_loss(pred["shared"])
                    + uniformity_loss(pred["siglip_global"])
                    + uniformity_loss(pred["siglip_obj"])
                    + uniformity_loss(pred["clip"])
                ) / 4.0
            else:
                l_unif = torch.tensor(0.0, device=device)

            loss = (args.loss_weight_siglip     * l_sg
                  + args.loss_weight_siglip_obj * l_so
                  + args.loss_weight_clip       * l_clip
                  + args.uniformity_weight      * l_unif)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch}.")

            # Main backbone step — cat_clf params excluded from this optimizer
            optimizer.zero_grad()
            loss.backward(retain_graph=args.cat_mode == "predict")
            optimizer.step()

            l_ce = torch.tensor(0.0, device=device)
            if args.cat_mode == "predict" and pred["cat_logits"] is not None:
                l_ce = F.cross_entropy(pred["cat_logits"], cat)
                clf_optimizer.zero_grad()
                l_ce.backward()
                clf_optimizer.step()

            train_loss   += loss.item()
            train_l_sg   += l_sg.item()
            train_l_so   += l_so.item()
            train_l_clip += l_clip.item()
            train_l_ce   += l_ce.item()

        n_batches = len(train_loader)
        train_loss   /= n_batches
        train_l_sg   /= n_batches
        train_l_so   /= n_batches
        train_l_clip /= n_batches
        train_l_ce   /= n_batches
        scheduler.step()
        if args.cat_mode == "predict":
            clf_scheduler.step()

        # --- val ---
        model.eval()
        preds_sg, preds_so, preds_clip = [], [], []
        gt_sg, gt_so, gt_clip = [], [], []
        cat_correct = cat_total = 0

        with torch.no_grad():
            for batch in val_loader:
                neural, tgt, cat = batch
                cat = cat.to(device)
                x = neural.permute(0, 2, 1).to(device)
                pred = model(x, cat, cat_mode=args.cat_mode)
                preds_sg.append(pred["siglip_global"].cpu())
                preds_so.append(pred["siglip_obj"].cpu())
                preds_clip.append(pred["clip"].cpu())
                gt_sg.append(tgt["siglip"])
                gt_so.append(tgt["siglip_obj"])
                gt_clip.append(tgt["clip_short"])
                if pred["cat_logits"] is not None:
                    cat_correct += (pred["cat_logits"].argmax(-1) == cat).sum().item()
                    cat_total   += cat.size(0)

        preds_sg_t   = torch.cat(preds_sg)
        preds_so_t   = torch.cat(preds_so)
        preds_clip_t = torch.cat(preds_clip)
        gt_sg_t      = torch.cat(gt_sg)
        gt_so_t      = torch.cat(gt_so)
        gt_clip_t    = torch.cat(gt_clip)

        afc_sg   = two_afc_identification(preds_sg_t, gt_sg_t)
        afc_so   = two_afc_identification(preds_so_t, gt_so_t)
        cos_sg   = F.cosine_similarity(preds_sg_t,   gt_sg_t,   dim=-1).mean().item()
        cos_so   = F.cosine_similarity(preds_so_t,   gt_so_t,   dim=-1).mean().item()
        cos_clip = F.cosine_similarity(preds_clip_t, gt_clip_t, dim=-1).mean().item()

        mean_2afc = (afc_sg + afc_so) / 2.0
        mean_cos  = (cos_sg + cos_so + cos_clip) / 3.0
        cat_acc   = cat_correct / cat_total if cat_total > 0 else float("nan")

        cat_str = f"  cat_acc={cat_acc:.3f}" if cat_total > 0 else ""
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.4f} (sg={train_l_sg:.3f} so={train_l_so:.3f} "
            f"clip={train_l_clip:.3f} ce={train_l_ce:.3f})  "
            f"val: cos_sg={cos_sg:.3f} cos_so={cos_so:.3f} cos_clip={cos_clip:.3f}  "
            f"2afc_sg={afc_sg:.3f} 2afc_so={afc_so:.3f}  "
            f"crit(cos)={mean_cos:.4f}{cat_str}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l_siglip_global": train_l_sg, "train_l_siglip_obj": train_l_so,
            "train_l_clip": train_l_clip, "train_l_ce": train_l_ce,
            "val_2afc_siglip_global": afc_sg, "val_2afc_siglip_obj": afc_so,
            "val_mean_2afc": mean_2afc,
            "val_cos_siglip_global": cos_sg, "val_cos_siglip_obj": cos_so,
            "val_cos_clip": cos_clip, "val_mean_cos": mean_cos,
            "val_cat_acc": cat_acc,
        }

        _clf_opt = clf_optimizer if args.cat_mode == "predict" else None
        _clf_sch = clf_scheduler if args.cat_mode == "predict" else None
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, args, neuron_indices,
                        clf_optimizer=_clf_opt, clf_scheduler=_clf_sch)
        if epoch > warmup_epochs and mean_cos > best_mean_cos:
            best_mean_cos = mean_cos
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, args, neuron_indices,
                            clf_optimizer=_clf_opt, clf_scheduler=_clf_sch)

    # --- test ---
    best_ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    test_sg, test_so, test_clip = [], [], []
    test_gt_sg, test_gt_so, test_gt_clip = [], [], []
    test_cat_correct = test_cat_total = 0

    with torch.no_grad():
        for batch in test_loader:
            neural, tgt, cat = batch
            cat = cat.to(device)
            x = neural.permute(0, 2, 1).to(device)
            pred = model(x, cat, cat_mode=args.cat_mode)
            test_sg.append(pred["siglip_global"].cpu())
            test_so.append(pred["siglip_obj"].cpu())
            test_clip.append(pred["clip"].cpu())
            test_gt_sg.append(tgt["siglip"])
            test_gt_so.append(tgt["siglip_obj"])
            test_gt_clip.append(tgt["clip_short"])
            if pred["cat_logits"] is not None:
                test_cat_correct += (pred["cat_logits"].argmax(-1) == cat).sum().item()
                test_cat_total   += cat.size(0)

    ts_sg = torch.cat(test_sg);  tg_sg = torch.cat(test_gt_sg)
    ts_so = torch.cat(test_so);  tg_so = torch.cat(test_gt_so)
    ts_c  = torch.cat(test_clip); tg_c  = torch.cat(test_gt_clip)

    def eval_siglip_head(pred_t, gt_t, label):
        afc  = two_afc_identification(pred_t, gt_t)
        topk = retrieval_accuracy(pred_t, gt_t, k=[1, 5, 10])
        cos  = F.cosine_similarity(pred_t, gt_t, dim=-1).mean().item()
        print(
            f"  {label}: cos={cos:.4f}  2AFC={afc:.3f}"
            f"  top-1={topk[1]:.3f}  top-5={topk[5]:.3f}  top-10={topk[10]:.3f}"
        )
        return afc, topk, cos

    def eval_clip_head(pred_t, gt_t):
        topk = retrieval_accuracy(pred_t, gt_t, k=[1, 5, 10])
        cos  = F.cosine_similarity(pred_t, gt_t, dim=-1).mean().item()
        print(
            f"  CLIP:          cos={cos:.4f}"
            f"  top-1={topk[1]:.3f}  top-5={topk[5]:.3f}  top-10={topk[10]:.3f}"
        )
        return topk, cos

    print(f"\nTest (N={len(ts_sg)}):")
    test_afc_sg, test_topk_sg, test_cos_sg = eval_siglip_head(ts_sg, tg_sg, "SigLIP-global")
    test_afc_so, test_topk_so, test_cos_so = eval_siglip_head(ts_so, tg_so, "SigLIP-obj   ")
    test_topk_clip, test_cos_clip          = eval_clip_head(ts_c, tg_c)

    test_cat_acc = test_cat_correct / test_cat_total if test_cat_total > 0 else float("nan")
    if test_cat_total > 0:
        print(f"  Cat accuracy: {test_cat_acc:.3f}")

    test_metrics = {
        "test_2afc_siglip_global": test_afc_sg,
        "test_top1_siglip_global": test_topk_sg[1], "test_top5_siglip_global": test_topk_sg[5],
        "test_top10_siglip_global": test_topk_sg[10], "test_cos_siglip_global": test_cos_sg,
        "test_2afc_siglip_obj": test_afc_so,
        "test_top1_siglip_obj": test_topk_so[1], "test_top5_siglip_obj": test_topk_so[5],
        "test_top10_siglip_obj": test_topk_so[10], "test_cos_siglip_obj": test_cos_so,
        "test_top1_clip": test_topk_clip[1], "test_top5_clip": test_topk_clip[5],
        "test_top10_clip": test_topk_clip[10], "test_cos_clip": test_cos_clip,
        "test_cat_acc": test_cat_acc,
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

    # Category conditioning mode
    p.add_argument("--cat-mode", choices=["given", "predict"], default="given",
                   help="'given': condition on GT category label. "
                        "'predict': classify category internally and feed back.")



    # Transformer backbone
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--n-heads",    type=int,   default=4)
    p.add_argument("--n-layers",   type=int,   default=1)
    p.add_argument("--shared-dim", type=int,   default=512,
                   help="Shared latent dimension before the three heads.")
    p.add_argument("--dropout",    type=float, default=0.1)

    # Loss weights
    p.add_argument("--loss-weight-siglip",     type=float, default=1.0,
                   help="Weight for global SigLIP head loss.")
    p.add_argument("--loss-weight-siglip-obj", type=float, default=1.0,
                   help="Weight for object-crop SigLIP head loss.")
    p.add_argument("--loss-weight-clip",       type=float, default=1.0)
    p.add_argument("--uniformity-weight",      type=float, default=0.1)
    p.add_argument("--target-noise",           type=float, default=0.02,
                   help="Gaussian noise std added to all three targets (re-normalised after).")
    p.add_argument("--input-noise",            type=float, default=0.05,
                   help="Std of Gaussian noise added to neural firing rates during training.")
    p.add_argument("--neuron-dropout",         type=float, default=0.1,
                   help="Fraction of neurons randomly zeroed per sample during training.")
    p.add_argument("--nce-weight",             type=float, default=0.8,
                   help="Mix between InfoNCE and cosine in each head loss.")
    p.add_argument("--nce-temperature",        type=float, default=0.07,
                   help="InfoNCE softmax temperature.")

    # Optimisation
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac",  type=float, default=0.1)

    # Neuron subset ablation
    p.add_argument("--n-neurons",   type=int, default=None,
                   help="Number of neurons to randomly sample. None = use all.")
    p.add_argument("--neuron-seed", type=int, default=0,
                   help="RNG seed for neuron subset sampling.")

    # Brain-area subset
    p.add_argument("--area", type=str, default=None,
                   help="Restrict to neurons from one brain area, e.g. 'TE2', 'PRH'.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

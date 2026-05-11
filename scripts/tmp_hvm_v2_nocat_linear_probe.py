"""Category probes on HVM MultiHeadTransformerV2 no-category predictions.

This is a temporary analysis script for the sweep_hvm_multihead_v2 notebook.
It loads the best existing cat_mode=given model, extracts neural-predicted
neural embeddings while bypassing the learned category embedding, then fits a
classifier to predict HVM category from that neural embedding.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_const import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    HVM_CATEGORIES,
    HVM_N_CAT,
    HVM_N_STIMULI,
    HVM_N_VAR,
)
from data_utils.hvm_loader import _category_stratified_split, _load_hvm_neural  # noqa: E402
from encoders import MultiHeadTransformerV2  # noqa: E402
from encoders.multihead_v2 import CategoryClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit MultiHeadTransformerV2 checkpoint. Defaults to best given config/cache or sweep best.",
    )
    p.add_argument(
        "--feature",
        choices=["neural_embedding", "shared", "siglip_global", "siglip_obj", "clip", "all_heads"],
        default="neural_embedding",
        help="No-category feature used as linear-probe input. Default matches train_multihead_v2 post-hoc decoder.",
    )
    p.add_argument("--seed", type=int, default=0, help="Split/training seed. Notebook uses split seed 0.")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=250)
    p.add_argument(
        "--probe-arch",
        choices=["linear", "nonlinear", "mlp", "all"],
        default="all",
        help="'linear' is a pure linear probe; 'nonlinear' is Linear-GELU-Linear; 'mlp' matches model.cat_clf.",
    )
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="Hidden width for --probe-arch nonlinear.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_path(path: Path | None) -> Path:
    if path is not None:
        return path

    config_path = CACHE_DIR / "best_hvm_multihead_v2_given_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return Path(json.load(f)["checkpoint"])

    sweep_dir = CHECKPOINT_DIR / "multihead_v2" / "hvm_v2"
    best_path = None
    best_score = -float("inf")
    for pt in sorted(sweep_dir.glob("*/best.pt")):
        ckpt = torch.load(pt, weights_only=False, map_location="cpu")
        ckpt_args = ckpt.get("args", {})
        if ckpt_args.get("cat_mode") != "given":
            continue
        score = float(ckpt.get("metrics", {}).get("val_mean_cos", -float("inf")))
        if score > best_score:
            best_score = score
            best_path = pt

    if best_path is None:
        raise FileNotFoundError(f"No given-mode checkpoint found under {sweep_dir}")
    return best_path


def build_model(ckpt: dict, n_neurons: int) -> MultiHeadTransformerV2:
    ckpt_args = ckpt.get("args", {})
    model = MultiHeadTransformerV2(
        n_neurons=n_neurons,
        d_model=int(ckpt_args["d_model"]),
        n_heads=int(ckpt_args["n_heads"]),
        n_layers=int(ckpt_args["n_layers"]),
        shared_dim=int(ckpt_args["shared_dim"]),
        dropout=float(ckpt_args["dropout"]),
        n_categories=HVM_N_CAT,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_without_category(
    model: MultiHeadTransformerV2,
    neural: torch.Tensor,
    feature: str,
    batch_size: int = 128,
) -> torch.Tensor:
    """Extract features without adding model.cat_emb(...)."""
    feats = []
    device = next(model.parameters()).device

    for i in range(0, len(neural), batch_size):
        x = neural[i : i + batch_size].to(device).permute(0, 2, 1)  # (B, T, N)

        shared = model.encode_neural_embedding(x)

        outputs = {
            "neural_embedding": shared,
            "shared": F.normalize(shared, dim=-1),
            "siglip_global": F.normalize(model.siglip_global_head(shared), dim=-1),
            "siglip_obj": F.normalize(model.siglip_obj_head(shared), dim=-1),
            "clip": F.normalize(model.clip_head(shared), dim=-1),
        }
        if feature == "all_heads":
            out = torch.cat(
                [outputs["siglip_global"], outputs["siglip_obj"], outputs["clip"]],
                dim=-1,
            )
        else:
            out = outputs[feature]
        feats.append(out.cpu())

    return torch.cat(feats, dim=0)


def standardize_from_train(
    x: torch.Tensor,
    train_idx: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    mu = x[train_idx].mean(dim=0, keepdim=True)
    sd = x[train_idx].std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mu) / sd


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == y).float().mean().item()


def fit_linear_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> nn.Linear:
    device = torch.device(args.device)
    clf = nn.Linear(x.shape[1], HVM_N_CAT).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x = x.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    best_state = copy.deepcopy(clf.state_dict())
    best_val = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        clf.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(x[train_idx]), y[train_idx])
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            val_acc = accuracy(clf(x[val_idx]), y[val_idx])
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(clf.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch} (best val acc={best_val:.4f})")
                break

    clf.load_state_dict(best_state)
    clf.eval()
    return clf.cpu()


def fit_mlp_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> nn.Module:
    device = torch.device(args.device)
    clf = CategoryClassifier(
        in_dim=x.shape[1],
        n_categories=HVM_N_CAT,
        dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)

    x = x.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    best_state = copy.deepcopy(clf.state_dict())
    best_val = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        clf.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(x[train_idx]), y[train_idx])
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            val_acc = accuracy(clf(x[val_idx]), y[val_idx])
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(clf.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop MLP at epoch {epoch} (best val acc={best_val:.4f})")
                break

    clf.load_state_dict(best_state)
    clf.eval()
    return clf.cpu()


def fit_nonlinear_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
) -> nn.Module:
    device = torch.device(args.device)
    clf = nn.Sequential(
        nn.Linear(x.shape[1], args.hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(args.hidden_dim, HVM_N_CAT),
    ).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)

    x = x.to(device)
    y = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    best_state = copy.deepcopy(clf.state_dict())
    best_val = -1.0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        clf.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(x[train_idx]), y[train_idx])
        loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            val_acc = accuracy(clf(x[val_idx]), y[val_idx])
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(clf.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stop nonlinear at epoch {epoch} (best val acc={best_val:.4f})")
                break

    clf.load_state_dict(best_state)
    clf.eval()
    return clf.cpu()


def print_split_metrics(
    name: str,
    clf: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> None:
    with torch.no_grad():
        logits = clf(x)
    print(f"\n{name}:")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"{name:>5} acc: {accuracy(logits[idx], y[idx]):.4f}  (n={len(idx)})")

    pred = logits[test_idx].argmax(dim=-1)
    y_test = y[test_idx]
    print("Test per-category accuracy:")
    for ci, cat in enumerate(HVM_CATEGORIES):
        mask = y_test == ci
        acc = (pred[mask] == y_test[mask]).float().mean().item()
        print(f"  {ci:2d} {cat:>8}: {acc:.4f}  (n={int(mask.sum())})")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = load_checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    rsp, _, _ = _load_hvm_neural()
    neural = torch.from_numpy(rsp).float()
    y = torch.arange(HVM_N_STIMULI, dtype=torch.long) // HVM_N_VAR

    train_np, val_np, test_np = _category_stratified_split(seed=args.seed)
    train_idx = torch.from_numpy(train_np).long()
    val_idx = torch.from_numpy(val_np).long()
    test_idx = torch.from_numpy(test_np).long()

    model = build_model(ckpt, n_neurons=neural.shape[1]).to(args.device)
    x = predict_without_category(model, neural, feature=args.feature)
    x = standardize_from_train(x, train_idx)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Checkpoint cat_mode: {ckpt.get('args', {}).get('cat_mode')}")
    print(f"Feature: {args.feature} no-category feature, shape={tuple(x.shape)}")
    print("Important: features bypass model.forward() and never add model.cat_emb(category).")

    if args.probe_arch in ("linear", "all"):
        linear = fit_linear_probe(x, y, train_idx, val_idx, args)
        print_split_metrics("Linear probe", linear, x, y, train_idx, val_idx, test_idx)

    if args.probe_arch in ("nonlinear", "all"):
        nonlinear = fit_nonlinear_probe(x, y, train_idx, val_idx, args)
        print_split_metrics("Nonlinear probe", nonlinear, x, y, train_idx, val_idx, test_idx)

    if args.probe_arch in ("mlp", "all"):
        mlp = fit_mlp_probe(x, y, train_idx, val_idx, args)
        print_split_metrics("CategoryClassifier MLP probe", mlp, x, y, train_idx, val_idx, test_idx)


if __name__ == "__main__":
    main()

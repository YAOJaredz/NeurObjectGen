"""HVM identity-retrieval analysis for neural-predicted embeddings.

This script evaluates whether a neural decoder can retrieve the exact held-out
image in embedding space, including the critical within-category 2-AFC setting.
When called with ``--save``, it writes:

  - retrieval_metrics.csv/json: top-1, top-5, mean rank, all/within/cross 2-AFC
  - retrieval_ranks.csv: per-image retrieval ranks
  - nearest_neighbors.csv: nearest target indices for example inspection
  - figure_retrieval_identity.png: compact Figure 1 style summary

Example:
    python scripts/hvm_retrieval_analysis.py --checkpoint checkpoints/.../best.pt --no-category
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_const import (  # noqa: E402
    CACHE_DIR,
    CHECKPOINT_DIR,
    HVM_CATEGORIES,
    HVM_N_CAT,
    HVM_N_VAR,
    HVM_STIM_DIR,
)
from data_utils.hvm_loader import _category_stratified_split, make_hvm_multihead_loader  # noqa: E402
from encoders import MultiHeadTransformerV2  # noqa: E402
from eval.metrics import (  # noqa: E402
    category_aware_retrieval_metrics,
    retrieval_ranks,
)


HEAD_SPECS = {
    "siglip_global": ("Global SigLIP pred", "siglip", "siglip_global_head"),
    "siglip_obj": ("Object SigLIP pred", "siglip_obj", "siglip_obj_head"),
    "clip": ("CLIP pred", "clip_short", "clip_head"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="MultiHeadTransformerV2 checkpoint. Defaults to cache best or sweep best.")
    p.add_argument("--cat-mode", choices=["given", "predict"], default=None,
                   help="Forward mode for category-conditioned evaluation. Defaults to checkpoint args.")
    p.add_argument("--no-category", action="store_true",
                   help="Bypass category embeddings and evaluate neural-only identity information.")
    p.add_argument("--split-seed", type=int, default=None,
                   help="HVM split seed. Defaults to checkpoint args seed when available, else 42.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save", action="store_true",
                   help="Write CSV/JSON outputs and the figure PNG. Default is display/print only.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for --save. Defaults to checkpoint directory/retrieval_analysis.")
    p.add_argument("--topk", type=int, nargs="+", default=[1, 5],
                   help="Top-k values for N-way retrieval accuracy.")
    p.add_argument("--n-neighbors", type=int, default=5,
                   help="Nearest neighbors per query used for examples and optional CSV output.")
    p.add_argument("--n-examples", type=int, default=6,
                   help="Number of nearest-neighbor examples to draw in the figure.")
    return p.parse_args()


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
        score = float(ckpt.get("metrics", {}).get("val_mean_cos", -float("inf")))
        if score > best_score:
            best_score = score
            best_path = pt

    if best_path is None:
        raise FileNotFoundError(f"No checkpoint found under {sweep_dir}")
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
def predict_loader(
    model: MultiHeadTransformerV2,
    loader,
    device: torch.device,
    cat_mode: str,
    no_category: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    preds = {name: [] for name in HEAD_SPECS}
    targets = {target_key: [] for _, target_key, _ in HEAD_SPECS.values()}
    cats = []

    for neural, tgt, cat in loader:
        x = neural.permute(0, 2, 1).to(device)
        cat = cat.to(device)

        if no_category:
            shared = model.encode_neural_embedding(x)
            out = {
                "siglip_global": F.normalize(model.siglip_global_head(shared), dim=-1),
                "siglip_obj": F.normalize(model.siglip_obj_head(shared), dim=-1),
                "clip": F.normalize(model.clip_head(shared), dim=-1),
            }
        else:
            out = model(x, cat, cat_mode=cat_mode)

        for head_name in HEAD_SPECS:
            preds[head_name].append(out[head_name].cpu())
        for _, target_key, _ in HEAD_SPECS.values():
            targets[target_key].append(tgt[target_key].cpu())
        cats.append(cat.cpu())

    return (
        {k: torch.cat(v) for k, v in preds.items()},
        {k: torch.cat(v) for k, v in targets.items()},
        torch.cat(cats),
    )


def category_centroid_predictions(
    train_loader,
    test_categories: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sums: dict[str, list[torch.Tensor]] = {
        target_key: [None for _ in range(HVM_N_CAT)] for _, target_key, _ in HEAD_SPECS.values()
    }
    counts = torch.zeros(HVM_N_CAT, dtype=torch.float32)

    for _, tgt, cat in train_loader:
        for c in range(HVM_N_CAT):
            mask = cat == c
            if not mask.any():
                continue
            counts[c] += mask.sum()
            for _, target_key, _ in HEAD_SPECS.values():
                val = tgt[target_key][mask].sum(dim=0)
                sums[target_key][c] = val if sums[target_key][c] is None else sums[target_key][c] + val

    out = {}
    for target_key, per_cat in sums.items():
        centroids = torch.stack([per_cat[c] / counts[c].clamp_min(1.0) for c in range(HVM_N_CAT)])
        centroids = F.normalize(centroids, dim=-1)
        out[target_key] = centroids[test_categories]
    return out


def shuffled_predictions(
    pred: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pred), generator=generator)
    fixed = torch.arange(len(pred))
    if len(pred) > 1:
        while torch.any(perm == fixed):
            perm = torch.randperm(len(pred), generator=generator)
    return pred[perm]


def collect_metrics(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    cats: torch.Tensor,
    train_loader,
    topk: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    rank_rows = []
    nn_rows = []
    centroid_preds = category_centroid_predictions(train_loader, cats)

    variants: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []
    for head_name, (label, target_key, _) in HEAD_SPECS.items():
        variants.append((label, target_key, preds[head_name], targets[target_key]))
    for head_name, (label, target_key, _) in HEAD_SPECS.items():
        variants.append((f"{label} shuffled", target_key,
                         shuffled_predictions(preds[head_name], seed=1234), targets[target_key]))
    for _, target_key, _ in HEAD_SPECS.values():
        variants.append((f"{target_key} category-centroid baseline", target_key,
                         centroid_preds[target_key], targets[target_key]))

    for label, target_key, pred, target in variants:
        metrics = category_aware_retrieval_metrics(pred, target, cats, k=topk)
        rows.append({"representation": label, "target_space": target_key, **metrics})

        ranks = retrieval_ranks(pred, target).cpu().numpy()
        for i, rank in enumerate(ranks):
            rank_rows.append({
                "representation": label,
                "target_space": target_key,
                "test_row": i,
                "category": HVM_CATEGORIES[int(cats[i])],
                "rank": int(rank),
            })

        sim = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T
        nn = sim.argsort(dim=1, descending=True)[:, : min(sim.size(1), 10)]
        for i in range(len(pred)):
            for j, target_row in enumerate(nn[i, :]):
                nn_rows.append({
                    "representation": label,
                    "target_space": target_key,
                    "query_test_row": i,
                    "neighbor_rank": j + 1,
                    "target_test_row": int(target_row),
                    "is_correct": int(target_row) == i,
                    "query_category": HVM_CATEGORIES[int(cats[i])],
                    "target_category": HVM_CATEGORIES[int(cats[target_row])],
                    "similarity": float(sim[i, target_row].item()),
                })

    return pd.DataFrame(rows), pd.DataFrame(rank_rows), pd.DataFrame(nn_rows)


def stimulus_path_from_category_and_within_index(category: str, within_index: int) -> Path:
    return HVM_STIM_DIR / category / f"{within_index:02d}.png"


def test_row_to_stimulus_path(test_row: int, test_indices: np.ndarray) -> Path | None:
    stim_idx = int(test_indices[test_row])
    category = HVM_CATEGORIES[stim_idx // HVM_N_VAR]
    within_index = stim_idx % HVM_N_VAR
    path = stimulus_path_from_category_and_within_index(category, within_index)
    return path if path.exists() else None


def make_figure(
    metrics_df: pd.DataFrame,
    ranks_df: pd.DataFrame,
    nn_df: pd.DataFrame,
    cats: torch.Tensor,
    test_indices: np.ndarray,
    out_path: Path | None,
    n_examples: int,
) -> plt.Figure:
    main = metrics_df[~metrics_df["representation"].str.contains("shuffled|baseline")]
    order = [HEAD_SPECS[k][0] for k in HEAD_SPECS]

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    ax0.text(0.02, 0.65, "Neural response", fontsize=12, weight="bold")
    ax0.text(0.30, 0.65, "embedding decoder", fontsize=12, weight="bold")
    ax0.text(0.58, 0.65, "GT embedding library", fontsize=12, weight="bold")
    ax0.text(0.84, 0.65, "image identity", fontsize=12, weight="bold")
    for x0, x1 in [(0.18, 0.28), (0.45, 0.56), (0.74, 0.82)]:
        ax0.annotate("", xy=(x1, 0.68), xytext=(x0, 0.68),
                     arrowprops={"arrowstyle": "->", "lw": 2})
    ax0.text(0.30, 0.26, "global SigLIP / object SigLIP / CLIP", fontsize=11)
    ax0.set_title("A. Within-category retrieval removes category shortcuts", loc="left")

    ax1 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(order))
    width = 0.35
    all_vals = [float(main.loc[main["representation"] == name, "2afc_all"].iloc[0]) for name in order]
    within_vals = [float(main.loc[main["representation"] == name, "2afc_within_category"].iloc[0]) for name in order]
    ax1.bar(x - width / 2, all_vals, width, label="All distractors")
    ax1.bar(x + width / 2, within_vals, width, label="Within-category distractors")
    ax1.axhline(0.5, color="0.3", lw=1, ls="--", label="Chance")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=15, ha="right")
    ax1.set_ylabel("2-AFC accuracy")
    ax1.set_title("B. Image-identification 2-AFC", loc="left")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[1, 2])
    rank_data = [
        ranks_df.loc[ranks_df["representation"] == name, "rank"].to_numpy()
        for name in order
    ]
    ax2.violinplot(rank_data, showmeans=True, showextrema=False)
    ax2.set_xticks(np.arange(1, len(order) + 1))
    ax2.set_xticklabels(["global", "object", "CLIP"], rotation=15, ha="right")
    ax2.set_ylabel("Correct-image rank")
    ax2.invert_yaxis()
    ax2.set_title("C. Rank distribution", loc="left")

    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    example_repr = order[0]
    examples = nn_df[(nn_df["representation"] == example_repr) & (nn_df["neighbor_rank"] == 1)]
    examples = examples.assign(ok=examples["is_correct"].astype(bool))
    examples = examples.sort_values(["ok", "query_test_row"], ascending=[True, True]).head(n_examples)
    ax3.set_title("D. Example nearest-neighbor retrievals (query -> top match)", loc="left")

    if examples.empty:
        ax3.text(0.02, 0.5, "No examples available.", transform=ax3.transAxes)
    else:
        for col, (_, row) in enumerate(examples.iterrows()):
            q = int(row["query_test_row"])
            t = int(row["target_test_row"])
            q_path = test_row_to_stimulus_path(q, test_indices)
            t_path = test_row_to_stimulus_path(t, test_indices)
            x0 = 0.02 + col * (0.96 / max(1, len(examples)))
            for offset, path, title in [(0.0, q_path, "GT"), (0.075, t_path, "NN")]:
                inset = ax3.inset_axes([x0 + offset, 0.18, 0.065, 0.58])
                inset.set_xticks([])
                inset.set_yticks([])
                if path is not None:
                    inset.imshow(Image.open(path).convert("RGB"))
                inset.set_title(title, fontsize=8)
            marker = "hit" if row["is_correct"] else "miss"
            ax3.text(x0, 0.03, f"{row['query_category']} / {marker}", fontsize=8,
                     transform=ax3.transAxes)

    fig.suptitle("Neural embeddings recover image identity beyond category", fontsize=16, weight="bold")
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    return fig


def main() -> None:
    args = parse_args()
    ckpt_path = load_checkpoint_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    split_seed = args.split_seed
    if split_seed is None:
        split_seed = int(ckpt_args.get("seed", 42))
    cat_mode = args.cat_mode or ckpt_args.get("cat_mode", "given")
    area = ckpt_args.get("area") or "all"
    neuron_indices = ckpt.get("neuron_indices")

    train_loader, _, test_loader = make_hvm_multihead_loader(
        batch_size=args.batch_size,
        seed=split_seed,
        area=area,
        neuron_indices=neuron_indices,
    )

    sample_neural, _, _ = next(iter(test_loader))
    model = build_model(ckpt, n_neurons=sample_neural.shape[1]).to(args.device)
    preds, targets, cats = predict_loader(
        model,
        test_loader,
        device=torch.device(args.device),
        cat_mode=cat_mode,
        no_category=args.no_category,
    )

    metrics_df, ranks_df, nn_df = collect_metrics(preds, targets, cats, train_loader, args.topk)
    _, _, test_indices = _category_stratified_split(split_seed)

    keep_cols = [
        "representation", "top1", "top5", "mean_rank",
        "2afc_all", "2afc_within_category", "2afc_cross_category",
    ]
    print(metrics_df[keep_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.save:
        out_dir = args.out_dir or ckpt_path.parent / "retrieval_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_dir / "retrieval_metrics.csv", index=False)
        ranks_df.to_csv(out_dir / "retrieval_ranks.csv", index=False)
        nn_df.to_csv(out_dir / "nearest_neighbors.csv", index=False)

        with open(out_dir / "retrieval_metrics.json", "w") as f:
            json.dump({
                "checkpoint": str(ckpt_path),
                "cat_mode": cat_mode,
                "no_category": args.no_category,
                "split_seed": split_seed,
                "metrics": metrics_df.to_dict(orient="records"),
            }, f, indent=2)

        make_figure(
            metrics_df,
            ranks_df,
            nn_df[nn_df["neighbor_rank"] <= args.n_neighbors],
            cats,
            test_indices,
            out_dir / "figure_retrieval_identity.png",
            n_examples=args.n_examples,
        )
        plt.close("all")
        print(f"\nWrote retrieval analysis to {out_dir}")


if __name__ == "__main__":
    main()

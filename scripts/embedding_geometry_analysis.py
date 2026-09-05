"""Embedding-geometry analysis for HVM neural decoders.

Compares raw neural, no-category shared latent, predicted embedding, GT
embedding, and category geometries on the HVM test split.

By default this prints summaries only. Pass ``--save`` to write CSV/PNG outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append('.')

from config_const import CACHE_DIR, CHECKPOINT_DIR, HVM_CATEGORIES, HVM_N_CAT, HVM_TIME_WINDOW, REPO_ROOT  # noqa: E402
from data_utils.hvm_loader import _category_stratified_split, make_hvm_multihead_loader  # noqa: E402
from encoders import MultiHeadTransformer  # noqa: E402


SOURCE_REPS = [
    "raw_neural",
    "shared_global_latent",
    "shared_object_latent",
    "pred_global_siglip",
    "pred_object_siglip",
    "pred_clip_global_model",
    "pred_clip_object_model",
]
GT_REPS = [
    "gt_global_siglip",
    "gt_object_siglip",
    "gt_clip",
]
RDM_REPS = SOURCE_REPS + GT_REPS + ["category_rdm"]
REFERENCE_RDMS = GT_REPS + ["category_rdm"]
MAIN_COMPARISONS = [
    ("raw_neural", "gt_global_siglip"),
    ("raw_neural", "gt_object_siglip"),
    ("shared_global_latent", "gt_global_siglip"),
    ("shared_object_latent", "gt_object_siglip"),
    ("pred_global_siglip", "gt_global_siglip"),
    ("pred_object_siglip", "gt_object_siglip"),
    ("pred_clip_global_model", "gt_clip"),
    ("pred_clip_object_model", "gt_clip"),
]


@dataclass
class GeometryData:
    reps: dict[str, np.ndarray]
    categories: np.ndarray
    image_ids: np.ndarray
    global_checkpoint: Path
    object_checkpoint: Path
    split_seed: int
    raw_window_ms: tuple[int, int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--global-checkpoint", type=Path, default=None,
                   help="Global MultiHeadTransformer checkpoint. Defaults to cached best HVM multihead checkpoint.")
    p.add_argument("--object-checkpoint", type=Path, default=None,
                   help="Object MultiHeadTransformer checkpoint. Defaults to cached best HVM object-multihead checkpoint.")
    p.add_argument("--split-seed", type=int, default=None,
                   help="HVM split seed. Defaults to checkpoint args seed when present, else 42.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--raw-window-ms", type=int, nargs=2, default=[60, 200],
                   metavar=("START", "END"),
                   help="Raw neural averaging window in ms relative to stimulus onset.")
    p.add_argument("--save", action="store_true",
                   help="Write CSV summaries and PNG figures. Default prints only.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "embedding_geometry",
                   help="Output directory used only with --save.")
    p.add_argument("--self-test", action="store_true",
                   help="Run synthetic checks and exit.")
    return p.parse_args()


def load_best_checkpoint_path(path: Path | None, config_name: str, subdir: Path) -> Path:
    if path is not None:
        return path

    config_path = CACHE_DIR / config_name
    if config_path.exists():
        with open(config_path) as f:
            return Path(json.load(f)["checkpoint"])

    best_path = None
    best_score = -float("inf")
    for pt in sorted(subdir.glob("*/best.pt")):
        ckpt = torch.load(pt, weights_only=False, map_location="cpu")
        score = float(ckpt.get("metrics", {}).get("val_mean_cos", -float("inf")))
        if score > best_score:
            best_score = score
            best_path = pt
    if best_path is None:
        raise FileNotFoundError(f"No checkpoint found under {subdir}")
    return best_path


def build_multihead_model(ckpt: dict, n_neurons: int) -> MultiHeadTransformer:
    ckpt_args = ckpt.get("args", {})
    use_category = bool(ckpt_args.get("use_category", False))
    model = MultiHeadTransformer(
        n_neurons=n_neurons,
        d_model=int(ckpt_args["d_model"]),
        n_heads=int(ckpt_args["n_heads"]),
        n_layers=int(ckpt_args["n_layers"]),
        shared_dim=int(ckpt_args["shared_dim"]),
        dropout=float(ckpt_args["dropout"]),
        n_categories=HVM_N_CAT if use_category else 0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def validate_matching_inputs(global_ckpt: dict, object_ckpt: dict) -> tuple[str, np.ndarray | None]:
    global_args = global_ckpt.get("args", {})
    object_args = object_ckpt.get("args", {})
    global_area = global_args.get("area") or "all"
    object_area = object_args.get("area") or "all"
    if global_area != object_area:
        raise ValueError(f"Global/object checkpoints use different areas: {global_area!r} vs {object_area!r}.")

    global_idx = global_ckpt.get("neuron_indices")
    object_idx = object_ckpt.get("neuron_indices")
    if global_idx is None and object_idx is None:
        return global_area, None
    if global_idx is None or object_idx is None:
        raise ValueError("Global/object checkpoints use different neuron subsets.")
    global_idx = np.asarray(global_idx)
    object_idx = np.asarray(object_idx)
    if not np.array_equal(global_idx, object_idx):
        raise ValueError("Global/object checkpoints use different neuron subsets.")
    return global_area, global_idx


def raw_window_to_slice(raw_window_ms: tuple[int, int], n_time: int) -> slice:
    start_ms, end_ms = raw_window_ms
    response_start_ms, response_end_ms = HVM_TIME_WINDOW
    bin_ms = (response_end_ms - response_start_ms) / n_time
    start_idx = int(round((start_ms - response_start_ms) / bin_ms))
    end_idx = int(round((end_ms - response_start_ms) / bin_ms))
    if start_idx < 0 or end_idx > n_time or start_idx >= end_idx:
        raise ValueError(
            f"raw-window-ms {raw_window_ms} maps to invalid bins "
            f"{start_idx}:{end_idx} for HVM_TIME_WINDOW={HVM_TIME_WINDOW}, n_time={n_time}."
        )
    return slice(start_idx, end_idx)


@torch.no_grad()
def collect_geometry_data(args: argparse.Namespace) -> GeometryData:
    global_ckpt_path = load_best_checkpoint_path(
        args.global_checkpoint,
        "best_hvm_multihead_config.json",
        CHECKPOINT_DIR / "multihead" / "hvm_cat",
    )
    object_ckpt_path = load_best_checkpoint_path(
        args.object_checkpoint,
        "best_hvm_multihead_obj_config.json",
        CHECKPOINT_DIR / "multihead" / "hvm_obj_cat",
    )
    global_ckpt = torch.load(global_ckpt_path, weights_only=False, map_location="cpu")
    object_ckpt = torch.load(object_ckpt_path, weights_only=False, map_location="cpu")

    split_seed = args.split_seed
    if split_seed is None:
        split_seed = int(global_ckpt.get("args", {}).get("seed", object_ckpt.get("args", {}).get("seed", 42)))
    area, neuron_indices = validate_matching_inputs(global_ckpt, object_ckpt)

    _, _, test_idx = _category_stratified_split(split_seed)
    _, _, test_loader = make_hvm_multihead_loader(
        batch_size=args.batch_size,
        seed=split_seed,
        area=area,
        neuron_indices=neuron_indices,
    )

    sample_neural, _, _ = next(iter(test_loader))
    raw_slice = raw_window_to_slice(tuple(args.raw_window_ms), sample_neural.shape[2])
    global_model = build_multihead_model(global_ckpt, n_neurons=sample_neural.shape[1]).to(args.device)
    object_model = build_multihead_model(object_ckpt, n_neurons=sample_neural.shape[1]).to(args.device)
    global_model.eval()
    object_model.eval()

    reps = {name: [] for name in SOURCE_REPS + GT_REPS}
    cats = []

    for neural, tgt, cat in test_loader:
        raw = neural[:, :, raw_slice].mean(dim=2)
        x = neural.permute(0, 2, 1).to(args.device)
        global_out = global_model(x, cat=None)
        object_out = object_model(x, cat=None)

        reps["raw_neural"].append(raw.cpu().numpy())
        reps["shared_global_latent"].append(global_out["shared"].cpu().numpy())
        reps["shared_object_latent"].append(object_out["shared"].cpu().numpy())
        reps["pred_global_siglip"].append(global_out["siglip"].cpu().numpy())
        reps["pred_object_siglip"].append(object_out["siglip"].cpu().numpy())
        reps["pred_clip_global_model"].append(global_out["clip"].cpu().numpy())
        reps["pred_clip_object_model"].append(object_out["clip"].cpu().numpy())
        reps["gt_global_siglip"].append(tgt["siglip"].cpu().numpy())
        reps["gt_object_siglip"].append(tgt["siglip_obj"].cpu().numpy())
        reps["gt_clip"].append(tgt["clip_short"].cpu().numpy())
        cats.append(cat.cpu().numpy())

    reps_np = {name: np.concatenate(parts, axis=0) for name, parts in reps.items()}
    categories = np.concatenate(cats, axis=0).astype(int)
    return GeometryData(
        reps=reps_np,
        categories=categories,
        image_ids=test_idx.astype(int),
        global_checkpoint=global_ckpt_path,
        object_checkpoint=object_ckpt_path,
        split_seed=split_seed,
        raw_window_ms=tuple(args.raw_window_ms),
    )


def cosine_rdm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.clip(norm, eps, None)
    sim = np.clip(x_norm @ x_norm.T, -1.0, 1.0)
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm


def category_rdm(categories: np.ndarray) -> np.ndarray:
    return (categories[:, None] != categories[None, :]).astype(float)


def upper_triangle_values(rdm: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(rdm.shape[0], k=1)
    return rdm[idx]


def same_category_upper_mask(categories: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(len(categories), k=1)
    return categories[idx[0]] == categories[idx[1]]


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    a = a[valid]
    b = b[valid]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def compute_rdms(data: GeometryData) -> dict[str, np.ndarray]:
    rdms = {name: cosine_rdm(data.reps[name]) for name in SOURCE_REPS + GT_REPS}
    rdms["category_rdm"] = category_rdm(data.categories)
    return rdms


def compute_rsa_tables(
    rdms: dict[str, np.ndarray],
    categories: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    upper = {name: upper_triangle_values(rdm) for name, rdm in rdms.items()}
    within_mask = same_category_upper_mask(categories)
    all_rows = []
    within_rows = []

    for source in RDM_REPS:
        for reference in REFERENCE_RDMS:
            all_r = spearman_safe(upper[source], upper[reference])
            within_r = spearman_safe(upper[source][within_mask], upper[reference][within_mask])
            is_main = (source, reference) in MAIN_COMPARISONS
            all_rows.append({
                "pair_scope": "all_pairs",
                "source": source,
                "reference": reference,
                "spearman_r": all_r,
                "is_main_comparison": is_main,
            })
            within_rows.append({
                "pair_scope": "within_category",
                "source": source,
                "reference": reference,
                "spearman_r": within_r,
                "is_main_comparison": is_main,
            })

    all_df = pd.DataFrame(all_rows)
    within_df = pd.DataFrame(within_rows)
    summary_df = pd.concat([all_df, within_df], ignore_index=True)
    all_matrix = all_df.pivot(index="source", columns="reference", values="spearman_r").reindex(
        index=RDM_REPS, columns=REFERENCE_RDMS
    )
    within_matrix = within_df.pivot(index="source", columns="reference", values="spearman_r").reindex(
        index=RDM_REPS, columns=REFERENCE_RDMS
    )
    return summary_df, all_df, all_matrix, within_matrix


def nearest_neighbors(x: np.ndarray, max_k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :max_k]


def overlap_at_k(source_nn: np.ndarray, reference_nn: np.ndarray, k: int) -> float:
    overlaps = []
    for src, ref in zip(source_nn[:, :k], reference_nn[:, :k]):
        overlaps.append(len(set(src.tolist()) & set(ref.tolist())) / k)
    return float(np.mean(overlaps))


def compute_nn_overlap(
    data: GeometryData,
    ks: tuple[int, ...] = (1, 5, 10),
    shuffle_seed: int = 1234,
) -> pd.DataFrame:
    max_k = max(ks)
    reference_names = ["gt_global_siglip", "gt_object_siglip"]
    reference_nn = {name: nearest_neighbors(data.reps[name], max_k) for name in reference_names}

    rows = []
    rng = np.random.default_rng(shuffle_seed)
    for source in SOURCE_REPS:
        variants = [(source, data.reps[source], False)]
        variants.append((f"{source}_shuffled", data.reps[source][rng.permutation(len(data.categories))], True))
        for label, values, shuffled in variants:
            src_nn = nearest_neighbors(values, max_k)
            for ref in reference_names:
                for k in ks:
                    rows.append({
                        "source": label,
                        "source_base": source,
                        "reference": ref,
                        "k": k,
                        "overlap": overlap_at_k(src_nn, reference_nn[ref], k),
                        "is_shuffled": shuffled,
                    })
    return pd.DataFrame(rows)


def plot_heatmap(matrix: pd.DataFrame, title: str, out_path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = values[i, j]
            text = "nan" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman r")
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    return fig


def plot_nn_overlap(nn_df: pd.DataFrame, reference: str, title: str, out_path: Path | None = None) -> plt.Figure:
    df = nn_df[nn_df["reference"] == reference].copy()
    sources = SOURCE_REPS
    ks = sorted(df["k"].unique())
    x = np.arange(len(sources))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for offset_idx, k in enumerate(ks):
        vals = [
            float(df[(df["source"] == source) & (df["k"] == k)]["overlap"].iloc[0])
            for source in sources
        ]
        ax.bar(x + (offset_idx - (len(ks) - 1) / 2) * width, vals, width, label=f"@{k}")

    for source_idx, source in enumerate(sources):
        shuffled = df[(df["source"] == f"{source}_shuffled") & (df["k"] == max(ks))]
        if not shuffled.empty:
            ax.plot(
                source_idx,
                float(shuffled["overlap"].iloc[0]),
                marker="x",
                color="black",
                ms=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean neighbor overlap")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.text(0.99, 0.98, "x = shuffled overlap@10", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
    return fig


def run_self_test() -> None:
    cats = np.array([0, 0, 1, 1])
    x = np.array([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])
    y = x.copy()
    rdms = {"x": cosine_rdm(x), "y": cosine_rdm(y), "category_rdm": category_rdm(cats)}
    assert abs(spearman_safe(upper_triangle_values(rdms["x"]), upper_triangle_values(rdms["y"])) - 1.0) < 1e-9
    within_mask = same_category_upper_mask(cats)
    assert within_mask.sum() == 2
    nn_x = nearest_neighbors(x, 2)
    nn_y = nearest_neighbors(y, 2)
    assert overlap_at_k(nn_x, nn_y, 1) == 1.0
    shuffled = y[[2, 3, 0, 1]]
    assert overlap_at_k(nn_x, nearest_neighbors(shuffled, 2), 1) < 1.0
    print("Self-test passed.")


def print_key_tables(rsa_summary: pd.DataFrame, nn_summary: pd.DataFrame) -> None:
    main = rsa_summary[rsa_summary["is_main_comparison"]].copy()
    print("\nMain RSA comparisons:")
    print(main.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    nn_main = nn_summary[
        (~nn_summary["is_shuffled"])
        & (nn_summary["reference"].isin(["gt_global_siglip", "gt_object_siglip"]))
    ].copy()
    print("\nNearest-neighbor preservation:")
    print(nn_main.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    data = collect_geometry_data(args)
    rdms = compute_rdms(data)
    rsa_summary, _, rsa_all_matrix, rsa_within_matrix = compute_rsa_tables(rdms, data.categories)
    nn_summary = compute_nn_overlap(data)

    print(f"global_checkpoint: {data.global_checkpoint}")
    print(f"object_checkpoint: {data.object_checkpoint}")
    print(f"split_seed: {data.split_seed} | raw_window_ms: {data.raw_window_ms}")
    print_key_tables(rsa_summary, nn_summary)

    if args.save:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        rsa_summary.to_csv(args.out_dir / "rsa_summary.csv", index=False)
        nn_summary.to_csv(args.out_dir / "nn_overlap_summary.csv", index=False)
        plot_heatmap(
            rsa_all_matrix,
            "RSA: all image pairs",
            args.out_dir / "rsa_heatmap_all_pairs.png",
        )
        plot_heatmap(
            rsa_within_matrix,
            "RSA: within-category image pairs",
            args.out_dir / "rsa_heatmap_within_category.png",
        )
        plot_nn_overlap(
            nn_summary,
            "gt_global_siglip",
            "Neighbor preservation vs GT global SigLIP",
            args.out_dir / "nn_overlap_global_siglip.png",
        )
        plot_nn_overlap(
            nn_summary,
            "gt_object_siglip",
            "Neighbor preservation vs GT object SigLIP",
            args.out_dir / "nn_overlap_object_siglip.png",
        )
        plt.close("all")
        print(f"\nWrote embedding-geometry outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

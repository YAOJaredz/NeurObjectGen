import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

from HexPred.object_response.get_hvm_response import get_hvm_responses, get_hvm_category_vector
from HexPred.constants import ALL_MONKEYS

from config_const import (
    SEED,
    HVM_SIGLIP_EMBEDDINGS_PATH, HVM_CLIP_EMBEDS_PATH,
    HVM_TIME_WINDOW, HVM_N_STIMULI, HVM_N_TRAIN, HVM_N_VAL, HVM_N_VAR,
)


def _load_hvm_neural() -> np.ndarray:
    """Return (450, neurons, time) mean-trial response across all monkeys, dead neurons dropped."""
    monkey_responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_hvm_responses(mode='area', monkey=monkey, area='all',
                                   time_window=HVM_TIME_WINDOW)
        monkey_responses.append(rsp)
    rsp = np.concatenate(monkey_responses, axis=1)  # (450, all_neurons, time)
    dead = (
        np.all((rsp == 0) | np.isnan(rsp), axis=(0, 2))
        | np.any(np.isnan(rsp), axis=(0, 2))
    )
    rsp = rsp[:, ~dead, :]
    assert not np.isnan(rsp).any(), "NaN values remain after neuron filtering."
    assert rsp.shape[0] == HVM_N_STIMULI, (
        f"expected {HVM_N_STIMULI} stimuli, got {rsp.shape[0]}"
    )
    return rsp


def _category_stratified_split(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test indices stratified by HVM category.

    HVM has 10 categories × 45 variations = 450 stimuli. We split within each
    category so every split sees all categories: 27 train / 9 val / 9 test per
    category (totals: 270 / 90 / 90).
    """
    cats = get_hvm_category_vector()           # (450,) str
    unique_cats = np.unique(cats)
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    n_var = 45
    n_val = n_test = 9                          # 9+9+27 = 45

    for cat in unique_cats:
        idx = np.where(cats == cat)[0]
        assert len(idx) == n_var
        perm = rng.permutation(idx)
        train_idx.append(perm[n_val + n_test:])
        val_idx.append(perm[:n_val])
        test_idx.append(perm[n_val:n_val + n_test])

    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def make_hvm_loader(
    batch_size: int = 64,
    seed: int = SEED,
    verbose: bool = True,
    use_embeddings: bool = False,
    target: str = 'siglip',
    use_categories: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build category-stratified train/val/test DataLoaders for HVM data.

    HVM has 10 object categories × 45 variations = 450 stimuli. Neural responses
    are concatenated across all monkeys (same as make_rust_loader). The split is
    stratified so each partition sees every category: 27 train / 9 val / 9 test
    per category (270 / 90 / 90 total).

    Each sample is ``(neural_response, target)`` where ``neural_response`` has
    shape ``(neurons, time)`` and ``target`` is an embedding of shape ``(D,)``
    (requires pre-cached embeddings) or a raw image ``(3, 224, 224)``.

    Args:
        batch_size:     Samples per batch.
        seed:           RNG seed for within-category permutation.
        verbose:        Print loader statistics.
        use_embeddings: Yield pre-cached embeddings instead of images.
        target:         Which embedding — 'siglip' or 'clip'.
        use_categories: If True, batches are 3-tuples (neural, target, cat_idx).

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    rsp = _load_hvm_neural()
    neural_tensor = torch.from_numpy(rsp).float()  # (450, neurons, time)

    if use_embeddings:
        _EMBED_PATHS = {
            'siglip': (HVM_SIGLIP_EMBEDDINGS_PATH, 'python scripts/cache_siglip.py --dataset hvm'),
            'clip':   (HVM_CLIP_EMBEDS_PATH,        'python scripts/cache_text_embeds.py --dataset hvm'),
        }
        if target not in _EMBED_PATHS:
            raise ValueError(f"Unknown target '{target}'. Choose from: {list(_EMBED_PATHS)}")
        embed_path, hint = _EMBED_PATHS[target]
        if not embed_path.exists():
            raise FileNotFoundError(f"{target} cache not found at {embed_path}. Run: {hint}")
        target_tensor = torch.load(embed_path, weights_only=True)  # (N, D)
        if target_tensor.shape[0] != HVM_N_STIMULI:
            raise ValueError(
                f"Embedding cache has {target_tensor.shape[0]} rows, expected {HVM_N_STIMULI}. "
                "Re-run the caching script against HVM stimuli."
            )
    else:
        from data_utils.stimuli import load_hvm_stimuli
        target_tensor = load_hvm_stimuli()  # (450, 3, 224, 224)

    train_idx_np, val_idx_np, test_idx_np = _category_stratified_split(seed)
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx   = torch.from_numpy(val_idx_np).long()
    test_idx  = torch.from_numpy(test_idx_np).long()

    # Sequential block layout: indices 0–44 = cat 0, 45–89 = cat 1, …, 405–449 = cat 9
    cat_indices = torch.arange(HVM_N_STIMULI) // HVM_N_VAR  # (450,) int64

    def make(idx, shuffle):
        if use_categories:
            ds = TensorDataset(neural_tensor[idx], target_tensor[idx], cat_indices[idx])
        else:
            ds = TensorDataset(neural_tensor[idx], target_tensor[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    if verbose:
        target_shape = tuple(target_tensor.shape[1:])
        print(
            f"HVM loaders created (all monkeys): "
            f"train={len(train_idx)}, "
            f"val={len(val_idx)}, "
            f"test={len(test_idx)}, "
            f"neurons={rsp.shape[1]}, "
            f"time={rsp.shape[2]}, "
            f"target={target_shape} "
            f"[stratified by category]"
        )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    make_hvm_loader(use_embeddings=False)

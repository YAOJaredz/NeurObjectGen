import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

from HexPred.object_response.get_rust_response import get_rust_responses
# from HexPred.object_response.get_hvm_response import get_hvm_responses
from HexPred.constants import ALL_MONKEYS

from config_const import (
    N_STIMULI, N_TRAIN, N_VAL, RUST_TIME_WINDOW, SEED,
    SIGLIP_EMBEDDINGS_PATH,
    CLIP_EMBEDS_PATH, CLIP_DETAILED_EMBEDS_PATH,
    T5_EMBEDS_PATH, T5_PCA_K, CACHE_DIR,
)
from data_utils.stimuli import load_rust_stimuli


def make_rust_loader(
    batch_size: int = 64,
    seed: int = SEED,
    verbose: bool = True,
    use_embeddings: bool = False,
    target: str = "siglip",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for neural-response → image reconstruction.

    Loads per-area spike responses for every monkey in ``ALL_MONKEYS`` over the
    0-250 ms post-stimulus window, concatenates them along the neuron axis,
    drops neurons whose responses are entirely zero or NaN across all stimuli
    and time bins, and splits the 300 stimuli into fixed 200/50/50
    train/val/test partitions using a seeded permutation.

    Each sample yielded by the loaders is ``(neural_response, target)`` where
    ``neural_response`` has shape ``(neurons, time)`` and ``target`` is either
    an image of shape ``(3, 224, 224)`` or a pre-cached SigLIP embedding of
    shape ``(D,)`` depending on ``use_embeddings``.

    Args:
        batch_size:      Number of stimuli per batch. Defaults to 64.
        seed:            Seed for the NumPy RNG used to permute stimulus indices.
        verbose:         Print loader statistics.
        use_embeddings:  If True, yield pre-cached SigLIP embeddings as targets
                         instead of raw images. Requires the cache at
                         SIGLIP_EMBEDDINGS_PATH (run scripts/cache_siglip.py).

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)`` of PyTorch
        ``DataLoader`` objects. The training loader is shuffled; the
        validation and test loaders preserve stimulus order.
    """
    monkey_responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_rust_responses(
            mode='area', area='all', monkey=monkey, time_window=RUST_TIME_WINDOW
        )
        monkey_responses.append(rsp)

    # stimuli x neurons x time
    object_responses = np.concatenate(monkey_responses, axis=1)

    # drop neurons that are all-zero or contain any NaN across stimuli and time
    dead_neuron = (
        np.all((object_responses == 0) | np.isnan(object_responses), axis=(0, 2))
        | np.any(np.isnan(object_responses), axis=(0, 2))
    )
    object_responses = object_responses[:, ~dead_neuron, :]
    # replace any remaining zeros with zero (no-op) and verify clean
    assert not np.isnan(object_responses).any(), "NaN values remain after neuron filtering."

    assert object_responses.shape[0] == N_STIMULI, (
        f"expected {N_STIMULI} stimuli, got {object_responses.shape[0]}"
    )

    neural_tensor = torch.from_numpy(object_responses).float()  # (N_STIMULI, neurons, time)

    if use_embeddings:
        _EMBED_PATHS = {
            "siglip": (SIGLIP_EMBEDDINGS_PATH,   "python scripts/cache_siglip.py"),
            "clip":   (CLIP_DETAILED_EMBEDS_PATH, "python scripts/cache_text_embeds.py --device cuda"),
        }
        if target not in _EMBED_PATHS:
            raise ValueError(f"Unknown target '{target}'. Choose from: {list(_EMBED_PATHS)}")
        embed_path, hint = _EMBED_PATHS[target]
        if not embed_path.exists():
            raise FileNotFoundError(f"{target} cache not found at {embed_path}. Run: {hint}")
        target_tensor = torch.load(embed_path, weights_only=True)  # (N_STIMULI, D)
    else:
        target_tensor = load_rust_stimuli()  # (N_STIMULI, 3, 224, 224)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_STIMULI)
    train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
    val_idx = torch.from_numpy(perm[N_TRAIN:N_TRAIN + N_VAL]).long()
    test_idx = torch.from_numpy(perm[N_TRAIN + N_VAL:]).long()

    def make(idx, shuffle):
        return DataLoader(
            TensorDataset(neural_tensor[idx], target_tensor[idx]),
            batch_size=batch_size, shuffle=shuffle,
        )

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    if verbose:
        target_shape = tuple(target_tensor.shape[1:])
        print(
            f"RUST loaders created: "
            f"train={train_idx.shape[0]}, "
            f"val={val_idx.shape[0]}, "
            f"test={test_idx.shape[0]}, "
            f"neurons={object_responses.shape[1]}, "
            f"time={object_responses.shape[2]}, "
            f"target={target_shape}"
        )

    return train_loader, val_loader, test_loader


class MultiHeadDataset(Dataset):
    """Dataset yielding (neural, target_dict) for multi-head training."""

    def __init__(self, neural: torch.Tensor, siglip: torch.Tensor, clip: torch.Tensor, t5_pca: torch.Tensor):
        self.neural  = neural
        self.targets = {"siglip": siglip, "clip": clip, "t5_pca": t5_pca}

    def __len__(self) -> int:
        return len(self.neural)

    def __getitem__(self, i: int):
        return self.neural[i], {k: v[i] for k, v in self.targets.items()}


def make_multihead_loader(
    batch_size: int = 64,
    seed: int = SEED,
    verbose: bool = True,
    t5_pca_k: int = T5_PCA_K,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for multi-head neural encoding.

    Yields (neural, target_dict) where target_dict has keys:
      - "siglip":  (1152,) L2-normalised SigLIP embedding
      - "clip":    (768,)  L2-normalised CLIP embedding (short captions)
      - "t5_pca":  (K,)    T5 PCA coordinates (short captions)

    Neural data loading and split permutation are identical to make_rust_loader
    to ensure test sets are directly comparable with single-head baselines.
    """
    t5_pca_basis_path = CACHE_DIR / f"t5_pca_basis_k{t5_pca_k}.pt"
    t5_pca_mean_path  = CACHE_DIR / f"t5_pca_mean_k{t5_pca_k}.pt"

    if not t5_pca_basis_path.exists():
        raise FileNotFoundError(
            f"T5 PCA basis not found at {t5_pca_basis_path}. "
            "Run: python scripts/precompute_t5_pca.py"
        )

    monkey_responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_rust_responses(
            mode='area', area='all', monkey=monkey, time_window=RUST_TIME_WINDOW
        )
        monkey_responses.append(rsp)

    object_responses = np.concatenate(monkey_responses, axis=1)

    dead_neuron = (
        np.all((object_responses == 0) | np.isnan(object_responses), axis=(0, 2))
        | np.any(np.isnan(object_responses), axis=(0, 2))
    )
    object_responses = object_responses[:, ~dead_neuron, :]
    assert not np.isnan(object_responses).any()
    assert object_responses.shape[0] == N_STIMULI

    neural_tensor = torch.from_numpy(object_responses).float()  # (300, neurons, time)

    siglip_t = torch.load(SIGLIP_EMBEDDINGS_PATH, weights_only=True)        # (300, 1152) already L2-normed

    clip_t = F.normalize(torch.load(CLIP_EMBEDS_PATH, weights_only=True), dim=-1)  # (300, 768) short captions

    t5_basis = torch.load(t5_pca_basis_path, weights_only=True).float()   # (K, 4096)
    t5_mean  = torch.load(t5_pca_mean_path,  weights_only=True).float()   # (4096,)
    t5_short_raw = torch.load(T5_EMBEDS_PATH, weights_only=True).float()  # (300, 512, 4096)
    norms = t5_short_raw.norm(dim=-1)                                      # (300, 512)
    weights = torch.softmax(norms, dim=-1).unsqueeze(-1)                   # (300, 512, 1)
    t5_short_pooled = (weights * t5_short_raw).sum(dim=1) - t5_mean       # (300, 4096)
    t5_pca_t = t5_short_pooled @ t5_basis.T                               # (300, K) short captions

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_STIMULI)
    train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
    val_idx   = torch.from_numpy(perm[N_TRAIN:N_TRAIN + N_VAL]).long()
    test_idx  = torch.from_numpy(perm[N_TRAIN + N_VAL:]).long()

    def make(idx, shuffle):
        ds = MultiHeadDataset(
            neural_tensor[idx],
            siglip_t[idx],
            clip_t[idx],
            t5_pca_t[idx],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    if verbose:
        k = t5_pca_t.shape[1]
        print(
            f"Multihead RUST loaders: "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
            f"neurons={object_responses.shape[1]}, time={object_responses.shape[2]}, "
            f"targets=(siglip=1152, clip=768, t5_pca={k})"
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    make_rust_loader()
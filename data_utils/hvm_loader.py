import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

from HexPred.object_response.get_hvm_response import get_hvm_responses, get_hvm_category_vector
from HexPred.constants import ALL_MONKEYS
from HexPred.stim_responses.get_stim_response import get_target_stim_channel_func, build_channel_mask
from HexPred.stim_responses.channels_area_target import (
    convert_channel_range, BRAIN_MAP_PATH, BRAIN_MAP_MONKEY_KEY,
)

from config_const import (
    SEED,
    HVM_SIGLIP_EMBEDDINGS_PATH, HVM_CLIP_EMBEDS_PATH,
    HVM_TIME_WINDOW, HVM_N_STIMULI, HVM_N_VAL, HVM_N_VAR,
)


def _session_channel_area_map(session: str, monkey: str) -> dict[int, str]:
    """Return {raw_channel_idx: area_name} for one session using the brain-map Excel file."""
    mapping = (
        pd.read_excel(BRAIN_MAP_PATH, sheet_name=BRAIN_MAP_MONKEY_KEY[monkey])
        .dropna(subset=['penetration'])
    )
    mapping = mapping[mapping['date'] != 'no data available']
    all_areas = ['TE0', 'TE2', 'TE3', 'PHC', 'ENT', 'PRH', 'V2', 'V3', 'WM', 'HC', 'Ventricle']
    all_penetrations = {
        tuple(x.replace(' ', '').split(',')): i
        for i, x in enumerate(mapping['penetration'])
    }
    m = re.search(r'(H\d+)_(P\d+)', session)
    if not m:
        return {}
    h, p = m.group(1), m.group(2)
    if (h, p) not in all_penetrations:
        return {}
    row = mapping.iloc[all_penetrations[(h, p)]]
    ch_to_area: dict[int, str] = {}
    for area in all_areas:
        if area not in row.index:
            continue
        for ch in convert_channel_range(row[area]):
            ch_to_area[int(ch)] = area
    return ch_to_area


def get_neuron_map(
    dead_mask: np.ndarray,
    monkey_sessions: dict[str, list[str]],
) -> list[dict]:
    """Return a mapping from each neuron in the final vector to its origin.

    Uses the session lists already returned by ``_load_hvm_neural`` (second return
    value per monkey) so no additional data loading is needed.  Applies the same
    ``valid_channels_only=True`` mask and dead-neuron filter as ``_load_hvm_neural``.

    Each entry corresponds to one neuron in ``_load_hvm_neural``'s output (axis=1)
    and contains:

        monkey     - str, e.g. 'West' or 'Bourgeois'
        session    - str, full session name
        local_idx  - int, 0-based channel index within the raw 384-channel array
                     for that session (before validity filtering)
        area       - str, brain area label (e.g. 'TE2', 'V3'); 'unknown' if the
                     channel falls outside all labelled ranges in the brain-map file

    Args:
        dead_mask:       Boolean array of shape ``(N_valid_channels_across_all_monkeys,)``
                         where True marks all-zero / NaN neurons dropped by
                         ``_load_hvm_neural``.
        monkey_sessions: Dict mapping each monkey name to the sessions list returned
                         by the corresponding ``get_hvm_responses(mode='area', ...)``
                         call inside ``_load_hvm_neural``, e.g.
                         ``{'West': [...], 'Bourgeois': [...]}``.

    Returns:
        List of dicts, one per surviving neuron, in the same order as axis=1 of
        ``_load_hvm_neural``'s return value.
    """
    entries: list[dict] = []

    for monkey in ALL_MONKEYS:
        sessions = monkey_sessions[monkey]
        channel_func = get_target_stim_channel_func('all', monkey=monkey)

        valid_sessions: list[str] = []
        per_session_channels: list[np.ndarray] = []
        for session in sessions:
            try:
                channels = channel_func(session)
            except Exception:
                continue
            valid_sessions.append(session)
            per_session_channels.append(channels)

        n_total = sum(len(ch) for ch in per_session_channels)
        valid_mask = build_channel_mask(
            valid_sessions, 'all', monkey,
            experiment_id='object',
            valid_channels_only=True,
            n_channels=n_total,
        )

        offset = 0
        for session, channels in zip(valid_sessions, per_session_channels):
            n_ch = len(channels)
            sess_valid = valid_mask[offset : offset + n_ch]
            sess_area  = _session_channel_area_map(session, monkey)
            for local_idx, keep in enumerate(sess_valid):
                if keep:
                    raw_ch = int(channels[local_idx])
                    entries.append({
                        'monkey':    monkey,
                        'session':   session,
                        'local_idx': local_idx,
                        'area':      sess_area.get(raw_ch, 'unknown'),
                    })
            offset += n_ch

    assert len(entries) == len(dead_mask), (
        f"Entry count {len(entries)} != dead_mask length {len(dead_mask)}."
    )
    return [e for e, dead in zip(entries, dead_mask) if not dead]


def _load_hvm_neural() -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Load HVM neural responses across all monkeys.

    Returns:
        rsp:             (450, neurons, time) mean-trial response, dead neurons dropped.
        dead_mask:       Boolean array (N_valid_pre_dead_filter,) — True = dead neuron.
                         Pass to ``get_neuron_map`` together with ``monkey_sessions``.
        monkey_sessions: Dict mapping each monkey to the session list returned by
                         ``get_hvm_responses``, needed by ``get_neuron_map``.
    """
    monkey_responses = []
    monkey_sessions: dict[str, list[str]] = {}
    for monkey in ALL_MONKEYS:
        rsp, sessions = get_hvm_responses(mode='area', monkey=monkey, area='all',
                                          time_window=HVM_TIME_WINDOW)
        monkey_responses.append(rsp)
        monkey_sessions[monkey] = sessions
    rsp = np.concatenate(monkey_responses, axis=1)  # (450, all_neurons, time)
    dead_mask = (
        np.all((rsp == 0) | np.isnan(rsp), axis=(0, 2))
        | np.any(np.isnan(rsp), axis=(0, 2))
    )
    rsp = rsp[:, ~dead_mask, :]
    assert not np.isnan(rsp).any(), "NaN values remain after neuron filtering."
    assert rsp.shape[0] == HVM_N_STIMULI, (
        f"expected {HVM_N_STIMULI} stimuli, got {rsp.shape[0]}"
    )
    return rsp, dead_mask, monkey_sessions


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
    rsp, _, _ = _load_hvm_neural()
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


class HVMMultiHeadDataset(Dataset):
    """Dataset yielding (neural, {'siglip', 'clip_short'}, cat_idx) for HVM multi-head training."""

    def __init__(self, neural: torch.Tensor, siglip: torch.Tensor,
                 clip_short: torch.Tensor, cat: torch.Tensor):
        self.neural  = neural
        self.cat     = cat
        self.targets = {"siglip": siglip, "clip_short": clip_short}

    def __len__(self) -> int:
        return len(self.neural)

    def __getitem__(self, i: int):
        return (self.neural[i],
                {k: v[i] for k, v in self.targets.items()},
                self.cat[i])


def make_hvm_multihead_loader(
    batch_size: int = 64,
    seed: int = SEED,
    verbose: bool = True,
    neuron_indices: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build category-stratified train/val/test DataLoaders for HVM multi-head training.

    Yields (neural, target_dict, cat_idx) where target_dict has keys:
      - "siglip":     (1152,) L2-normalised SigLIP embedding
      - "clip_short": (768,)  L2-normalised CLIP embedding (short captions)

    Split: 270 train / 90 val / 90 test, stratified across 10 HVM categories
    (identical split layout to make_hvm_loader for comparability).

    Args:
        neuron_indices: Optional 1-D array of neuron indices to keep. If None,
                        all neurons are used.
    """
    rsp, _, _ = _load_hvm_neural()
    if neuron_indices is not None:
        rsp = rsp[:, neuron_indices, :]
    neural_tensor = torch.from_numpy(rsp).float()  # (450, neurons, time)

    if not HVM_SIGLIP_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"SigLIP cache not found at {HVM_SIGLIP_EMBEDDINGS_PATH}. "
            "Run: python scripts/cache_siglip.py --dataset hvm"
        )
    if not HVM_CLIP_EMBEDS_PATH.exists():
        raise FileNotFoundError(
            f"CLIP short-caption cache not found at {HVM_CLIP_EMBEDS_PATH}. "
            "Run: python scripts/cache_text_embeds.py --dataset hvm"
        )

    siglip_t     = torch.load(HVM_SIGLIP_EMBEDDINGS_PATH, weights_only=True)        # (450, 1152) already L2-normed
    clip_short_t = F.normalize(torch.load(HVM_CLIP_EMBEDS_PATH, weights_only=True), dim=-1)  # (450, 768)

    for name, t in [("siglip", siglip_t), ("clip_short", clip_short_t)]:
        if t.shape[0] != HVM_N_STIMULI:
            raise ValueError(
                f"{name} cache has {t.shape[0]} rows, expected {HVM_N_STIMULI}."
            )

    train_idx_np, val_idx_np, test_idx_np = _category_stratified_split(seed)
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx   = torch.from_numpy(val_idx_np).long()
    test_idx  = torch.from_numpy(test_idx_np).long()

    cat_indices = torch.arange(HVM_N_STIMULI) // HVM_N_VAR  # (450,) int64

    def make(idx, shuffle):
        ds = HVMMultiHeadDataset(
            neural_tensor[idx], siglip_t[idx], clip_short_t[idx], cat_indices[idx]
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make(train_idx, shuffle=True)
    val_loader   = make(val_idx,   shuffle=False)
    test_loader  = make(test_idx,  shuffle=False)

    if verbose:
        neuron_str = f"{rsp.shape[1]}" + (" (subset)" if neuron_indices is not None else "")
        print(
            f"HVM multihead loaders (all monkeys): "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
            f"neurons={neuron_str}, time={rsp.shape[2]}, "
            f"targets=(siglip=1152, clip_short=768) [stratified by category]"
        )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    make_hvm_loader(use_embeddings=False)

"""DataLoader that expands each stimulus into one datapoint per T5 token.

Each item is (neural_response, pca_coords) where pca_coords is projected from
a single real token embedding of the stimulus's caption(s).

Using both short and detailed captions:
  - short:    ~8.6 tokens/stimulus  →  ~1,700 train pairs
  - detailed: ~61  tokens/stimulus  →  ~12,200 train pairs
  - combined: ~70  tokens/stimulus  →  ~13,900 train pairs (from 200 train stimuli)

Cache files (produced by scripts/cache_t5_xxl_tokens.py):
  cache/t5_xxl_tokens_short.pt    — {"hidden": (300, S, 4096), "lengths": (300,)}
  cache/t5_xxl_tokens_detailed.pt — {"hidden": (300, L, 4096), "lengths": (300,)}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import (
    CACHE_DIR,
    N_STIMULI, N_TRAIN, N_VAL,
    SEED,
    T5_PCA_K,
)
from data_utils.rust_loader import ALL_MONKEYS, RUST_TIME_WINDOW, get_rust_responses

T5_TOKENS_SHORT_PATH    = CACHE_DIR / "t5_xxl_tokens_short.pt"
T5_TOKENS_DETAILED_PATH = CACHE_DIR / "t5_xxl_tokens_detailed.pt"


class T5TokenDataset(Dataset):
    """Each item: (neural, pca_coords) — one real token per item.

    Args:
        neural:     (N_stim, n_neurons, T) neural responses for this split's stimuli
        pca_coords: (total_tokens, K)      PCA coords for every real token, all stimuli
        stim_index: (total_tokens,)        which stimulus each token belongs to
    """

    def __init__(
        self,
        neural: torch.Tensor,       # (N_stim, n_neurons, T)
        pca_coords: torch.Tensor,   # (total_tokens, K)
        stim_index: torch.Tensor,   # (total_tokens,) — index into neural
    ):
        self.neural     = neural
        self.pca_coords = pca_coords
        self.stim_index = stim_index

    def __len__(self):
        return len(self.pca_coords)

    def __getitem__(self, i: int):
        return self.neural[self.stim_index[i]], self.pca_coords[i]


def _load_neural() -> torch.Tensor:
    import numpy as np
    monkey_responses = []
    for monkey in ALL_MONKEYS:
        rsp, _ = get_rust_responses(
            mode='area', area='all', monkey=monkey, time_window=RUST_TIME_WINDOW
        )
        monkey_responses.append(rsp)
    obj_resp = np.concatenate(monkey_responses, axis=1)
    dead = (
        np.all((obj_resp == 0) | np.isnan(obj_resp), axis=(0, 2))
        | np.any(np.isnan(obj_resp), axis=(0, 2))
    )
    obj_resp = obj_resp[:, ~dead, :]
    return torch.from_numpy(obj_resp).float()   # (300, n_neurons, T)


def _project_tokens(
    hidden:  torch.Tensor,   # (300, max_seq, 4096)
    lengths: torch.Tensor,   # (300,) int32
    stim_subset: torch.Tensor,  # stimulus indices for this split
    basis: torch.Tensor,     # (K, 4096)
    mean:  torch.Tensor,     # (4096,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pca_coords, stim_local_index) for all real tokens in stim_subset."""
    coords_list = []
    sidx_list   = []
    for local_i, stim_i in enumerate(stim_subset.tolist()):
        n   = int(lengths[stim_i].item())
        emb = hidden[stim_i, :n]           # (n, 4096)
        centered = emb - mean              # (n, 4096)
        proj = F.normalize(centered @ basis.T, dim=-1)  # (n, K)
        coords_list.append(proj)
        sidx_list.append(torch.full((n,), local_i, dtype=torch.long))
    return torch.cat(coords_list, dim=0), torch.cat(sidx_list, dim=0)


def make_t5_token_loader(
    batch_size: int = 64,
    seed: int = SEED,
    t5_pca_k: int = T5_PCA_K,
    captions: str = "both",   # "short", "detailed", or "both"
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders pairing each neural response with individual T5 tokens.

    Each batch item is (neural_response, token_pca_coords) where token_pca_coords
    is the PCA projection of a single real T5 token embedding.

    Args:
        captions: which caption set to use — "short", "detailed", or "both"
    """
    t5_pca_basis_path = CACHE_DIR / f"t5_pca_basis_k{t5_pca_k}.pt"
    t5_pca_mean_path  = CACHE_DIR / f"t5_pca_mean_k{t5_pca_k}.pt"

    for path, name in [
        (t5_pca_basis_path, "T5 PCA basis"),
        (t5_pca_mean_path,  "T5 PCA mean"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}. Run: python scripts/precompute_t5_pca.py")

    cap_paths = []
    if captions in ("short", "both"):
        if not T5_TOKENS_SHORT_PATH.exists():
            raise FileNotFoundError(
                f"Short token cache not found at {T5_TOKENS_SHORT_PATH}. "
                "Run: python scripts/cache_t5_xxl_tokens.py"
            )
        cap_paths.append(T5_TOKENS_SHORT_PATH)
    if captions in ("detailed", "both"):
        if not T5_TOKENS_DETAILED_PATH.exists():
            raise FileNotFoundError(
                f"Detailed token cache not found at {T5_TOKENS_DETAILED_PATH}. "
                "Run: python scripts/cache_t5_xxl_tokens.py"
            )
        cap_paths.append(T5_TOKENS_DETAILED_PATH)

    basis = torch.load(t5_pca_basis_path, weights_only=True).float()   # (K, 4096)
    mean  = torch.load(t5_pca_mean_path,  weights_only=True).float()   # (4096,)

    neural = _load_neural()   # (300, n_neurons, T)

    rng = np.random.default_rng(seed)
    perm      = rng.permutation(N_STIMULI)
    train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
    val_idx   = torch.from_numpy(perm[N_TRAIN:N_TRAIN + N_VAL]).long()
    test_idx  = torch.from_numpy(perm[N_TRAIN + N_VAL:]).long()

    def build_dataset(stim_idx: torch.Tensor) -> T5TokenDataset:
        all_coords, all_sidx = [], []
        for cap_path in cap_paths:
            cache = torch.load(cap_path, weights_only=True)
            hidden  = cache["hidden"].float()    # (300, max_seq, 4096)
            lengths = cache["lengths"]           # (300,)
            coords, sidx = _project_tokens(hidden, lengths, stim_idx, basis, mean)
            all_coords.append(coords)
            all_sidx.append(sidx)

        if len(all_coords) > 1:
            # cat tokens from both caption sets; sidx already maps to local neural index
            coords_cat = torch.cat(all_coords, dim=0)
            sidx_cat   = torch.cat(all_sidx,   dim=0)
        else:
            coords_cat = all_coords[0]
            sidx_cat   = all_sidx[0]

        return T5TokenDataset(neural[stim_idx], coords_cat, sidx_cat)

    train_ds = build_dataset(train_idx)
    val_ds   = build_dataset(val_idx)
    test_ds  = build_dataset(test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    if verbose:
        print(
            f"T5 token loaders ({captions} captions, k={t5_pca_k}): "
            f"train={len(train_ds)} tokens ({len(train_idx)} stimuli), "
            f"val={len(val_ds)}, test={len(test_ds)}"
        )

    return train_loader, val_loader, test_loader

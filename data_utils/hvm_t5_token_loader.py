"""DataLoader that expands each HVM stimulus into one datapoint per T5 token.

Each item is (neural_response, pca_coords, cat_idx) where pca_coords is projected
from a single real token embedding of the stimulus's short caption.

Cache files (produced by scripts/cache_hvm_t5_xxl_tokens.py):
  cache/hvm_t5_xxl_tokens_short.pt    — {"hidden": (450, S, 4096), "lengths": (450,)}
  cache/hvm_t5_xxl_tokens_detailed.pt — {"hidden": (450, L, 4096), "lengths": (450,)}

Split: 270 train / 90 val / 90 test, stratified by category (same as other HVM loaders).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import (
    CACHE_DIR,
    SEED,
    T5_PCA_K,
    HVM_N_STIMULI,
    HVM_N_VAR,
    HVM_T5_TOKENS_SHORT_PATH, HVM_T5_TOKENS_DETAILED_PATH,
)
from data_utils.hvm_loader import _load_hvm_neural, _category_stratified_split


class HVMTokenDataset(Dataset):
    """Each item: (neural, pca_coords, cat_idx) — one real T5 token per item.

    Args:
        neural:     (N_stim, n_neurons, T) neural responses for this split's stimuli
        pca_coords: (total_tokens, K)      PCA coords for every real token
        stim_index: (total_tokens,)        which stimulus each token belongs to (local)
        cat_index:  (N_stim,)             category index for each stimulus
    """

    def __init__(
        self,
        neural: torch.Tensor,       # (N_stim, n_neurons, T)
        pca_coords: torch.Tensor,   # (total_tokens, K)
        stim_index: torch.Tensor,   # (total_tokens,) — local index into neural
        cat_index: torch.Tensor,    # (N_stim,) int64
    ):
        self.neural     = neural
        self.pca_coords = pca_coords
        self.stim_index = stim_index
        self.cat_index  = cat_index

    def __len__(self):
        return len(self.pca_coords)

    def __getitem__(self, i: int):
        si = self.stim_index[i]
        return self.neural[si], self.pca_coords[i], self.cat_index[si]


def _project_tokens(
    hidden:      torch.Tensor,   # (450, max_seq, 4096)
    lengths:     torch.Tensor,   # (450,) int32
    stim_subset: torch.Tensor,   # global stimulus indices for this split
    basis:       torch.Tensor,   # (K, 4096)
    mean:        torch.Tensor,   # (4096,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pca_coords, stim_local_index) for all real tokens in stim_subset."""
    coords_list = []
    sidx_list   = []
    for local_i, stim_i in enumerate(stim_subset.tolist()):
        n   = int(lengths[stim_i].item())
        emb = hidden[stim_i, :n]            # (n, 4096)
        centered = emb - mean
        proj = F.normalize(centered @ basis.T, dim=-1)  # (n, K)
        coords_list.append(proj)
        sidx_list.append(torch.full((n,), local_i, dtype=torch.long))
    return torch.cat(coords_list, dim=0), torch.cat(sidx_list, dim=0)


def make_hvm_t5_token_loader(
    batch_size: int = 64,
    seed: int = SEED,
    t5_pca_k: int = T5_PCA_K,
    captions: str = "short",    # "short", "detailed", or "both"
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build category-stratified train/val/test DataLoaders for HVM T5 token training.

    Each batch item is (neural_response, token_pca_coords, cat_idx).
    Split: 270 train / 90 val / 90 test, stratified by HVM category.

    Args:
        captions: which caption set — "short", "detailed", or "both"
    """
    t5_pca_basis_path = CACHE_DIR / f"t5_pca_basis_k{t5_pca_k}.pt"
    t5_pca_mean_path  = CACHE_DIR / f"t5_pca_mean_k{t5_pca_k}.pt"

    for path, name in [
        (t5_pca_basis_path, "T5 PCA basis"),
        (t5_pca_mean_path,  "T5 PCA mean"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}. Run: python scripts/precompute_t5_pca.py"
            )

    cap_paths = []
    if captions in ("short", "both"):
        if not HVM_T5_TOKENS_SHORT_PATH.exists():
            raise FileNotFoundError(
                f"HVM short token cache not found at {HVM_T5_TOKENS_SHORT_PATH}. "
                "Run: python scripts/cache_hvm_t5_xxl_tokens.py"
            )
        cap_paths.append(HVM_T5_TOKENS_SHORT_PATH)
    if captions in ("detailed", "both"):
        if not HVM_T5_TOKENS_DETAILED_PATH.exists():
            raise FileNotFoundError(
                f"HVM detailed token cache not found at {HVM_T5_TOKENS_DETAILED_PATH}. "
                "Run: python scripts/cache_hvm_t5_xxl_tokens.py --captions detailed"
            )
        cap_paths.append(HVM_T5_TOKENS_DETAILED_PATH)

    basis = torch.load(t5_pca_basis_path, weights_only=True).float()   # (K, 4096)
    mean  = torch.load(t5_pca_mean_path,  weights_only=True).float()   # (4096,)

    rsp = _load_hvm_neural()                             # (450, n_neurons, T)
    neural_tensor = torch.from_numpy(rsp).float()

    train_idx_np, val_idx_np, test_idx_np = _category_stratified_split(seed)
    train_idx = torch.from_numpy(train_idx_np).long()
    val_idx   = torch.from_numpy(val_idx_np).long()
    test_idx  = torch.from_numpy(test_idx_np).long()

    cat_indices = torch.arange(HVM_N_STIMULI) // HVM_N_VAR   # (450,) int64

    def build_dataset(stim_idx: torch.Tensor) -> HVMTokenDataset:
        all_coords, all_sidx = [], []
        for cap_path in cap_paths:
            cache   = torch.load(cap_path, weights_only=True)
            hidden  = cache["hidden"].float()   # (450, max_seq, 4096)
            lengths = cache["lengths"]          # (450,)
            coords, sidx = _project_tokens(hidden, lengths, stim_idx, basis, mean)
            all_coords.append(coords)
            all_sidx.append(sidx)

        if len(all_coords) > 1:
            coords_cat = torch.cat(all_coords, dim=0)
            sidx_cat   = torch.cat(all_sidx,   dim=0)
        else:
            coords_cat = all_coords[0]
            sidx_cat   = all_sidx[0]

        return HVMTokenDataset(
            neural_tensor[stim_idx],
            coords_cat,
            sidx_cat,
            cat_indices[stim_idx],
        )

    train_ds = build_dataset(train_idx)
    val_ds   = build_dataset(val_idx)
    test_ds  = build_dataset(test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    if verbose:
        print(
            f"HVM T5 token loaders ({captions} captions, k={t5_pca_k}): "
            f"train={len(train_ds)} tokens ({len(train_idx)} stimuli), "
            f"val={len(val_ds)}, test={len(test_ds)}"
        )

    return train_loader, val_loader, test_loader

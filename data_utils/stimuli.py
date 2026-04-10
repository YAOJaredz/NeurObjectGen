"""Load the 300 Rust & DiCarlo naturalistic image stimuli."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from config_const import N_STIMULI, N_TRAIN, N_VAL, SEED, RUST_STIM_DIR

_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_stimuli() -> torch.Tensor:
    """Load all cropped stimuli as a (N_STIMULI, 3, 224, 224) tensor.

    Images are sorted by index so that position i corresponds to stimulus i
    in the neural response arrays from rust_loader.py.
    """
    imgs = []
    for i in range(N_STIMULI):
        path = RUST_STIM_DIR / f"{i:04d}.png"
        with Image.open(path) as im:
            imgs.append(_transform(im.convert("RGB")))
    return torch.stack(imgs)  # (N_STIMULI, 3, 224, 224)


def make_stimuli_loaders(batch_size: int = 64, seed: int = SEED) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for stimulus images.

    Uses the same seeded permutation as make_rust_loader so that stimulus
    index i in these loaders corresponds to the same stimulus in the neural
    response loaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    tensor = load_stimuli()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_STIMULI)
    train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
    val_idx = torch.from_numpy(perm[N_TRAIN:N_TRAIN + N_VAL]).long()
    test_idx = torch.from_numpy(perm[N_TRAIN + N_VAL:]).long()

    train_loader = DataLoader(TensorDataset(tensor[train_idx]), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(tensor[val_idx]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(tensor[test_idx]), batch_size=batch_size, shuffle=False)

    print(
        f"stimuli loaders created: "
        f"train={train_idx.shape[0]}, val={val_idx.shape[0]}, test={test_idx.shape[0]}, "
        f"shape={tuple(tensor.shape[1:])}"
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    make_stimuli_loaders()

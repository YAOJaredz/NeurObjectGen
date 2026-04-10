"""Load the 300 Rust & DiCarlo naturalistic image stimuli."""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from config_const import N_STIMULI, RUST_STIM_DIR

_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_rust_stimuli() -> torch.Tensor:
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


if __name__ == "__main__":
    t = load_rust_stimuli()
    print(f"stimuli loaded: {tuple(t.shape)}")

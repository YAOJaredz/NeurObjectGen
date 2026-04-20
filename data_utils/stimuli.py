"""Load the Rust and HVM image stimuli."""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config_const import N_STIMULI, RUST_STIM_DIR, HVM_N_STIMULI, HVM_CATEGORIES, HVM_NOFIXATION_DIR

_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_rust_stimuli() -> torch.Tensor:
    """Load all cropped stimuli as a (N_STIMULI, 3, 224, 224) tensor.

    Images are sorted by index so that position i corresponds to stimulus i
    in the neural response arrays from rust_loader.py.
    """
    imgs = []
    for i in tqdm(range(N_STIMULI), desc="Loading stimuli"):
        path = RUST_STIM_DIR / f"{i:04d}.png"
        with Image.open(path) as im:
            imgs.append(_transform(im.convert("RGB")))
    return torch.stack(imgs)  # (N_STIMULI, 3, 224, 224)


def load_hvm_stimuli() -> torch.Tensor:
    """Load all HVM stimuli as a (450, 3, 224, 224) tensor.

    Stimuli are ordered to match the neural response array from hvm_loader.py:
    categories in alphabetical order (apple, bear, car, ..., turtle), and
    within each category the 45 variations sorted by index (0–44).

    Requires images to be copied locally first via scripts/copy_hvm_stimuli.py.
    """
    imgs = []
    for cat in tqdm(HVM_CATEGORIES, desc="Loading HVM stimuli"):
        cat_dir = HVM_NOFIXATION_DIR / cat
        for k in range(45):
            path = cat_dir / f"{k:02d}.png"
            with Image.open(path) as im:
                imgs.append(_transform(im.convert("RGB")))
    return torch.stack(imgs)  # (450, 3, 224, 224)


if __name__ == "__main__":
    t = load_rust_stimuli()
    print(f"stimuli loaded: {tuple(t.shape)}")

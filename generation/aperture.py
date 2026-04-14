"""Rust-dataset aperture mask for FLUX packed-latent compositing.

Builds a circular aperture minus a central fixation square, soft-downsampled
to FLUX's packed-latent grid. Used by the img2img generator to keep the
untrained area of the stimulus pinned to the original init latent while
letting the IP-Adapter repaint the trained aperture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config_const import RUST_LOSS_MASK_PATH

DEFAULT_CENTER_FRAC = 0.06


def build_packed_aperture_mask(
    image_size: int,
    center_frac: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Build and cache the circle-minus-fixation mask at packed-latent resolution.

    Args:
        image_size:  Square pixel resolution of the target generation.
        center_frac: Fixation square side length as a fraction of image_size.
        device:      Target device for the returned tensor.

    Returns:
        (1, packed_seq_len, 1) float tensor where packed_seq_len = (image_size/16)**2.
    """
    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    c = (image_size - 1) / 2
    r = image_size / 2
    circle = ((yy - c) ** 2 + (xx - c) ** 2) <= r ** 2

    half = int(round(image_size * center_frac / 2))
    center_sq = torch.zeros_like(circle, dtype=torch.bool)
    lo = int(round(c)) - half
    hi = int(round(c)) + half + 1
    center_sq[lo:hi, lo:hi] = True

    pixel_mask = (circle & ~center_sq).float().view(1, 1, image_size, image_size)
    packed_grid = image_size // 16
    mask = F.avg_pool2d(pixel_mask, kernel_size=16).view(1, packed_grid * packed_grid, 1)

    RUST_LOSS_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    params = {"image_size": image_size, "center_frac": center_frac}
    torch.save({"mask": mask.cpu(), "params": params}, RUST_LOSS_MASK_PATH)

    return mask.to(device)


def load_packed_aperture_mask(
    image_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    center_frac: float = DEFAULT_CENTER_FRAC,
    binarize: bool = True,
) -> torch.Tensor:
    """Load the cached aperture mask, rebuilding it if missing or stale.

    Args:
        image_size:  Must match whatever image_size the mask was built for.
        device:      Target device.
        dtype:       Target dtype.
        center_frac: Fixation square fraction. Defaults to the value used by
                     the training loss mask.
        binarize:    If True, threshold the soft pool at 0.5 — appropriate for
                     hard inside/outside compositing at inference. If False,
                     return the raw soft weights (useful for weighted losses).

    Returns:
        (1, packed_seq_len, 1) tensor on device, in requested dtype.
    """
    needs_build = True
    if RUST_LOSS_MASK_PATH.exists():
        blob = torch.load(RUST_LOSS_MASK_PATH, weights_only=False, map_location="cpu")
        params = blob.get("params", {})
        if params.get("image_size") == image_size and params.get("center_frac") == center_frac:
            mask = blob["mask"]
            needs_build = False

    if needs_build:
        mask = build_packed_aperture_mask(image_size, center_frac, device="cpu")

    if binarize:
        mask = (mask > 0.5).to(dtype=dtype)
    else:
        mask = mask.to(dtype=dtype)
    return mask.to(device)

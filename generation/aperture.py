"""Aperture masks for FLUX packed-latent compositing (RUST and HVM datasets).

Builds a circular aperture minus a central fixation square, soft-downsampled
to FLUX's packed-latent grid. Used by the img2img generator to keep the
untrained area of the stimulus pinned to the original init latent while
letting the IP-Adapter repaint the trained aperture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config_const import (
    RUST_LOSS_MASK_PATH, HVM_LOSS_MASK_PATH,
    APERTURE_CENTER_FRAC, HVM_RADIUS_FRAC, HVM_CENTER_FRAC,
)


def build_packed_aperture_mask(
    image_size: int,
    center_frac: float,
    device: torch.device | str,
    radius_frac: float = 0.5,
    save_path=None,
) -> torch.Tensor:
    """Build and cache the circle-minus-fixation mask at packed-latent resolution.

    Args:
        image_size:  Square pixel resolution of the target generation.
        center_frac: Fixation square side length as a fraction of image_size.
        device:      Target device for the returned tensor.
        radius_frac: Circle radius as a fraction of image_size (default 0.5 = full).
        save_path:   Where to cache the mask (.pt). Defaults to RUST_LOSS_MASK_PATH.

    Returns:
        (1, packed_seq_len, 1) float tensor where packed_seq_len = (image_size/16)**2.
    """
    if save_path is None:
        save_path = RUST_LOSS_MASK_PATH

    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    c = (image_size - 1) / 2
    r = image_size * radius_frac
    circle = ((yy - c) ** 2 + (xx - c) ** 2) <= r ** 2

    pixel_mask = circle.float().view(1, 1, image_size, image_size)
    packed_grid = image_size // 16
    mask = F.avg_pool2d(pixel_mask, kernel_size=16).view(1, packed_grid * packed_grid, 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    params = {"image_size": image_size, "center_frac": center_frac, "radius_frac": radius_frac}
    torch.save({"mask": mask.cpu(), "params": params}, save_path)

    return mask.to(device)


def load_packed_aperture_mask(
    image_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    center_frac: float = APERTURE_CENTER_FRAC,
    radius_frac: float = 0.5,
    binarize: bool = True,
    save_path=None,
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
    if save_path is None:
        save_path = RUST_LOSS_MASK_PATH

    needs_build = True
    if save_path.exists():
        blob = torch.load(save_path, weights_only=False, map_location="cpu")
        params = blob.get("params", {})
        if (params.get("image_size") == image_size
                and params.get("center_frac") == center_frac
                and params.get("radius_frac", 0.5) == radius_frac):
            mask = blob["mask"]
            needs_build = False

    if needs_build:
        mask = build_packed_aperture_mask(
            image_size, center_frac, device="cpu",
            radius_frac=radius_frac, save_path=save_path,
        )

    if binarize:
        mask = (mask > 0.5).to(dtype=dtype)
    else:
        mask = mask.to(dtype=dtype)
    return mask.to(device)


def load_hvm_packed_aperture_mask(
    image_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    binarize: bool = True,
) -> torch.Tensor:
    """Load (or build) the HVM aperture mask.

    HVM aperture params measured from stimuli/hvm_cropped (276×276 px):
      radius_frac=0.4909, center_frac=0.0362
    """
    return load_packed_aperture_mask(
        image_size=image_size,
        device=device,
        dtype=dtype,
        center_frac=HVM_CENTER_FRAC,
        radius_frac=HVM_RADIUS_FRAC,
        binarize=binarize,
        save_path=HVM_LOSS_MASK_PATH,
    )

"""Frozen SigLIP image encoder: defines the target space for neural decoding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor, SiglipProcessor, SiglipVisionModel

from config_const import SIGLIP_MODEL_ID, APERTURE_CENTER_FRAC
from get_device import get_device

_BATCH_SIZE = 32


def strip_aperture(pil_image: Image.Image) -> Image.Image:
    """Remove the circular aperture mask and fixation square from a stimulus.

    The Rust stimuli are RGBA with alpha=0 outside the circular aperture and a
    white fixation square at the centre. Both regions are out-of-distribution
    for SigLIP (trained on natural full-frame images). This function:

      1. Fills alpha=0 corner pixels with the mean colour of the visible circle.
      2. Fills the fixation square with the local mean of a ring around it.

    Returns a plain RGB PIL image suitable for SiglipImageProcessor.
    """
    arr = np.array(pil_image.convert("RGBA")).astype(np.float32)
    h, w = arr.shape[:2]
    rgb, alpha = arr[..., :3], arr[..., 3]

    inside = alpha > 0  # (H, W) bool — True inside the circular aperture

    # --- fill corners (outside circle) with per-channel mean of inside pixels ---
    mean_color = rgb[inside].mean(axis=0)  # (3,)
    rgb[~inside] = mean_color

    # --- fill fixation square with local surround mean ---
    cx, cy = (w - 1) / 2, (h - 1) / 2
    half = int(round(min(h, w) * APERTURE_CENTER_FRAC / 2))
    lo_y, hi_y = int(round(cy)) - half, int(round(cy)) + half + 1
    lo_x, hi_x = int(round(cx)) - half, int(round(cx)) + half + 1

    # Surround ring: same bounding box expanded by half again
    ring = half
    slo_y = max(0, lo_y - ring)
    shi_y = min(h, hi_y + ring)
    slo_x = max(0, lo_x - ring)
    shi_x = min(w, hi_x + ring)
    surround_mask = np.zeros((h, w), dtype=bool)
    surround_mask[slo_y:shi_y, slo_x:shi_x] = True
    surround_mask[lo_y:hi_y, lo_x:hi_x] = False  # exclude the fixation square itself
    surround_mask &= inside

    if surround_mask.any():
        fix_fill = rgb[surround_mask].mean(axis=0)
    else:
        fix_fill = mean_color
    rgb[lo_y:hi_y, lo_x:hi_x] = fix_fill

    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8), mode="RGB")


def load_siglip(device: str | None = None) -> tuple[SiglipImageProcessor, SiglipVisionModel]:
    """Load frozen SigLIP vision model and its processor onto device."""
    if device is None:
        device = get_device()
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
    vision_model = AutoModel.from_pretrained(SIGLIP_MODEL_ID).vision_model
    vision_model.to(device).eval()
    for p in vision_model.parameters():
        p.requires_grad_(False)
    return processor, vision_model


def embed_images(
    processor: SiglipImageProcessor,
    vision_model: SiglipVisionModel,
    images: torch.Tensor,
) -> torch.Tensor:
    """Embed a (N, 3, H, W) [0, 1] tensor using frozen SigLIP.

    Returns a (N, D) L2-normalised global embedding tensor on CPU.
    """
    device = get_device()

    pil_images = [
        Image.fromarray((images[i].permute(1, 2, 0).numpy() * 255).astype("uint8"))
        for i in range(len(images))
    ]

    all_embeds = []
    for start in range(0, len(pil_images), _BATCH_SIZE):
        batch = pil_images[start : start + _BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vision_model(**inputs)
        embeds = outputs.pooler_output  # (B, D)
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)  # (N, D)


def embed_images_stripped(
    processor: SiglipImageProcessor,
    vision_model: SiglipVisionModel,
    pil_images: list[Image.Image],
) -> torch.Tensor:
    """Embed raw RGBA stimulus PIL images after stripping the aperture mask.

    Calls strip_aperture on each image before encoding, so SigLIP sees a
    full-frame natural-looking RGB image instead of a circle on black with a
    white fixation dot.

    Args:
        pil_images: List of RGBA (or RGB) PIL images straight from disk.

    Returns:
        (N, D) L2-normalised global embedding tensor on CPU.
    """
    device = get_device()
    stripped = [strip_aperture(im) for im in pil_images]

    all_embeds = []
    for start in range(0, len(stripped), _BATCH_SIZE):
        batch = stripped[start : start + _BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vision_model(**inputs)
        embeds = outputs.pooler_output  # (B, D)
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)  # (N, D)


def embed_images_patches(
    processor: SiglipImageProcessor,
    vision_model: SiglipVisionModel,
    images: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Extract per-patch SigLIP features and pool to a (grid_size x grid_size) grid.

    SigLIP-so400m-patch14-384 emits a 27x27 patch grid in last_hidden_state.
    We adaptive-avg-pool that to (grid_size, grid_size), L2-normalise per token,
    and return shape (N, grid_size*grid_size, D) on CPU.
    """
    device = get_device()

    pil_images = [
        Image.fromarray((images[i].permute(1, 2, 0).numpy() * 255).astype("uint8"))
        for i in range(len(images))
    ]

    all_tokens = []
    for start in range(0, len(pil_images), _BATCH_SIZE):
        batch = pil_images[start : start + _BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vision_model(**inputs)
        hidden = outputs.last_hidden_state  # (B, N_patches, D)
        b, n_patches, d = hidden.shape
        side = int(round(n_patches ** 0.5))
        assert side * side == n_patches, f"non-square patch grid: {n_patches}"
        # (B, D, side, side) for adaptive_avg_pool2d
        grid = hidden.transpose(1, 2).reshape(b, d, side, side)
        pooled = F.adaptive_avg_pool2d(grid, output_size=grid_size)  # (B, D, g, g)
        pooled = pooled.reshape(b, d, grid_size * grid_size).transpose(1, 2)  # (B, K, D)
        pooled = F.normalize(pooled, dim=-1)
        all_tokens.append(pooled.cpu())

    return torch.cat(all_tokens, dim=0)  # (N, K, D)

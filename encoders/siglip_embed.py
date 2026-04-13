"""Frozen SigLIP image encoder: defines the target space for neural decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, SiglipImageProcessor, SiglipProcessor, SiglipVisionModel

from get_device import get_device

SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
_BATCH_SIZE = 32


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

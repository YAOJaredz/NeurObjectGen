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

    Returns a (N, D) L2-normalised embedding tensor on CPU.
    """
    device = get_device()

    # images are [0, 1] tensors; convert to PIL for SigLIP processor
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
        # Use the [CLS] token (pooler_output) as the image embedding
        embeds = outputs.pooler_output  # (B, D)
        embeds = F.normalize(embeds, dim=-1)
        all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)  # (N, D)

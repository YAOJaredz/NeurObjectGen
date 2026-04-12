"""Compose conditioning signals: neural-only, text-only, neural+text.

Three conditions from the proposal's ablation study:
  - neural_only:      IP-Adapter embedding from neural decoder, empty prompt
  - text_only:        No IP-Adapter signal, BLIP-2 caption as prompt
  - neural_plus_text: IP-Adapter embedding from neural decoder + BLIP-2 caption
"""

import torch


def neural_only(neural_embedding: torch.Tensor) -> dict:
    """Condition on neural embedding only, no text prompt.

    Args:
        neural_embedding: (D,) or (1, D) tensor from the neural encoder.

    Returns:
        kwargs dict for ``generate()``: image_embedding set, prompt=None.
    """
    return {
        "image_embedding": neural_embedding,
        "prompt": None,
    }


def text_only(caption: str) -> dict:
    """Condition on text caption only, no image embedding signal.

    A zero embedding is passed as the image_embedding so that the IP-Adapter
    projection is present but contributes nothing.

    Args:
        caption: BLIP-2 auto-caption string for this stimulus.

    Returns:
        kwargs dict for ``generate()``: zero image_embedding, prompt set.
    """
    from config_const import SIGLIP_DIM
    return {
        "image_embedding": torch.zeros(SIGLIP_DIM),
        "prompt": caption,
        "ip_adapter_scale": 0.0,
    }


def neural_plus_text(neural_embedding: torch.Tensor, caption: str) -> dict:
    """Condition on both neural embedding and text caption.

    Args:
        neural_embedding: (D,) or (1, D) tensor from the neural encoder.
        caption:          BLIP-2 auto-caption string for this stimulus.

    Returns:
        kwargs dict for ``generate()``: both image_embedding and prompt set.
    """
    return {
        "image_embedding": neural_embedding,
        "prompt": caption,
    }

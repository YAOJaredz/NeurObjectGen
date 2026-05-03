"""LoRA fine-tuning of the FLUX transformer backbone.

Wraps Q/K/V/O of the main attention path, the joint-attention text-side
projections, and both feed-forward modules across all 57 FLUX transformer
blocks (19 double-stream + 38 single-stream). At rank 8 this gives ~30-40M
trainable parameters while the FLUX base weights, VAE, text encoders,
SigLIP encoder, and IP-Adapter all stay frozen.

We use the standard HuggingFace pattern (``peft.LoraConfig`` +
``transformer.add_adapter``), with checkpoints saved in the
``FluxLoraLoaderMixin`` format so they are loaded at inference via
``pipe.load_lora_weights(path)`` -- the same format every FLUX LoRA in the
diffusers ecosystem uses.

Target modules are the canonical list from
``diffusers/examples/dreambooth/train_dreambooth_lora_flux.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from diffusers import FluxPipeline
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict


FLUX_LORA_TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]


def add_flux_lora(
    pipe: FluxPipeline,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> list[nn.Parameter]:
    """Inject LoRA adapters into the FLUX transformer's attention + FF layers.

    Returns the list of trainable LoRA parameters (for handing to AdamW).
    The base FLUX weights remain frozen; ``add_adapter`` flips
    ``requires_grad=True`` only on the LoRA factors.
    """
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        init_lora_weights="gaussian",
        target_modules=FLUX_LORA_TARGET_MODULES,
    )
    pipe.transformer.add_adapter(cfg)
    # LoRA factors live in fp32 for optimizer stability; base weights stay bf16.
    cast_training_params(pipe.transformer, dtype=torch.float32)
    return [p for p in pipe.transformer.parameters() if p.requires_grad]


def save_flux_lora(pipe: FluxPipeline, path: str | Path) -> None:
    """Save LoRA weights in FluxLoraLoaderMixin format (loadable via ``pipe.load_lora_weights``).

    The on-disk file is a ``safetensors`` blob with keys like
    ``transformer.transformer_blocks.0.attn.to_k.lora_A.weight`` etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    transformer_lora_layers = get_peft_model_state_dict(pipe.transformer)
    FluxPipeline.save_lora_weights(
        save_directory=str(path.parent),
        transformer_lora_layers=transformer_lora_layers,
        weight_name=path.name,
    )

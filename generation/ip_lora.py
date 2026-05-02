"""LoRA adaptation of the IP-Adapter K/V projections.

The frozen InstantX IP-Adapter installs an :class:`IPAFluxAttnProcessor` on
each of FLUX's 57 attention blocks (19 double + 38 single). Each processor has
two ``nn.Linear(4096, 3072)`` modules — ``to_k_ip`` and ``to_v_ip`` — that map
the projected SigLIP IP tokens into the K/V space the FLUX attention reads
from. Those are the only IP-specific weights in the whole FLUX stack.

This module wraps both linears with a lightweight LoRA adapter so we can
fine-tune the adapter to the HVM stimulus distribution while keeping the
~12B-param FLUX transformer, the VAE, the text encoders, and the MLPProjModel
fully frozen. The wrapper is a drop-in ``nn.Linear`` replacement: its
``forward(x)`` returns ``base(x) + (alpha / r) * lora_B(lora_A(x))`` so the
existing IPAFluxAttnProcessor call sites at flux_instantx.py:153/169 keep
working without edits.

We deliberately do not use ``peft.tuners.lora.layer.Linear``: its constructor
signature has shifted across peft 0.11–0.14, and IPAFluxAttnProcessor lives
outside the module tree that ``inject_adapter_in_model`` discovers. A 30-line
hand-rolled wrapper is simpler and version-stable.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` with an additive low-rank update.

    The base linear stays in its original dtype (typically bf16). The LoRA
    factors are kept in fp32 for optimizer stability and downcast inside
    ``forward`` to match the input dtype.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        # PEFT-style init: A ~ Kaiming, B = 0 → adapter starts as identity.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        # LoRA params live in fp32; casting happens in forward.
        self.lora_A.to(dtype=torch.float32)
        self.lora_B.to(dtype=torch.float32)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        x32 = self.dropout(x).to(self.lora_A.weight.dtype)
        delta = self.lora_B(self.lora_A(x32)) * self.scaling
        return out + delta.to(out.dtype)


def _iter_ip_processors(pipe):
    """Yield (block_idx, processor) for every IPAFluxAttnProcessor on the FLUX transformer.

    Block index matches the order returned by ``pipe.transformer.attn_processors``,
    which is the same 0..56 ordering InstantX's checkpoint uses.
    """
    from generation.flux_instantx import IPAFluxAttnProcessor  # local import to avoid cycle

    for i, (_, proc) in enumerate(pipe.transformer.attn_processors.items()):
        if isinstance(proc, IPAFluxAttnProcessor):
            yield i, proc


def add_ip_lora(pipe, rank: int = 8, alpha: int = 16, dropout: float = 0.0) -> list[nn.Parameter]:
    """Wrap ``to_k_ip`` and ``to_v_ip`` on every IP processor with a LoRA adapter.

    Returns the list of trainable LoRA parameters (for handing to AdamW).
    Idempotency: if a processor's ``to_k_ip`` is already a ``LoRALinear`` we
    skip it, so callers can re-wrap safely.
    """
    trainable: list[nn.Parameter] = []
    n_wrapped = 0
    for _, proc in _iter_ip_processors(pipe):
        for attr in ("to_k_ip", "to_v_ip"):
            base = getattr(proc, attr)
            if isinstance(base, LoRALinear):
                continue
            wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
            wrapped.to(device=base.weight.device)
            setattr(proc, attr, wrapped)
            trainable.extend(p for p in wrapped.parameters() if p.requires_grad)
            n_wrapped += 1
    if n_wrapped == 0:
        raise RuntimeError("no IPAFluxAttnProcessor found on pipe.transformer — did you call _install_ip_adapter()?")
    return trainable


def ip_lora_state_dict(pipe) -> dict[str, torch.Tensor]:
    """Collect LoRA A/B weights into a flat state dict keyed by block index + attr."""
    sd: dict[str, torch.Tensor] = {}
    for i, proc in _iter_ip_processors(pipe):
        for attr in ("to_k_ip", "to_v_ip"):
            mod = getattr(proc, attr)
            if not isinstance(mod, LoRALinear):
                continue
            sd[f"{i}.{attr}.lora_A.weight"] = mod.lora_A.weight.detach().cpu().contiguous()
            sd[f"{i}.{attr}.lora_B.weight"] = mod.lora_B.weight.detach().cpu().contiguous()
    return sd


def save_ip_lora(pipe, path: str | Path, rank: int, alpha: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = ip_lora_state_dict(pipe)
    metadata = {"rank": str(rank), "alpha": str(alpha), "format": "ip_lora_v1"}
    save_file(sd, str(path), metadata=metadata)


def load_ip_lora(pipe, path: str | Path) -> None:
    """Load a LoRA checkpoint, wrapping K/V layers in-place if not already wrapped."""
    path = Path(path)
    sd = load_file(str(path))
    # Infer rank from any lora_A row.
    sample_A = next(v for k, v in sd.items() if k.endswith("lora_A.weight"))
    rank = sample_A.shape[0]
    alpha_meta = None
    try:
        with open(path, "rb") as f:
            # safetensors stores metadata in the header; load_file's metadata is via the
            # safe_open context manager. Use it for round-trip alpha recovery.
            from safetensors import safe_open
            with safe_open(str(path), framework="pt") as sf:
                meta = sf.metadata() or {}
                alpha_meta = int(meta["alpha"]) if "alpha" in meta else None
    except Exception:
        alpha_meta = None
    alpha = alpha_meta if alpha_meta is not None else rank * 2

    add_ip_lora(pipe, rank=rank, alpha=alpha)

    # Now copy weights in.
    missing = []
    for i, proc in _iter_ip_processors(pipe):
        for attr in ("to_k_ip", "to_v_ip"):
            mod = getattr(proc, attr)
            assert isinstance(mod, LoRALinear)
            for sub in ("lora_A", "lora_B"):
                key = f"{i}.{attr}.{sub}.weight"
                if key not in sd:
                    missing.append(key)
                    continue
                target = getattr(mod, sub).weight
                target.data.copy_(sd[key].to(device=target.device, dtype=target.dtype))
    if missing:
        raise RuntimeError(f"ip_lora checkpoint missing {len(missing)} keys, e.g. {missing[:3]}")

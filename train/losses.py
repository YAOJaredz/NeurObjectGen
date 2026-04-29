"""Shared loss functions for neural encoder training."""

import torch
import torch.nn.functional as F


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance: 1 − cos(pred, target), averaged over batch."""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style). Directly optimises 2-AFC-like discrimination."""
    pred_n = F.normalize(pred, dim=-1)
    tgt_n  = F.normalize(target, dim=-1)
    logits = (pred_n @ tgt_n.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def info_nce_loss_with_bank(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    memory_bank: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE with optional memory-bank negative augmentation.

    Args:
        pred:        (B, D) L2-normalised predicted embeddings.
        target:      (B, D) L2-normalised target embeddings (positives).
        temperature: Scalar temperature for logit scaling.
        memory_bank: (M, D) all training target embeddings. When provided,
                     each query is evaluated against its in-batch positive
                     plus all M bank embeddings as negatives. Should be detached.
    """
    if memory_bank is not None:
        keys = torch.cat([target, memory_bank], dim=0)
        logits_p = pred @ keys.T / temperature
        labels = torch.arange(len(pred), device=pred.device)
        loss_p = F.cross_entropy(logits_p, labels)
        logits_t = target @ pred.T / temperature
        loss_t = F.cross_entropy(logits_t, labels)
    else:
        logits = pred @ target.T / temperature
        labels = torch.arange(len(pred), device=pred.device)
        loss_p = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
    return (loss_p + loss_t) / 2


def head_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    nce_weight: float,
    temperature: float,
) -> torch.Tensor:
    """InfoNCE + cosine mix: nce_weight * InfoNCE + (1 - nce_weight) * cosine."""
    l_cos = cosine_loss(pred, target)
    if nce_weight > 0.0 and pred.size(0) > 1:
        l_nce = info_nce_loss(pred, target, temperature)
        return nce_weight * l_nce + (1.0 - nce_weight) * l_cos
    return l_cos


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Uniformity loss: encourages embeddings to spread uniformly on the hypersphere."""
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

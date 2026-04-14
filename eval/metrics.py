"""Reconstruction metrics: cosine similarity, SSIM, 2AFC identification, R²."""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity between predicted and target embeddings.

    Args:
        pred:   (N, D) predicted embeddings
        target: (N, D) target embeddings

    Returns:
        (N,) cosine similarity scores in [-1, 1]
    """
    return F.cosine_similarity(pred, target, dim=-1)


def ssim(pred_image: torch.Tensor, target_image: torch.Tensor) -> float:
    """Mean SSIM over a batch of images.

    Args:
        pred_image:   (N, C, H, W) tensor, values in [0, 1]
        target_image: (N, C, H, W) tensor, values in [0, 1]

    Returns:
        Scalar mean SSIM across the batch.
    """
    pred_np = pred_image.detach().cpu().numpy()
    target_np = target_image.detach().cpu().numpy()

    scores = []
    for p, t in zip(pred_np, target_np):
        # skimage expects (H, W, C); images are (C, H, W)
        p_hwc = p.transpose(1, 2, 0)
        t_hwc = t.transpose(1, 2, 0)
        score = structural_similarity(p_hwc, t_hwc, channel_axis=-1, data_range=1.0)
        scores.append(score)

    return sum(scores) / len(scores)


def two_afc_identification(
    pred_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
) -> float:
    """2-AFC pairwise identification accuracy (chance = 0.5).

    For each sample i, the correct match is target_embeddings[i].
    A foil is drawn from every other target in the batch.
    The prediction is correct when pred_embeddings[i] is closer to
    target_embeddings[i] than to the foil.

    Concretely this computes an N×N cosine-similarity matrix and checks
    whether the diagonal entry is the maximum in each row — equivalent
    to the fraction of (i, j≠i) pairs where the correct target wins.

    Args:
        pred_embeddings:   (N, D) predicted embeddings
        target_embeddings: (N, D) target embeddings

    Returns:
        Scalar accuracy in [0, 1].
    """
    pred_norm = F.normalize(pred_embeddings, dim=-1)
    target_norm = F.normalize(target_embeddings, dim=-1)

    # (N, N) similarity matrix: sim[i, j] = cos(pred_i, target_j)
    sim = pred_norm @ target_norm.T  # (N, N)

    N = sim.size(0)
    correct = 0
    total = 0
    for i in range(N):
        # compare diagonal to every off-diagonal entry in row i
        diag = sim[i, i]
        foils = torch.cat([sim[i, :i], sim[i, i + 1:]])
        correct += (diag > foils).sum().item()
        total += len(foils)

    return correct / total if total > 0 else 0.0


def r2_per_component(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """R² for each output dimension independently.

    Args:
        pred:   (N, D) predicted values
        target: (N, D) ground-truth values

    Returns:
        (D,) array of R² scores, one per dimension.
    """
    ss_res = ((pred - target) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    return r2

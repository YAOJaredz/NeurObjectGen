"""Reconstruction metrics: cosine similarity, SSIM, LPIPS, PixCorr, 2AFC identification, R²."""

import numpy as np
import torch
import torch.nn.functional as F
import lpips as lpips_lib
from skimage.metrics import structural_similarity


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cosine distance loss: 1 − cos(pred, target), averaged over batch."""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity between predicted and target embeddings.

    Args:
        pred:   (N, D) predicted embeddings
        target: (N, D) target embeddings

    Returns:
        (N,) cosine similarity scores in [-1, 1]
    """
    return F.cosine_similarity(pred, target, dim=-1)


def ssim(pred_image: torch.Tensor, target_image: torch.Tensor) -> np.ndarray:
    """Per-image SSIM scores.

    Args:
        pred_image:   (N, C, H, W) tensor, values in [0, 1]
        target_image: (N, C, H, W) tensor, values in [0, 1]

    Returns:
        (N,) array of SSIM scores.
    """
    pred_np   = pred_image.detach().cpu().numpy()
    target_np = target_image.detach().cpu().numpy()
    return np.array([
        structural_similarity(
            p.transpose(1, 2, 0), t.transpose(1, 2, 0),
            channel_axis=-1, data_range=1.0)
        for p, t in zip(pred_np, target_np)
    ])


def lpips_distance(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
    net: str = 'vgg',
    batch_size: int = 8,
    lpips_fn: lpips_lib.LPIPS | None = None,
) -> np.ndarray:
    """Per-image LPIPS perceptual distance (lower is better).

    Args:
        pred_images:   (N, C, H, W) tensor, values in [0, 1]
        target_images: (N, C, H, W) tensor, values in [0, 1]
        net: backbone for LPIPS ('vgg' or 'alex'); ignored if lpips_fn is given
        batch_size: images per forward pass
        lpips_fn: pre-built LPIPS instance to reuse across calls

    Returns:
        (N,) array of LPIPS scores.
    """
    fn = lpips_fn if lpips_fn is not None else lpips_lib.LPIPS(net=net).cpu()
    vals = []
    with torch.no_grad():
        for i in range(0, len(pred_images), batch_size):
            r_b = pred_images[i:i + batch_size].cpu() * 2 - 1
            o_b = target_images[i:i + batch_size].cpu() * 2 - 1
            vals.extend(fn(r_b, o_b).squeeze().tolist())
    return np.array(vals)


def pixcorr(
    pred_images: torch.Tensor,
    target_images: torch.Tensor,
) -> np.ndarray:
    """Per-image Pearson correlation on flattened pixels.

    Args:
        pred_images:   (N, C, H, W) tensor, values in [0, 1]
        target_images: (N, C, H, W) tensor, values in [0, 1]

    Returns:
        (N,) array of Pearson r values.
    """
    n = pred_images.shape[0]
    rec_flat  = pred_images.numpy().reshape(n, -1)
    orig_flat = target_images.numpy().reshape(n, -1)
    return np.array([np.corrcoef(rec_flat[i], orig_flat[i])[0, 1] for i in range(n)])


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


def retrieval_accuracy(
    pred_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    k: int | list[int] = (1, 5, 10),
) -> dict[int, float]:
    """Top-k retrieval accuracy: fraction of samples whose correct target is
    ranked in the top-k by cosine similarity.

    Args:
        pred_embeddings:   (N, D) predicted embeddings
        target_embeddings: (N, D) ground-truth embeddings; target_embeddings[i]
                           is the correct match for pred_embeddings[i]
        k: single int or list of ints

    Returns:
        Dict mapping each k to accuracy in [0, 1].
    """
    ks = [k] if isinstance(k, int) else list(k)
    pred_norm   = F.normalize(pred_embeddings, dim=-1)
    target_norm = F.normalize(target_embeddings, dim=-1)

    sim = pred_norm @ target_norm.T  # (N, N)
    N = sim.size(0)

    # rank each row descending; correct match is the diagonal
    ranks = sim.argsort(dim=1, descending=True)  # (N, N)
    correct_rank = (ranks == torch.arange(N, device=sim.device).unsqueeze(1)).nonzero(as_tuple=False)[:, 1]

    return {ki: float((correct_rank < ki).float().mean().item()) for ki in ks}


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

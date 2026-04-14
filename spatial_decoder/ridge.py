"""Closed-form ridge regression and inference helper for the spatial decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge: W = (X^T X + alpha I)^{-1} X^T Y.

    Args:
        X:     (N, d_in)  neural features
        Y:     (N, d_out) targets (PCA codes)
        alpha: L2 regularisation strength

    Returns:
        W: (d_in, d_out) weight matrix
    """
    XtX = X.T @ X
    XtX_reg = XtX + alpha * np.eye(XtX.shape[0])
    XtY = X.T @ Y
    return np.linalg.solve(XtX_reg, XtY)


def decode_spatial_ridge(
    neural_flat: np.ndarray,
    ckpt_path: str | Path,
) -> np.ndarray:
    """Map neural response vector(s) to flat FLUX packed latent(s) using the ridge model.

    Args:
        neural_flat: (neurons*time,) or (N, neurons*time) float32 array
        ckpt_path:   path to the .npz saved by train_spatial_ridge.py

    Returns:
        (N, D_latent) float32 array; reshape to (1, L, C_packed) before use
        as an init latent in generate_img2img.
    """
    ckpt = np.load(ckpt_path)
    W, Z_mean, V = ckpt["W"], ckpt["pca_mean"], ckpt["pca_V"]
    if neural_flat.ndim == 1:
        neural_flat = neural_flat[None]
    pca_codes = neural_flat @ W       # (N, K)
    return pca_codes @ V.T + Z_mean   # (N, D_latent)

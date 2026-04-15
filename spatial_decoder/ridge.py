"""Closed-form ridge regression and inference helper for the spatial decoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression, solved in whichever space is smaller.

    When d_in > N (over-determined features) uses the dual form:
        W = X^T (X X^T + alpha I)^{-1} Y
    which only requires an N×N solve instead of d_in×d_in.

    Args:
        X:     (N, d_in)  neural features
        Y:     (N, d_out) targets (PCA codes)
        alpha: L2 regularisation strength

    Returns:
        W: (d_in, d_out) weight matrix
    """
    N, d_in = X.shape
    if d_in > N:
        # dual form: solve (K + alpha I) C = Y, then W = X^T C
        K = X @ X.T                          # (N, N)
        K_reg = K + alpha * np.eye(N)
        C = np.linalg.solve(K_reg, Y)        # (N, d_out)
        return X.T @ C                       # (d_in, d_out)
    else:
        XtX = X.T @ X
        XtX_reg = XtX + alpha * np.eye(d_in)
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

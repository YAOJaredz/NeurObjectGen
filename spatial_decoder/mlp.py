"""Bottleneck MLP spatial decoder: (neurons*time) -> PCA latent codes."""

import torch
import torch.nn as nn


class SpatialMLP(nn.Module):
    """Bottleneck MLP for spatial latent regression.

    Maps flattened neural activity to K-dimensional PCA codes.
    No final L2 normalisation — PCA codes are regression targets, not
    embedding space.

    Architecture: in_dim -> bottleneck -> K
    with LayerNorm + GELU + dropout.

    Args:
        in_dim:     Dimensionality of flattened neural input (neurons * time).
        bottleneck: Hidden layer width. Keep small (128-512) given ~200
                    training stimuli.
        out_dim:    Number of PCA components K.
        dropout:    Dropout probability after the hidden activation.
    """

    def __init__(self, in_dim: int, bottleneck: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) flattened neural activity.
        Returns:
            (B, K) predicted PCA codes.
        """
        return self.net(x)

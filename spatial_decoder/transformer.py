"""Pre-LN CLS Transformer spatial decoder: (B, T, N_neurons) -> PCA latent codes."""

import torch
import torch.nn as nn


class SpatialTransformer(nn.Module):
    """Pre-LN CLS transformer reading across the time axis -> PCA latent codes.

    A learned [CLS] token is prepended; its final representation is projected
    to out_dim PCA codes. No final L2 normalisation.

    Args:
        n_neurons: Number of neurons (input features at each time step).
        d_model:   Internal transformer width. Keep small (64-128) given ~200
                   training stimuli.
        n_heads:   Number of attention heads. Must divide d_model.
        n_layers:  Number of Pre-LN transformer encoder layers.
        out_dim:   Number of PCA components K.
        dropout:   Dropout probability inside the transformer.
    """

    def __init__(
        self,
        n_neurons: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_neurons, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_neurons) neural activity over time.
        Returns:
            (B, K) predicted PCA codes.
        """
        x = self.input_proj(x)                           # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                   # (B, T+1, d_model)
        x = self.encoder(x)                              # (B, T+1, d_model)
        return self.proj(x[:, 0])                        # (B, K) — CLS token

"""Transformer encoder over the temporal dimension of neural responses (N, T) -> SigLIP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformer(nn.Module):
    """Small Pre-LN transformer encoder that reads across the time axis.

    Input shape:  (B, T, N_neurons) — batch of population responses over time.
    Output shape: (B, out_dim)      — L2-normalised SigLIP embedding.

    A learned [CLS] token is prepended; its final representation is projected
    to out_dim. Pre-LN (LayerNorm before attention/FFN) is used for stability.

    Args:
        n_neurons:  Number of neurons (input features at each time step).
        d_model:    Internal transformer width. Keep small (64-128) given ~200
                    training stimuli.
        n_heads:    Number of attention heads. Must divide d_model evenly.
        n_layers:   Number of transformer encoder layers (1-2 recommended).
        dropout:    Dropout probability applied inside the transformer.
        out_dim:    SigLIP embedding dimension (1152 for so400m-patch14-384).
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
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_neurons) float tensor of neural firing rates.

        Returns:
            (B, out_dim) L2-normalised embedding tensor.
        """
        x = self.input_proj(x)                          # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                  # (B, T+1, d_model)
        x = self.encoder(x)                             # (B, T+1, d_model)
        out = self.proj(x[:, 0])                        # (B, out_dim) — CLS token
        return F.normalize(out, dim=-1)

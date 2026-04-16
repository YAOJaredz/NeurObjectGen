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
        max_time: int = 32,
    ):
        super().__init__()
        # Normalise the 6802-d neural vector at each time step before projection.
        # Without this, a handful of high-variance neurons dominate the input
        # projection to d_model and attention collapses across time steps.
        self.input_norm = nn.LayerNorm(n_neurons)
        self.input_proj = nn.Linear(n_neurons, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Learned positional embeddings for CLS + up to max_time time steps.
        # Without these, self-attention is permutation-invariant across time.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_time + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        # Temporal attention pooling over time-step tokens (excludes CLS)
        self.attn = nn.Linear(d_model, 1, bias=False)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_neurons) float tensor of neural firing rates.

        Returns:
            (B, out_dim) L2-normalised embedding tensor.
        """
        x = self.input_norm(x)                          # (B, T, N_neurons)
        x = self.input_proj(x)                          # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                  # (B, T+1, d_model)
        T_plus_1 = x.size(1)
        if T_plus_1 > self.pos_embed.size(1):
            raise ValueError(
                f"Input has {T_plus_1} tokens but pos_embed only has "
                f"{self.pos_embed.size(1)}. Increase max_time."
            )
        x = x + self.pos_embed[:, :T_plus_1]            # add positional encoding
        x = self.encoder(x)                             # (B, T+1, d_model)
        # Attention-pool over time tokens; CLS token at index 0 provides global context
        # but learned weights decide the final mixture across all T+1 positions
        w = torch.softmax(self.attn(x), dim=1)          # (B, T+1, 1)
        pooled = (w * x).sum(dim=1)                     # (B, d_model)
        out = self.proj(pooled)                         # (B, out_dim)
        return F.normalize(out, dim=-1)

"""Multi-head transformer encoder: shared backbone → SigLIP, CLIP, and T5-PCA heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadTransformer(nn.Module):
    """Shared Pre-LN transformer backbone with three projection heads.

    A single forward pass produces embeddings for all three target spaces:
      - siglip:  (B, 1152) L2-normalised
      - clip:    (B, 768)  L2-normalised
      - t5_pca:  (B, t5_pca_k)  L2-normalised PCA coordinates (cosine target)
      - shared:  (B, shared_dim) L2-normalised shared latent (for uniformity loss)

    The backbone is identical to TemporalTransformer up to temporal attention
    pooling. After pooling, a shared projection maps d_model → shared_dim before
    branching into the three task-specific linear heads.

    Args:
        n_neurons:   Number of input neurons.
        d_model:     Internal transformer width (keep 128 given N=200 stimuli).
        n_heads:     Attention heads (must divide d_model).
        n_layers:    Transformer encoder layers.
        shared_dim:  Bottleneck dimension shared across all heads (default 512).
        t5_pca_k:    Number of T5 PCA components the t5_head predicts.
        dropout:     Dropout inside the transformer encoder layers.
        max_time:    Maximum number of time steps (for positional embeddings).
    """

    def __init__(
        self,
        n_neurons: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        shared_dim: int = 512,
        t5_pca_k: int = 64,
        dropout: float = 0.1,
        max_time: int = 32,
    ):
        super().__init__()

        # Backbone (identical to TemporalTransformer)
        self.input_norm = nn.LayerNorm(n_neurons)
        self.input_proj = nn.Linear(n_neurons, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed  = nn.Parameter(torch.zeros(1, max_time + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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
        self.attn = nn.Linear(d_model, 1, bias=False)

        # Shared projection: temporal pooled (d_model) → shared latent (shared_dim)
        self.shared_proj = nn.Linear(d_model, shared_dim)

        # Three task heads
        self.siglip_head = nn.Linear(shared_dim, 1152)
        self.clip_head   = nn.Linear(shared_dim, 768)
        self.t5_head     = nn.Linear(shared_dim, t5_pca_k)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, N_neurons) neural firing rates.

        Returns dict with keys:
            siglip:  (B, 1152) L2-normalised
            clip:    (B, 768)  L2-normalised
            t5_pca:  (B, t5_pca_k) L2-normalised PCA coords
            shared:  (B, shared_dim) L2-normalised (for uniformity loss)
        """
        x = self.input_norm(x)
        x = self.input_proj(x)                               # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)       # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                       # (B, T+1, d_model)
        T_plus_1 = x.size(1)
        if T_plus_1 > self.pos_embed.size(1):
            raise ValueError(
                f"Input has {T_plus_1} tokens but pos_embed only supports "
                f"{self.pos_embed.size(1)}. Increase max_time."
            )
        x = x + self.pos_embed[:, :T_plus_1]
        x = self.encoder(x)                                  # (B, T+1, d_model)

        w = torch.softmax(self.attn(x), dim=1)               # (B, T+1, 1)
        pooled = (w * x).sum(dim=1)                          # (B, d_model)

        shared = F.gelu(self.shared_proj(pooled))            # (B, shared_dim)

        return {
            "siglip": F.normalize(self.siglip_head(shared), dim=-1),
            "clip":   F.normalize(self.clip_head(shared),   dim=-1),
            "t5_pca": F.normalize(self.t5_head(shared), dim=-1),
            "shared": F.normalize(shared, dim=-1),
        }

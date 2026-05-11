"""Multi-head transformer V2: shared backbone → global SigLIP, obj SigLIP, and CLIP heads.

Adds two features over MultiHeadTransformer:
  1. A third head for object-crop SigLIP embeddings (siglip_obj).
  2. Dual category conditioning: "given" (GT label as input) or "predict"
     (dedicated CategoryClassifier transformer reads the same neural input
     independently, predicts the category, and feeds the predicted embedding
     into the main backbone — zero shared weights, fully decoupled gradients).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryClassifier(nn.Module):
    """Mean-pool MLP classifier for neural population → object category.

    Shares no weights with the main MultiHeadTransformerV2 backbone.
    Trained purely via cross-entropy loss; gradients do not flow into the
    embedding heads.

    Architecture: mean-pool over time → LayerNorm → Linear → GELU →
                  Linear → GELU → Linear (logits).

    Args:
        n_neurons:    Number of input neurons (must match main backbone).
        n_categories: Number of output classes.
        dropout:      Dropout probability (default 0.1).
    """

    def __init__(
        self,
        n_neurons: int,
        n_categories: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_neurons),
            nn.Linear(n_neurons, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_categories),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_neurons) neural firing rates.

        Returns:
            (B, n_categories) unnormalised logits.
        """
        return self.net(x.mean(dim=1))                       # mean-pool over time


class MultiHeadTransformerV2(nn.Module):
    """Shared Pre-LN transformer backbone with three projection heads.

    A single forward pass produces embeddings for all target spaces:
      - siglip_global: (B, 1152) L2-normalised global-image SigLIP
      - siglip_obj:    (B, 1152) L2-normalised object-crop SigLIP
      - clip:          (B, 768)  L2-normalised CLIP
      - shared:        (B, shared_dim) L2-normalised shared latent (for uniformity loss)

    Category conditioning is always active (n_categories must be > 0) and
    operates in one of two modes selected per forward call via `cat_mode`:

      "given"   — caller supplies ground-truth category indices; the
                  corresponding embedding is added to the shared latent.
      "predict" — a dedicated CategoryClassifier MLP (own weights,
                  decoupled gradients) classifies the category from the raw
                  neural input; its predicted label is used to condition the
                  shared latent. `cat_logits` is returned for CE loss.

    Args:
        n_neurons:    Number of input neurons.
        d_model:      Internal transformer width.
        n_heads:      Attention heads (must divide d_model).
        n_layers:     Transformer encoder layers.
        shared_dim:   Bottleneck dimension shared across all heads (default 512).
        dropout:      Dropout inside the transformer encoder layers.
        max_time:     Maximum number of time steps (for positional embeddings).
        n_categories: Number of object categories (required, must be > 0).
    """

    def __init__(
        self,
        n_neurons: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        shared_dim: int = 512,
        dropout: float = 0.1,
        max_time: int = 32,
        n_categories: int = 10,
    ):
        super().__init__()
        if n_categories <= 0:
            raise ValueError("MultiHeadTransformerV2 requires n_categories > 0.")

        # Backbone (identical to MultiHeadTransformer)
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
        self.siglip_global_head = nn.Linear(shared_dim, 1152)
        self.siglip_obj_head    = nn.Linear(shared_dim, 1152)
        self.clip_head          = nn.Linear(shared_dim, 768)

        # Category embedding table — zero-init, used in both modes
        self.cat_emb = nn.Embedding(n_categories, shared_dim)
        nn.init.zeros_(self.cat_emb.weight)

        # Dedicated category classifier — independent mean-pool MLP, no shared weights
        self.cat_clf = CategoryClassifier(
            n_neurons=n_neurons,
            n_categories=n_categories,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        cat: torch.Tensor | None = None,
        cat_mode: str = "given",
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x:        (B, T, N_neurons) neural firing rates.
            cat:      (B,) integer category indices. Required when cat_mode="given".
            cat_mode: "given" uses GT label; "predict" runs CategoryClassifier.

        Returns dict with keys:
            siglip_global: (B, 1152) L2-normalised
            siglip_obj:    (B, 1152) L2-normalised
            clip:          (B, 768)  L2-normalised
            shared:        (B, shared_dim) L2-normalised
            cat_logits:    (B, n_categories) or None (only set in "predict" mode)
        """
        x_raw = x                                              # save for CategoryClassifier
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

        shared_pre = F.gelu(self.shared_proj(pooled))        # (B, shared_dim)

        cat_logits = None

        if cat_mode == "given":
            if cat is None:
                raise ValueError("cat_mode='given' requires a cat tensor.")
            shared = shared_pre + self.cat_emb(cat)

        elif cat_mode == "predict":
            cat_logits = self.cat_clf(x_raw)                 # independent transformer
            cat_pred   = cat_logits.argmax(dim=-1)
            shared = shared_pre + self.cat_emb(cat_pred)

        else:
            raise ValueError(f"Unknown cat_mode: {cat_mode!r}. Choose 'given' or 'predict'.")

        return {
            "siglip_global": F.normalize(self.siglip_global_head(shared), dim=-1),
            "siglip_obj":    F.normalize(self.siglip_obj_head(shared),    dim=-1),
            "clip":          F.normalize(self.clip_head(shared),           dim=-1),
            "shared":        F.normalize(shared, dim=-1),
            "cat_logits":    cat_logits,
        }


def extract_attn_weights(model: MultiHeadTransformerV2, n_categories: int) -> np.ndarray:
    """Return (n_categories, T+1) softmax attention weights from a zero-input pass.

    Token 0 is the CLS token; tokens 1..T are the time bins.
    """
    captured = []

    def hook(module, inp, out):
        captured.append(torch.softmax(out, dim=1).squeeze(-1).detach().cpu())

    handle = model.attn.register_forward_hook(hook)
    n_time    = model.pos_embed.size(1) - 1
    n_neurons = model.input_proj.weight.size(1)
    x_dummy   = torch.zeros(1, n_time, n_neurons)
    with torch.no_grad():
        for c in range(n_categories):
            model(x_dummy, torch.tensor([c]), cat_mode="given")
    handle.remove()
    return torch.cat(captured, dim=0).numpy()  # (n_categories, T+1)


def neuron_importance(model: MultiHeadTransformerV2) -> np.ndarray:
    """L1 column norm of input_proj.weight — (N_neurons,) array."""
    W = model.input_proj.weight.detach().cpu()  # (d_model, N_neurons)
    return W.abs().sum(dim=0).numpy()           # (N_neurons,)

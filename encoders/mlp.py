"""Low-rank bottleneck MLP encoder: neural -> SigLIP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckMLP(nn.Module):
    """Two-layer MLP with a low-rank bottleneck.

    Architecture: in_dim -> bottleneck -> out_dim
    with LayerNorm + GELU activations and a final L2 normalisation to match
    the SigLIP embedding space (which is L2-normalised in siglip_embed.py).

    Args:
        in_dim:     Dimensionality of flattened neural input (N_neurons * T or
                    mean-fired N_neurons depending on how the caller prepares x).
        bottleneck: Width of the hidden layer — kept small (e.g. 128-256) to
                    act as a regularising bottleneck given the limited stimulus
                    count (~200 training samples).
        out_dim:    SigLIP embedding dimension (1152 for so400m-patch14-384).
        dropout:    Dropout probability applied after the hidden activation.
    """

    def __init__(self, in_dim: int, bottleneck: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, bottleneck)
        self.norm = nn.LayerNorm(bottleneck)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(bottleneck, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map neural activity to a L2-normalised SigLIP embedding.

        Args:
            x: (B, in_dim) float tensor of neural firing rates.

        Returns:
            (B, out_dim) L2-normalised embedding tensor.
        """
        x = self.drop(F.gelu(self.norm(self.fc1(x))))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)

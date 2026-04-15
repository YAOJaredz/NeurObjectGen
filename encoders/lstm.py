"""LSTM encoder over the temporal dimension of neural responses (N, T) -> SigLIP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTM(nn.Module):
    """Unidirectional LSTM that reads across the time axis of neural responses.

    Input shape:  (B, T, N_neurons) — batch of population responses over time.
    Output shape: (B, out_dim)      — L2-normalised SigLIP embedding.

    The final hidden state of the top LSTM layer is projected to out_dim.

    Args:
        n_neurons: Number of neurons (input features at each time step).
        hidden:    LSTM hidden size.
        out_dim:   Embedding output dimension.
        n_layers:  Number of stacked LSTM layers (default 1).
        dropout:   Dropout probability applied between LSTM layers and before projection.
    """

    def __init__(self, n_neurons: int, hidden: int, out_dim: int, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_norm = nn.LayerNorm(n_neurons)
        self.lstm = nn.LSTM(
            input_size=n_neurons, hidden_size=hidden,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        # Temporal attention pooling: learn to weight each time step
        self.attn = nn.Linear(hidden, 1, bias=False)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_neurons) float tensor of neural firing rates.

        Returns:
            (B, out_dim) L2-normalised embedding tensor.
        """
        x = self.input_norm(x)                              # normalise across neurons at each time step
        all_h, _ = self.lstm(x)                             # all_h: (B, T, hidden)
        # Learned temporal attention pooling over all time steps
        w = torch.softmax(self.attn(all_h), dim=1)          # (B, T, 1)
        pooled = (w * all_h).sum(dim=1)                     # (B, hidden)
        out = self.proj(self.drop(pooled))
        return F.normalize(out, dim=-1)

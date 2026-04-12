"""LSTM encoder over the temporal dimension of neural responses (N, T) -> SigLIP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTM(nn.Module):
    """Unidirectional LSTM that reads across the time axis of neural responses.

    Input shape:  (B, T, N_neurons) — batch of population responses over time.
    Output shape: (B, out_dim)      — L2-normalised SigLIP embedding.

    The final hidden state of the LSTM is projected to out_dim.

    Args:
        n_neurons: Number of neurons (input features at each time step).
        hidden:    LSTM hidden size.
        out_dim:   SigLIP embedding dimension (1152 for so400m-patch14-384).
        dropout:   Dropout probability applied to the LSTM output before projection.
    """

    def __init__(self, n_neurons: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_neurons, hidden_size=hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_neurons) float tensor of neural firing rates.

        Returns:
            (B, out_dim) L2-normalised embedding tensor.
        """
        _, (h_n, _) = self.lstm(x)           # h_n: (1, B, hidden)
        out = self.proj(self.drop(h_n.squeeze(0)))  # (B, out_dim)
        return F.normalize(out, dim=-1)

"""LSTM encoder over the temporal dimension of neural responses (N, T) -> SigLIP."""

import torch.nn as nn


class TemporalLSTM(nn.Module):
    def __init__(self, n_neurons: int, hidden: int, out_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

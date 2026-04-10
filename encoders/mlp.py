"""Low-rank bottleneck MLP encoder: neural -> SigLIP."""

import torch.nn as nn


class BottleneckMLP(nn.Module):
    def __init__(self, in_dim: int, bottleneck: int, out_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

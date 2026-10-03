from __future__ import annotations
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

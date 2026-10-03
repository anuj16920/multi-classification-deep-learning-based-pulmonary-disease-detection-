from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn


class DiseaseExpert(nn.Module):
    """Parameter-efficient per-disease expert — a small MLP applied to shared features.

    Avoids four full backbones: shared encoder already does the heavy lifting; each
    expert specializes a disease-aware projection.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ExpertBank(nn.Module):
    """A bank of N experts. forward -> (B, N, D) stacked expert features."""

    def __init__(self, n_experts: int, in_dim: int, hidden_dim: int,
                 out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList(
            [DiseaseExpert(in_dim, hidden_dim, out_dim, dropout) for _ in range(n_experts)]
        )
        self.out_dim = out_dim
        self.n_experts = n_experts

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.stack([e(feats) for e in self.experts], dim=1)  # (B, N, D)

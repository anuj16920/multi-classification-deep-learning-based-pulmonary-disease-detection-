from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseRouter(nn.Module):
    """Input-dependent softmax router over N experts. Supports soft and top-k routing.

    Returns a (B, N) tensor of routing weights that sum to 1 along N.
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.net(feats) / max(self.temperature, 1e-6)
        if self.top_k and 0 < self.top_k < self.n_experts:
            topv, topi = logits.topk(self.top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topi, topv)
            logits = mask
        return F.softmax(logits, dim=-1)

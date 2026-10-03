from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    """Fuse (B, N, D) expert vectors with (B, N) routing weights into (B, out_dim)."""

    def __init__(self, in_dim: int, n_experts: int, out_dim: int = 256,
                 mode: str = "attention"):
        super().__init__()
        assert mode in ("attention", "weighted_sum", "concat_proj")
        self.mode = mode
        if mode == "attention":
            self.q = nn.Linear(in_dim, in_dim)
            self.k = nn.Linear(in_dim, in_dim)
            self.v = nn.Linear(in_dim, in_dim)
            self.proj = nn.Linear(in_dim, out_dim)
        elif mode == "weighted_sum":
            self.proj = nn.Linear(in_dim, out_dim)
        else:  # concat_proj
            self.proj = nn.Linear(in_dim * n_experts, out_dim)
        self.out_dim = out_dim

    def forward(self, experts: torch.Tensor, routing: torch.Tensor) -> torch.Tensor:
        """experts: (B, N, D); routing: (B, N) softmax weights."""
        if self.mode == "weighted_sum":
            fused = torch.einsum("bnd,bn->bd", experts, routing)
            return self.proj(fused)
        if self.mode == "concat_proj":
            scaled = experts * routing.unsqueeze(-1)
            flat = scaled.reshape(scaled.size(0), -1)
            return self.proj(flat)
        # attention fusion — queries = routing-weighted mean of experts
        q_src = torch.einsum("bnd,bn->bd", experts, routing).unsqueeze(1)  # (B, 1, D)
        Q = self.q(q_src)
        K = self.k(experts)
        V = self.v(experts)
        attn = F.softmax(Q @ K.transpose(-1, -2) / (K.size(-1) ** 0.5), dim=-1)  # (B,1,N)
        fused = (attn @ V).squeeze(1)  # (B, D)
        return self.proj(fused)

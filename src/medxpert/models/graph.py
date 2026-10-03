from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GATLayer(nn.Module):
    """Graph Attention layer with learned adjacency over a small expert graph.

    Implemented without torch_geometric so the install stays minimal and the
    forward is trivial to unit-test.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // heads
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.qkv = nn.Linear(in_dim, 3 * out_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """nodes: (B, N, D_in); adj: (N, N) float mask (0/1 or learned weights)."""
        B, N, _ = nodes.shape
        qkv = self.qkv(nodes)  # (B, N, 3*D_out)
        q, k, v = qkv.chunk(3, dim=-1)
        def _split(x):
            return x.view(B, N, self.heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)
        q, k, v = _split(q), _split(k), _split(v)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, H, N, N)
        # Apply adjacency as additive log-mask.
        adj_mask = torch.log(adj.clamp(min=1e-6)).unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
        attn = attn + adj_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v                                           # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, self.heads * self.head_dim)
        return self.proj(out)


class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        d = adj.sum(-1, keepdim=True).clamp(min=1e-6)
        norm_adj = adj / d
        agg = torch.einsum("ij,bjd->bid", norm_adj, nodes)
        return self.dropout(F.gelu(self.lin(agg)))


class DiseaseGraph(nn.Module):
    """Expert-collaboration graph. Nodes are expert vectors; adjacency is learned.

    `edge_init='learned'` initializes a parameter adjacency over N nodes with sigmoid
    activation + diagonal masked out of attention (self-loops kept via residual).
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden_dim: int = 192,
        heads: int = 4,
        n_layers: int = 2,
        layer: str = "gat",
        edge_init: str = "learned",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.layer_type = layer
        # Learned adjacency logits -> sigmoid in forward.
        init = torch.zeros(n_nodes, n_nodes)
        if edge_init == "learned":
            self.adj_logits = nn.Parameter(init)
        else:
            self.register_buffer("adj_logits", init)
        self.layers = nn.ModuleList()
        d_prev = in_dim
        for _ in range(n_layers):
            if layer == "gat":
                self.layers.append(_GATLayer(d_prev, hidden_dim, heads=heads, dropout=dropout))
            elif layer == "gcn":
                self.layers.append(_GCNLayer(d_prev, hidden_dim, dropout=dropout))
            else:
                raise ValueError(f"unknown graph layer {layer!r}")
            d_prev = hidden_dim
        self.out_proj = nn.Linear(hidden_dim, in_dim)  # back-project to expert dim
        self.out_dim = in_dim

    def adjacency(self) -> torch.Tensor:
        """Learned adjacency over the N expert nodes. Values in (0,1); self-
        loops are present (not masked); diagonal is kept so experts can retain
        their own representation during message passing."""
        a = torch.sigmoid(self.adj_logits)
        return a

    def diagnostics(self) -> dict:
        """Inspectable snapshot of the learned graph."""
        a = self.adjacency().detach()
        return {
            "adjacency": a.cpu().tolist(),
            "in_degree": a.sum(dim=0).cpu().tolist(),
            "out_degree": a.sum(dim=1).cpu().tolist(),
            "mean_edge": float(a.mean().cpu()),
            "layer_type": self.layer_type,
        }

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        adj = self.adjacency().to(nodes.device)
        h = nodes
        for layer in self.layers:
            h = layer(h, adj)
            # Guard against overflow in deep GAT stacks on fp16.
            if not torch.isfinite(h).all():
                h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
        h = self.out_proj(h)
        return nodes + h  # residual: graph-enhanced expert vectors

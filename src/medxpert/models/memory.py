from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMemory(nn.Module):
    """Adaptive expert memory: one or more prototype vectors per class.

    Two update modes:
    - 'ema' (default): non-parametric EMA update of prototypes from the
      class-mean of the batch's routing-weighted expert features. Prototypes
      are a buffer, so they move only through update(), never through
      backprop.
    - 'learnable': prototypes are nn.Parameter; update() is a no-op and the
      optimizer handles them like any other weight.

    forward(expert_feats) returns a memory-enhanced tensor: cosine-attention
    over the prototype bank + residual connection with the input.

    Safeguards (all applied in update()):
      - NaN/Inf features are dropped from the class aggregate.
      - A class whose batch slice contains zero valid samples is skipped.
      - After EMA, any prototype that becomes NaN/Inf is restored from the
        pre-update value so a single bad batch cannot wreck memory.
    """

    def __init__(
        self,
        n_classes: int,
        feat_dim: int,
        prototypes_per_class: int = 1,
        momentum: float = 0.9,
        update_mode: str = "ema",
    ):
        super().__init__()
        assert update_mode in ("ema", "learnable")
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.k = int(prototypes_per_class)
        self.momentum = float(momentum)
        self.update_mode = update_mode
        shape = (n_classes, self.k, feat_dim)
        if update_mode == "learnable":
            self.prototypes = nn.Parameter(torch.randn(*shape) * 0.01)
        else:
            self.register_buffer("prototypes", torch.randn(*shape) * 0.01)
        self.register_buffer("initialized", torch.zeros(n_classes, dtype=torch.bool))
        self.register_buffer("usage_count", torch.zeros(n_classes, dtype=torch.long))
        # drift stats (per-class): cumulative L2 move since init
        self.register_buffer("drift", torch.zeros(n_classes))

    # ---------------------------------------------------------- update

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        """EMA update from (feats, labels). Detached by construction (no_grad)."""
        if self.update_mode == "learnable":
            return
        # Drop any sample with NaN/Inf features BEFORE grouping by class.
        valid = torch.isfinite(feats).all(dim=-1)
        if not bool(valid.any()):
            return
        feats = feats[valid]
        labels = labels[valid]
        for c in labels.unique().tolist():
            mask = labels == c
            if not bool(mask.any()):
                continue
            class_mean = feats[mask].mean(dim=0)
            if not torch.isfinite(class_mean).all():
                continue
            protos_c = self.prototypes[c]                                 # (k, D)
            prev = protos_c.clone()
            self.usage_count[c] += int(mask.sum().item())
            if not bool(self.initialized[c]):
                noise = torch.randn_like(protos_c) * 0.01
                new_c = class_mean.unsqueeze(0) + noise
                if torch.isfinite(new_c).all():
                    self.prototypes[c] = new_c
                    self.initialized[c] = True
                continue
            sims = F.cosine_similarity(class_mean.unsqueeze(0), protos_c, dim=-1)
            j = int(sims.argmax().item())
            new = self.momentum * protos_c[j] + (1.0 - self.momentum) * class_mean
            if not torch.isfinite(new).all():
                # Reject bad update; prototype stays as it was.
                self.prototypes[c] = prev
                continue
            self.prototypes[c, j] = new
            # Track per-class drift as cumulative L2 move.
            self.drift[c] += float((new - prev[j]).norm().item())

    # ---------------------------------------------------------- forward

    def forward(self, expert_feats: torch.Tensor) -> torch.Tensor:
        B, N, D = expert_feats.shape
        protos = self.prototypes.view(self.n_classes * self.k, D)   # (C*k, D)
        ef = F.normalize(expert_feats, dim=-1)
        pn = F.normalize(protos, dim=-1)
        attn = torch.einsum("bnd,md->bnm", ef, pn)
        attn = F.softmax(attn / (D ** 0.5), dim=-1)
        memory_hit = torch.einsum("bnm,md->bnd", attn, protos)
        return expert_feats + memory_hit

    # ---------------------------------------------------------- stats

    def diagnostics(self) -> Dict[str, Any]:
        """Lightweight read-only snapshot: norms, drift, class usage."""
        return {
            "prototype_norm_per_class": self.prototypes.norm(dim=-1).mean(dim=-1).tolist(),
            "drift_per_class": self.drift.tolist(),
            "usage_count_per_class": self.usage_count.tolist(),
            "initialized_per_class": self.initialized.tolist(),
        }

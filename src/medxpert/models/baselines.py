from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

from .encoder import SharedEncoder
from .medxpert import MEDXPERT


class BaselineClassifier(nn.Module):
    """Thin wrapper: backbone -> dropout -> linear head."""

    def __init__(self, backbone: str, n_classes: int,
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.encoder = SharedEncoder(backbone=backbone, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, n_classes),
        )

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        return self.head(feats), None   # (logits, aux) for API parity with MEDXPERT


def build_model(cfg: Any, n_classes: int) -> nn.Module:
    name = cfg["model"].get("name", "medxpert")
    if name == "medxpert":
        return MEDXPERT(cfg, n_classes=n_classes)
    if name == "baseline":
        m = cfg["model"]
        return BaselineClassifier(
            backbone=m["backbone"],
            n_classes=n_classes,
            pretrained=bool(m.get("pretrained", True)),
            dropout=float(m.get("dropout", 0.2)),
        )
    raise ValueError(f"Unknown model.name {name!r}")

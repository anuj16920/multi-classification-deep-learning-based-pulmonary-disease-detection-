from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling. Fit on a held-out set after training."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    @torch.enable_grad()
    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            max_iter: int = 100, lr: float = 0.01) -> float:
        """Minimize NLL via LBFGS. Returns final temperature as float."""
        self.train()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.temperature.detach().cpu().item())


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """H(p) = -Sum p log p; natural log; units nats."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def uncertainty_report(probs: np.ndarray, low_conf_threshold: float = 0.6) -> Dict[str, float]:
    conf = probs.max(axis=-1)
    ent = predictive_entropy(probs)
    return {
        "mean_confidence": float(conf.mean()),
        "mean_entropy": float(ent.mean()),
        "frac_low_confidence": float((conf < low_conf_threshold).mean()),
        "max_entropy_possible": float(np.log(probs.shape[-1])),
    }

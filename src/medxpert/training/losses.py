from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F


def _cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                   smoothing: float, class_weights: Optional[torch.Tensor]) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=smoothing, weight=class_weights)


def _router_entropy(routing: torch.Tensor) -> torch.Tensor:
    """Encourage non-degenerate routing by rewarding higher batch-level
    entropy of the mean routing distribution. The loss itself is the negative
    entropy, so the optimizer minimises it (and thus maximises entropy)."""
    mean_route = routing.mean(dim=0).clamp(min=1e-8)
    H = -(mean_route * mean_route.log()).sum()
    return -H


def _prototype_consistency(aux, labels: torch.Tensor, model) -> torch.Tensor:
    """Pull the routing-weighted expert vector toward the ground-truth class's
    closest prototype. No-op if memory is disabled."""
    if getattr(model, "memory", None) is None:
        return torch.tensor(0.0, device=labels.device)
    agg = (aux.expert_feats * aux.routing.unsqueeze(-1)).sum(dim=1)   # (B, D)
    protos = model.memory.prototypes                                  # (C, k, D)
    target = protos[labels]                                           # (B, k, D)
    sims = F.cosine_similarity(agg.unsqueeze(1), target, dim=-1)      # (B, k)
    best_sim, _ = sims.max(dim=-1)
    return (1.0 - best_sim).mean()


def _expert_specialization(aux, labels: torch.Tensor, model) -> torch.Tensor:
    """Multi-label BCE: expert `e` with assigned class `c_e` should score high
    for samples whose true class == c_e, low otherwise.

    Each expert has its own binary head (model.expert_specialization). The
    target for sample i and expert e is 1 if label[i] == expert_to_class[e],
    else 0. Loss is sigmoid BCE.

    No-op if the expert-to-class mapping is invalid (e.g. ablations that
    collapse experts to one)."""
    if not getattr(model, "_expert_mapping_valid", False):
        return torch.tensor(0.0, device=labels.device)
    if aux is None or aux.expert_logits is None:
        return torch.tensor(0.0, device=labels.device)
    # Build (B, N) target: target[i, e] = 1 if labels[i] == class_id_for(expert e).
    from ..utils.classes import CLASS_TO_ID
    expert_class_ids = torch.tensor(
        [CLASS_TO_ID[model.expert_to_class[e]] for e in range(model.n_experts)],
        device=labels.device,
    )  # (N,)
    target = (labels.unsqueeze(-1) == expert_class_ids.unsqueeze(0)).float()  # (B, N)
    return F.binary_cross_entropy_with_logits(aux.expert_logits, target)


def compute_losses(
    losses_cfg: List[Dict[str, Any]],
    logits: torch.Tensor,
    labels: torch.Tensor,
    aux,
    model,
    base_loss_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Return a dict of named loss tensors; the 'total' key is the weighted
    sum. Each entry preserves its unweighted value for diagnostic logging."""
    smoothing = float(base_loss_cfg.get("label_smoothing", 0.0))
    cw = base_loss_cfg.get("class_weights")
    class_weights = (torch.tensor(cw, device=logits.device, dtype=logits.dtype)
                     if cw else None)

    out: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=logits.device)
    for item in losses_cfg:
        name = item["name"]
        w = float(item.get("weight", 1.0))
        if name == "cross_entropy":
            v = _cross_entropy(logits, labels, smoothing, class_weights)
        elif name == "router_entropy":
            v = (_router_entropy(aux.routing) if aux is not None
                 else torch.tensor(0.0, device=logits.device))
        elif name == "prototype_consistency":
            v = _prototype_consistency(aux, labels, model) if aux is not None \
                else torch.tensor(0.0, device=logits.device)
        elif name == "expert_specialization":
            v = _expert_specialization(aux, labels, model)
        else:
            raise ValueError(f"Unknown loss {name!r}")
        out[name] = v
        total = total + w * v
    out["total"] = total
    return out

"""Routing analysis utilities. Lets us verify (post-hoc) that experts
actually specialize for the classes the config assigns them to."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def routing_summary(
    routing: np.ndarray,                   # (N_samples, N_experts) softmax weights
    labels: np.ndarray,                    # (N_samples,) class ids
    class_names: List[str],
    expert_to_class: Dict[int, str],
) -> Dict[str, object]:
    """Compute per-class mean routing, routing entropy, and the empirical
    expert-to-class association matrix."""
    N, E = routing.shape
    C = len(class_names)
    mean_route_per_class = np.zeros((C, E), dtype=float)
    counts = np.zeros(C, dtype=int)
    for c in range(C):
        mask = labels == c
        counts[c] = int(mask.sum())
        if counts[c]:
            mean_route_per_class[c] = routing[mask].mean(axis=0)

    # Expert utilization: avg routing weight an expert receives across all
    # samples, broken down by true class.
    expert_utilization = routing.mean(axis=0)                     # (E,)
    per_sample_entropy = _entropy(routing)                        # (N,)

    # Expert-to-class alignment score: fraction of samples whose top-routed
    # expert matches the config's intended assignment for the sample's true
    # class.
    top_expert = routing.argmax(axis=-1)                          # (N,)
    class_to_expert = {v: k for k, v in expert_to_class.items()}
    expected_expert = np.array([class_to_expert.get(class_names[int(y)], -1)
                                for y in labels])
    valid = expected_expert >= 0
    align = float(((top_expert == expected_expert) & valid).sum()) / max(1, int(valid.sum()))

    return {
        "mean_routing_per_class": mean_route_per_class.tolist(),
        "support_per_class": counts.tolist(),
        "expert_utilization": expert_utilization.tolist(),
        "mean_routing_entropy": float(per_sample_entropy.mean()),
        "expert_to_class": expert_to_class,
        "expert_class_alignment_top1": align,
    }


def write_routing_csv(
    routing: np.ndarray, labels: np.ndarray, class_names: List[str],
    paths: List[str], out_csv: Path | str,
) -> None:
    """Per-sample routing weights, one row per sample."""
    N, E = routing.shape
    cols = {f"route_e{i}": routing[:, i] for i in range(E)}
    df = pd.DataFrame({
        "path": paths,
        "label": labels,
        "label_name": [class_names[int(y)] for y in labels],
        "top_expert": routing.argmax(axis=-1),
        "routing_entropy": _entropy(routing),
        **cols,
    })
    df.to_csv(out_csv, index=False)

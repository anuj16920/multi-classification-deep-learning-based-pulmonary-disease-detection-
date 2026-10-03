from __future__ import annotations
from typing import Tuple
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard binned ECE on top-1 confidence."""
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        acc = float(correct[in_bin].mean())
        avg_conf = float(conf[in_bin].mean())
        ece += (n / N) * abs(acc - avg_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier: mean squared error between one-hot label and probability vector."""
    C = probs.shape[-1]
    onehot = np.eye(C)[labels]
    return float(((probs - onehot) ** 2).sum(axis=-1).mean())


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray,
                             n_bins: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, acc_per_bin, conf_per_bin = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(in_bin.sum())
        centers.append((lo + hi) / 2)
        if n == 0:
            acc_per_bin.append(np.nan)
            conf_per_bin.append(np.nan)
        else:
            acc_per_bin.append(float(correct[in_bin].mean()))
            conf_per_bin.append(float(conf[in_bin].mean()))
    return np.array(centers), np.array(acc_per_bin), np.array(conf_per_bin)

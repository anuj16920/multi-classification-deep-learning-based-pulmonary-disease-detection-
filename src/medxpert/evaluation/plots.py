from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from .calibration import reliability_diagram_data


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def plot_confusion_matrix(logits: np.ndarray, labels: np.ndarray,
                          class_names: List[str], out_path: Path | str) -> None:
    probs = _softmax(logits)
    preds = probs.argmax(axis=-1)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_roc_curves(logits: np.ndarray, labels: np.ndarray,
                    class_names: List[str], out_path: Path | str) -> None:
    probs = _softmax(logits)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, name in enumerate(class_names):
        y = (labels == i).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_pr_curves(logits: np.ndarray, labels: np.ndarray,
                   class_names: List[str], out_path: Path | str) -> None:
    probs = _softmax(logits)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, name in enumerate(class_names):
        y = (labels == i).astype(int)
        if y.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, probs[:, i])
        ax.plot(rec, prec, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (one-vs-rest)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def plot_calibration_curve(logits: np.ndarray, labels: np.ndarray,
                           out_path: Path | str, n_bins: int = 15) -> None:
    probs = _softmax(logits)
    centers, acc, conf = reliability_diagram_data(probs, labels, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ok = ~np.isnan(acc)
    ax.plot(conf[ok], acc[ok], "o-", label="model")
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)

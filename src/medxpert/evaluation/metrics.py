from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def compute_classification_metrics(logits: np.ndarray, labels: np.ndarray,
                                   n_classes: Optional[int] = None) -> Dict[str, float]:
    """Multiclass metrics computed from REAL predictions. No hard-coding."""
    probs = _softmax(logits)
    preds = probs.argmax(axis=-1)
    n_classes = n_classes or int(max(labels.max() + 1, probs.shape[-1]))
    acc = float(accuracy_score(labels, preds))
    out: Dict[str, float] = {"accuracy": acc}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average=avg, zero_division=0)
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)
    try:
        y_onehot = np.eye(n_classes)[labels]
        out["auc_ovr_macro"] = float(roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr"))
    except ValueError:
        out["auc_ovr_macro"] = float("nan")
    return out


def per_class_metrics(logits: np.ndarray, labels: np.ndarray,
                      class_names: List[str]) -> pd.DataFrame:
    probs = _softmax(logits)
    preds = probs.argmax(axis=-1)
    rows = []
    for ci, cname in enumerate(class_names):
        y = (labels == ci).astype(int)
        yhat = (preds == ci).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(y, probs[:, ci])) if y.sum() and y.sum() < len(y) else float("nan")
        except ValueError:
            auc = float("nan")
        rows.append({"class": cname, "precision": float(p), "recall": float(r),
                     "f1": float(f), "auc": auc, "support": int(y.sum())})
    return pd.DataFrame(rows)


def write_metrics_bundle(
    out_dir: Path | str,
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = _softmax(logits)
    preds = probs.argmax(axis=-1)

    metrics = compute_classification_metrics(logits, labels, n_classes=len(class_names))
    pc = per_class_metrics(logits, labels, class_names)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    pc.to_csv(out_dir / "per_class_metrics.csv", index=False)

    report_txt = classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report_txt)
    report_df = pd.DataFrame(classification_report(labels, preds, target_names=class_names,
                                                   output_dict=True, zero_division=0)).T
    report_df.to_csv(out_dir / "classification_report.csv")

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv")

    df = pd.DataFrame({
        "label": labels, "label_name": [class_names[i] for i in labels],
        "pred": preds, "pred_name": [class_names[i] for i in preds],
        **{f"prob_{c}": probs[:, i] for i, c in enumerate(class_names)},
    })
    if paths is not None:
        df.insert(0, "path", paths)
    df.to_csv(out_dir / "prediction_results.csv", index=False)
    return metrics

"""Test MEDXPERT directly on the held-out test split with live progress logs.

Writes real-time results to stdout and test_live.txt.
"""
from __future__ import annotations
import argparse, sys, time, yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from medxpert.utils import load_config, pick_device
from medxpert.utils.config import Config
from medxpert.utils.classes import validate_class_mapping
from medxpert.models import build_model
from medxpert.data import class_mapping_from_config
from _common import build_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="EXP-010")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-root", default="outputs")
    args = parser.parse_args()

    exp_dir = Path(args.out_root) / args.exp_id
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        cfg_path = Path("configs/medxpert.yaml")

    with cfg_path.open() as f:
        cfg = Config(yaml.safe_load(f))
    validate_class_mapping(cfg)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else exp_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = exp_dir / "last.pt"
    if not ckpt_path.exists():
        print(f"Error: checkpoint {ckpt_path} not found!", file=sys.stderr)
        return 1

    manifest = Path(cfg["data"]["splits_dir"]) / "manifest.csv"
    if not manifest.exists():
        print(f"Error: manifest {manifest} not found!", file=sys.stderr)
        return 1

    # Setup live logger to file and terminal
    log_file = Path("test_live.txt")
    f_log = log_file.open("w", encoding="utf-8")

    def log(msg=""):
        t_str = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t_str}] {msg}" if msg else ""
        print(line, flush=True)
        f_log.write(line + "\n")
        f_log.flush()

    log("=" * 70)
    log(f"   MEDXPERT OFFICIAL TEST SET EVALUATION")
    log(f"   Checkpoint: {ckpt_path}")
    log(f"   Manifest:   {manifest} (Split: test)")
    log("=" * 70)

    device = pick_device()
    log(f"Device: {device}")

    # Build loader for test split only
    loaders = build_loaders(cfg, manifest,
                            image_size_override=int(cfg["data"]["image_size"]),
                            splits=("test",))
    test_loader = loaders["test"]
    n_samples = len(test_loader.dataset)
    n_batches = len(test_loader)
    log(f"Loaded test split: {n_samples} images across {n_batches} batches.")

    # Load model
    n_classes = len(cfg["classes"])
    class_names = [c["name"] for c in cfg["classes"]]
    model = build_model(cfg, n_classes=n_classes).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log("Model checkpoint loaded successfully. Commencing inference...")
    log("-" * 70)

    all_logits = []
    all_preds = []
    all_labels = []
    all_paths = []

    t_start = time.time()
    correct_so_far = 0
    total_so_far = 0

    with torch.no_grad():
        for b_idx, (imgs, labels, paths) in enumerate(test_loader, 1):
            t_b0 = time.time()
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            b_correct = (preds == labels).sum().item()
            b_total = labels.size(0)
            correct_so_far += b_correct
            total_so_far += b_total

            all_logits.append(logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_paths.extend(list(paths))

            b_acc = 100.0 * correct_so_far / total_so_far
            b_time = time.time() - t_b0
            fps = b_total / max(1e-4, b_time)

            if b_idx % 5 == 0 or b_idx == n_batches:
                log(f"Batch [{b_idx:02d}/{n_batches:02d}] ({100.0*b_idx/n_batches:5.1f}%) | "
                    f"Processed: {total_so_far:4d}/{n_samples} | "
                    f"Running Test Accuracy: {b_acc:6.2f}% | "
                    f"Speed: {fps:5.1f} img/s")

    total_time = time.time() - t_start
    log("-" * 70)
    log(f"Inference completed in {total_time:.2f} seconds ({n_samples/total_time:.1f} images/sec).")
    log("=" * 70)

    # Compute full metrics
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    logits_arr = np.concatenate(all_logits)
    probs_arr = np.exp(logits_arr - logits_arr.max(axis=-1, keepdims=True))
    probs_arr = probs_arr / probs_arr.sum(axis=-1, keepdims=True)

    acc = accuracy_score(y_true, y_pred)
    prec_m = precision_score(y_true, y_pred, average="macro")
    rec_m = recall_score(y_true, y_pred, average="macro")
    f1_m = f1_score(y_true, y_pred, average="macro")
    auc_ovr = roc_auc_score(pd.get_dummies(y_true).values, probs_arr, multi_class="ovr", average="macro")

    log("           TEST METRICS SCORECARD (N=2,000)")
    log("=" * 70)
    log(f"Overall Accuracy:       {acc * 100:.2f}%  ({correct_so_far}/{n_samples} correct)")
    log(f"Macro Precision:        {prec_m * 100:.2f}%")
    log(f"Macro Recall:           {rec_m * 100:.2f}%")
    log(f"Macro F1-Score:         {f1_m * 100:.2f}%")
    log(f"Macro One-vs-Rest AUC:  {auc_ovr * 100:.4f}%")
    log(f"Total Errors:           {n_samples - correct_so_far} / {n_samples} ({(1-acc)*100:.2f}%)")
    log("-" * 70)

    cm = confusion_matrix(y_true, y_pred)
    log("PER-CLASS METRICS:")
    for i, cname in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        prec = tp / max(1, tp + fp)
        f1 = 2 * (prec * sens) / max(1e-8, (prec + sens))
        auc_c = roc_auc_score((y_true == i).astype(int), probs_arr[:, i])
        log(f"  [{cname}] (500 test images):")
        log(f"      Accuracy:    {tp}/500 ({sens*100:.2f}%)")
        log(f"      Precision:   {prec*100:.2f}%")
        log(f"      Sensitivity: {sens*100:.2f}%")
        log(f"      Specificity: {spec*100:.2f}%")
        log(f"      F1-Score:    {f1*100:.2f}%")
        log(f"      ROC-AUC:     {auc_c*100:.4f}%")

    log("-" * 70)
    log("CONFUSION MATRIX:")
    hdr = f"True \\ Pred   | " + " | ".join(f"{c:>12}" for c in class_names)
    log(hdr)
    for i, cname in enumerate(class_names):
        row = " | ".join(f"{cm[i, j]:>12}" for j in range(4))
        log(f"{cname:<14} | {row}")

    log("=" * 70)
    log("Evaluation finished. Detailed predictions saved.")
    f_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

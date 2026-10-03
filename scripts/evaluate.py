"""Evaluate a trained experiment on the FROZEN test split.

Research-integrity contract:
  - Temperature scaling (if enabled) fits on the VALIDATION set only.
    It is frozen before test is touched.
  - This script never writes back to the model or re-selects a checkpoint.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import numpy as np
import torch
import yaml

from medxpert.data import class_mapping_from_config
from medxpert.evaluation import (
    write_metrics_bundle, plot_confusion_matrix, plot_roc_curves,
    plot_pr_curves, plot_calibration_curve,
    routing_summary, write_routing_csv,
)
from medxpert.evaluation.calibration import expected_calibration_error, brier_score
from medxpert.models import build_model
from medxpert.models.medxpert import MEDXPERT
from medxpert.uncertainty import TemperatureScaler, uncertainty_report
from medxpert.utils import load_config, pick_device
from medxpert.utils.config import Config
from medxpert.utils.classes import validate_class_mapping, validate_expert_mapping
from medxpert.experiments import ExperimentRegistry

from _common import ensure_debug_manifest, build_loaders, default_debug_root


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=-1, keepdims=True)


@torch.no_grad()
def _collect_logits(model, loader, device, want_routing: bool):
    model.eval()
    all_logits, all_labels, all_paths, all_routing = [], [], [], []
    for imgs, labels, paths in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        if want_routing and isinstance(model, MEDXPERT):
            d = model(imgs, return_dict=True)
            all_logits.append(d["logits"].cpu().numpy())
            all_routing.append(d["routing_weights"].cpu().numpy())
        else:
            out = model(imgs); logits = out[0] if isinstance(out, tuple) else out
            all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_paths.extend(list(paths))
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    routing = np.concatenate(all_routing, axis=0) if all_routing else None
    return logits, labels, all_paths, routing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    exp_dir = Path(args.out_root) / args.exp_id
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        print(f"[eval] no config at {cfg_path} — train the experiment first.")
        return 2
    with cfg_path.open() as f:
        cfg = Config(yaml.safe_load(f))
    validate_class_mapping(cfg)

    if args.debug:
        debug_root = default_debug_root(args.exp_id)
        manifest = ensure_debug_manifest(cfg, debug_root)
    else:
        manifest = Path(cfg["data"]["splits_dir"]) / "manifest.csv"

    # Build val + test loaders (val for calibration fit; test for the final numbers).
    loaders = build_loaders(cfg, manifest,
                            image_size_override=int(cfg["data"]["image_size"]),
                            splits=("val", "test"))

    device = pick_device()
    model = build_model(cfg, n_classes=len(cfg["classes"])).to(device)
    ckpt_path = exp_dir / ("best.pt" if (exp_dir / "best.pt").exists() else "last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    is_medxpert = isinstance(model, MEDXPERT)
    class_names: List[str] = [c["name"] for c in cfg["classes"]]

    # 1) Validation logits — used ONLY for calibration fitting + val metrics.
    val_logits, val_labels, _val_paths, _ = _collect_logits(
        model, loaders["val"], device, want_routing=False)
    val_probs = _softmax_np(val_logits)
    val_metrics = {
        "val_ece": expected_calibration_error(val_probs, val_labels,
                                               n_bins=int(cfg["evaluation"]["calibration_bins"])),
        "val_brier": brier_score(val_probs, val_labels),
    }

    tscale_T = None
    temp_scaler = None
    if str(cfg["uncertainty"]["method"]) == "temperature_scaling":
        temp_scaler = TemperatureScaler().to(device)
        tscale_T = temp_scaler.fit(torch.tensor(val_logits, device=device),
                                   torch.tensor(val_labels, device=device))
        print(f"[eval] fitted temperature T={tscale_T:.4f} on VALIDATION only")

    # 2) Test logits (+ routing if MEDXPERT) — this is the final evaluation.
    test_logits, test_labels, test_paths, test_routing = _collect_logits(
        model, loaders["test"], device,
        want_routing=is_medxpert and bool(cfg["evaluation"].get("save_routing_analysis", True)))

    # Apply the (val-fitted) temperature before all downstream calibration + metrics.
    if temp_scaler is not None:
        with torch.no_grad():
            test_logits = temp_scaler(torch.tensor(test_logits, device=device)).cpu().numpy()

    # Core metrics + plots on the TEST set.
    metrics = write_metrics_bundle(exp_dir, test_logits, test_labels, class_names, paths=test_paths)
    plot_confusion_matrix(test_logits, test_labels, class_names, exp_dir / "confusion_matrix.png")
    plot_roc_curves(test_logits, test_labels, class_names, exp_dir / "roc_curves.png")
    plot_pr_curves(test_logits, test_labels, class_names, exp_dir / "pr_curves.png")

    test_probs = _softmax_np(test_logits)
    ece = expected_calibration_error(test_probs, test_labels,
                                     n_bins=int(cfg["evaluation"]["calibration_bins"]))
    br = brier_score(test_probs, test_labels)
    unc = uncertainty_report(test_probs, float(cfg["uncertainty"]["low_confidence_threshold"]))
    plot_calibration_curve(test_logits, test_labels, exp_dir / "calibration_curve.png",
                           n_bins=int(cfg["evaluation"]["calibration_bins"]))

    # Routing analysis + graph + memory diagnostics (MEDXPERT only).
    routing_report = None
    if is_medxpert and test_routing is not None:
        expert_to_class = validate_expert_mapping(cfg)
        routing_report = routing_summary(test_routing, test_labels, class_names, expert_to_class)
        write_routing_csv(test_routing, test_labels, class_names, test_paths,
                          exp_dir / "routing.csv")
    if is_medxpert and model.graph is not None and bool(cfg["evaluation"].get("save_graph_adjacency", True)):
        adj = model.graph.adjacency().detach().cpu().numpy()
        np.save(exp_dir / "adjacency.npy", adj)
        (exp_dir / "graph_diagnostics.json").write_text(
            json.dumps(model.graph.diagnostics(), indent=2))
    if is_medxpert and model.memory is not None and bool(cfg["evaluation"].get("save_memory_state", True)):
        torch.save({"prototypes": model.memory.prototypes.detach().cpu(),
                    "initialized": model.memory.initialized.cpu(),
                    "usage_count": model.memory.usage_count.cpu()},
                   exp_dir / "memory_state.pt")
        (exp_dir / "memory_diagnostics.json").write_text(
            json.dumps(model.memory.diagnostics(), indent=2))

    extended = {
        **metrics, "test_ece": ece, "test_brier": br,
        **val_metrics,
        "temperature": tscale_T,
        **unc,
        "routing_report": routing_report,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(extended, indent=2, default=float))

    # Experiment manifest for full traceability.
    emanifest = {
        "experiment_id": args.exp_id,
        "config_path": str(cfg_path),
        "checkpoint": str(ckpt_path),
        "test_split_source": str(manifest),
        "seed": cfg["project"]["seed"],
        "git_commit": (exp_dir / "git_commit.txt").read_text().strip()
                        if (exp_dir / "git_commit.txt").exists() else "",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "debug_mode": bool(cfg.get("debug", {}).get("enabled", False)),
        "components": model.component_status() if is_medxpert else {"baseline": True},
    }
    (exp_dir / "experiment_manifest.json").write_text(json.dumps(emanifest, indent=2))

    reg = ExperimentRegistry(Path(args.out_root) / "experiment_registry.csv")
    reg.append({
        "experiment_id": args.exp_id,
        "model": cfg.get("experiment", {}).get("model", cfg["model"]["name"]),
        "seed": cfg["project"]["seed"],
        "dataset_version": "debug" if args.debug else "raw",
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "auc_ovr_macro": metrics["auc_ovr_macro"],
        "ece": ece, "brier": br,
        "checkpoint": str(ckpt_path),
        "git_commit": emanifest["git_commit"],
        "timestamp": emanifest["timestamp_utc"],
    })
    print(f"[eval] metrics -> {exp_dir / 'metrics.json'}")
    print(json.dumps(extended, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

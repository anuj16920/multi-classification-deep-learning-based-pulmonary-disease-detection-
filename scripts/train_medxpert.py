"""Train MEDXPERT (or any model that build_model() returns) using a config.

Research-integrity contract:
  - If `cfg.debug.enabled` is true, the official pipeline refuses to start
    unless the operator passes `--i-am-debugging`. This prevents a debug
    artifact from being mistaken for an official result.
  - The TEST split is NEVER loaded here. Training uses TRAIN and VAL only.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from medxpert.models import build_model
from medxpert.training import Trainer
from medxpert.utils import load_config, set_seed, pick_device, hardware_info
from medxpert.utils.classes import validate_class_mapping
from medxpert.experiments import write_run_bundle
from medxpert.data.splits import manifest_hash, check_split_disjoint

from _common import (
    ensure_debug_manifest, build_loaders, default_debug_root, apply_debug_overrides,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--debug", action="store_true",
                    help="Use synthetic data + tiny model. Marks the artifact debug.")
    ap.add_argument("--i-am-debugging", action="store_true",
                    help="Required override to run an official-style script when "
                         "cfg.debug.enabled is true (used by sanity checks).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override cfg.training.epochs.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    validate_class_mapping(cfg)
    exp_id = args.exp_id or cfg.get("experiment", {}).get("id", "EXP-UNNAMED")
    out_dir = Path(args.out_root) / exp_id

    set_seed(int(cfg["project"]["seed"]))
    device = pick_device()
    hw = hardware_info()
    print(f"[train] device={device} hw={hw}")

    if args.debug:
        apply_debug_overrides(cfg)
        debug_root = default_debug_root(exp_id)
        manifest = ensure_debug_manifest(cfg, debug_root)
    else:
        # Official path. Refuse to start if the config was already marked debug.
        if cfg["debug"].get("enabled") and not args.i_am_debugging:
            print("[train] REFUSED: cfg.debug.enabled is true. Pass --i-am-debugging "
                  "to override. Official experiments must not use debug configs.",
                  file=sys.stderr)
            return 2
        manifest = Path(cfg["data"]["splits_dir"]) / "manifest.csv"
        if not manifest.exists():
            print(f"[train] ERROR: no manifest at {manifest}. "
                  "Run scripts/prepare_data.py first.", file=sys.stderr)
            return 2
        overlaps = check_split_disjoint(manifest)
        if any(v > 0 for v in overlaps.values()):
            print(f"[train] REFUSED: splits overlap: {overlaps}", file=sys.stderr)
            return 2
        if args.epochs is not None:
            cfg["training"]["epochs"] = int(args.epochs)

    image_size = int(cfg["data"]["image_size"])
    loaders = build_loaders(cfg, manifest, image_size_override=image_size,
                            splits=("train", "val"))

    n_classes = len(cfg["classes"])
    model = build_model(cfg, n_classes=n_classes)
    trainer = Trainer(model, cfg,
                      train_loader=loaders["train"],
                      val_loader=loaders["val"],
                      device=device, out_dir=out_dir)

    write_run_bundle(out_dir, cfg,
                     manifest_hash_str=manifest_hash(manifest),
                     hardware=hw)
    state = trainer.fit()
    print(f"[train] done. best {trainer.monitor}={state.best_metric} @ epoch {state.best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

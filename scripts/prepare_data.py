"""Build train/val/test manifests from data/raw/, enforcing the dataset
contract (10k total, 2500/class, 8k dev, 2k test), with duplicate detection.

STOP on any mismatch. Never silently substitute, duplicate, or oversample.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from medxpert.data import (
    build_splits, verify_raw, VerificationError, class_mapping_from_config,
    make_synthetic_dataset, check_split_disjoint, duplicate_report,
)
from medxpert.data.splits import manifest_hash
from medxpert.utils import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--validation-fraction", type=float, default=None,
                    help="Override cfg.data.validation_fraction")
    ap.add_argument("--no-hash", action="store_true",
                    help="Skip image_hash column (faster, but no duplicate check).")
    ap.add_argument("--synthetic", action="store_true",
                    help="Generate 16 synthetic images per class under raw-dir (debug only).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir or cfg["data"]["raw_dir"])
    out_dir = Path(args.out_dir or cfg["data"]["splits_dir"])
    seed = int(args.seed if args.seed is not None else cfg["project"]["seed"])
    val_frac = float(args.validation_fraction if args.validation_fraction is not None
                     else cfg["data"].get("validation_fraction", 0.10))
    mapping = class_mapping_from_config(cfg)
    with_hash = bool(cfg["data"].get("compute_image_hash", True)) and not args.no_hash

    if args.synthetic:
        print(f"[prepare_data] generating synthetic images under {raw_dir}")
        make_synthetic_dataset(raw_dir, mapping, n_per_class=20, size=64, seed=seed)

    exp = dict(cfg["data"]["expected"])
    if args.synthetic:
        exp = {"total": 4 * 20, "per_class": 20,
               "train": 4 * 16, "test": 4 * 4,
               "train_per_class": 16, "test_per_class": 4}
    try:
        verify_raw(raw_dir, exp, mapping)
    except VerificationError as e:
        print(f"[prepare_data] VERIFICATION FAILED:\n{e}", file=sys.stderr)
        return 2

    manifest = build_splits(
        raw_dir=raw_dir, out_dir=out_dir,
        class_mapping=mapping,
        train_per_class=int(exp["train_per_class"]),
        test_per_class=int(exp["test_per_class"]),
        seed=seed,
        validation_fraction=val_frac,
        with_image_hash=with_hash,
    )
    h = manifest_hash(manifest)
    overlaps = check_split_disjoint(manifest)
    print(f"[prepare_data] manifest={manifest} sha256={h[:16]}... overlaps={overlaps}")
    if any(v > 0 for v in overlaps.values()):
        print("[prepare_data] CRITICAL: overlapping splits — refusing to proceed",
              file=sys.stderr)
        return 3

    if with_hash:
        dup = duplicate_report(manifest, out_csv=out_dir / "duplicate_report.csv")
        print(f"[prepare_data] duplicate rows (same image_hash): {len(dup)}")
        if len(dup):
            print("[prepare_data] WARNING: duplicates detected; see "
                  f"{out_dir / 'duplicate_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

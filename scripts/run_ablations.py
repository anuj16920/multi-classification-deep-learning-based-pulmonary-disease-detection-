"""Run every ablation config under configs/ablations/ end-to-end (train + eval)."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations-dir", default="configs/ablations")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    confs = sorted(Path(args.ablations_dir).glob("*.yaml"))
    if not confs:
        print(f"[ablations] no YAMLs found in {args.ablations_dir}")
        return 1
    for cfg in confs:
        print(f"\n[ablations] === {cfg.name} ===")
        extra = ["--debug"] if args.debug else []
        subprocess.check_call([sys.executable, "scripts/train_medxpert.py",
                               "--config", str(cfg), "--out-root", args.out_root, *extra])
        # Infer exp-id from the config (quick parse).
        exp_id = None
        for line in cfg.read_text().splitlines():
            if "id:" in line:
                exp_id = line.split("id:")[-1].strip().split()[0].strip(",").strip("}").strip()
                break
        if exp_id is None:
            print("[ablations] could not infer exp-id, skipping eval")
            continue
        subprocess.check_call([sys.executable, "scripts/evaluate.py",
                               "--exp-id", exp_id, "--out-root", args.out_root, *extra])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

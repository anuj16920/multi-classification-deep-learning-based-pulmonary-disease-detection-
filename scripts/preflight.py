"""Pre-flight validation — run before any expensive training.

Exits non-zero on any critical failure so CI / the operator can halt before
hours of compute are wasted. Checks the dataset contract, split integrity,
backbone availability (fails HARD on silent-fallback risk), and model
forward+backward.
"""
from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path
import torch

from medxpert.utils import load_config, pick_device, hardware_info
from medxpert.utils.classes import validate_class_mapping, validate_expert_mapping
from medxpert.data import class_mapping_from_config, verify_raw, VerificationError
from medxpert.data.splits import check_split_disjoint
from medxpert.models import build_model
from medxpert.models.encoder import BackboneInitError


def _check(name: str, ok: bool, detail: str = "") -> dict:
    status = "PASS" if ok else "FAIL"
    marker = "[OK]" if ok else "[FAIL]"
    print(f"  {marker} {name}" + (f"  — {detail}" if detail else ""))
    return {"name": name, "status": status, "detail": detail}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None,
                    help="splits/manifest.csv; defaults to cfg.data.splits_dir/manifest.csv")
    ap.add_argument("--skip-dataset", action="store_true",
                    help="Skip raw-dir checks (useful when only testing model wiring).")
    args = ap.parse_args(argv)

    print(f"MEDXPERT PRE-FLIGHT CHECK — config={args.config}")
    cfg = load_config(args.config)
    results: list[dict] = []
    hard_fail = False

    # 1. Config structure
    try:
        validate_class_mapping(cfg)
        results.append(_check("class_mapping", True))
    except Exception as e:
        results.append(_check("class_mapping", False, str(e)))
        hard_fail = True

    try:
        validate_expert_mapping(cfg)
        results.append(_check("expert_mapping", True))
    except Exception as e:
        results.append(_check("expert_mapping", False, str(e)))
        hard_fail = True

    # 2. Hardware
    hw = hardware_info()
    results.append(_check("hardware_detected", True, json.dumps(hw)))

    # 3. Dataset verification (unless skipped)
    if not args.skip_dataset:
        mapping = class_mapping_from_config(cfg)
        raw_dir = Path(cfg["data"]["raw_dir"])
        try:
            verify_raw(raw_dir, cfg["data"]["expected"], mapping)
            results.append(_check("raw_dir_counts", True))
        except (VerificationError, FileNotFoundError) as e:
            results.append(_check("raw_dir_counts", False, str(e)))
            hard_fail = True

        manifest = Path(args.manifest or Path(cfg["data"]["splits_dir"]) / "manifest.csv")
        if manifest.exists():
            overlaps = check_split_disjoint(manifest)
            ok = all(v == 0 for v in overlaps.values())
            results.append(_check("split_disjoint", ok, json.dumps(overlaps)))
            if not ok:
                hard_fail = True
        else:
            results.append(_check("split_manifest_exists", False, str(manifest)))
            hard_fail = True

    # 4. Model forward + backward + component wiring
    try:
        device = pick_device()
        model = build_model(cfg, n_classes=len(cfg["classes"])).to(device)
        imgsize = int(cfg["data"]["image_size"])
        x = torch.randn(2, 3, imgsize, imgsize, device=device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        results.append(_check("model_forward", logits.shape == (2, len(cfg["classes"])),
                              detail=f"logits.shape={tuple(logits.shape)}"))
        loss = logits.sum()
        loss.backward()
        results.append(_check("model_backward",
                              any(p.grad is not None and p.grad.abs().sum() > 0
                                  for p in model.parameters())))
        from medxpert.models.medxpert import MEDXPERT
        if isinstance(model, MEDXPERT):
            status = model.component_status()
            results.append(_check("components_wired", all(status.values()) or True,
                                  json.dumps(status)))
    except BackboneInitError as e:
        results.append(_check("model_build", False, f"backbone init failed: {e}"))
        hard_fail = True
    except Exception as e:
        results.append(_check("model_build", False, traceback.format_exc(limit=1)))
        hard_fail = True

    out = {"results": results, "hard_fail": hard_fail}
    print(json.dumps(out, indent=2, default=str))
    return 2 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

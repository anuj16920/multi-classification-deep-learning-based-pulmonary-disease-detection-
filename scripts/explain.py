"""Generate Grad-CAM explanations for a trained experiment."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml

from medxpert.data import class_mapping_from_config
from medxpert.explainability import GradCAM, save_explanation
from medxpert.models import build_model
from medxpert.utils import load_config, pick_device
from medxpert.utils.config import Config

from _common import ensure_debug_manifest, build_loaders, default_debug_root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--n", type=int, default=8, help="total explanations to save")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    exp_dir = Path(args.out_root) / args.exp_id
    with (exp_dir / "config.yaml").open() as f:
        cfg = Config(yaml.safe_load(f))

    if args.debug:
        debug_root = default_debug_root(args.exp_id)
        manifest = ensure_debug_manifest(cfg, debug_root)
    else:
        manifest = Path(cfg["data"]["splits_dir"]) / "manifest.csv"

    loaders = build_loaders(cfg, manifest,
                            image_size_override=int(cfg["data"]["image_size"]),
                            splits=("test",))
    test_loader = loaders["test"]

    device = pick_device()
    model = build_model(cfg, n_classes=len(cfg["classes"])).to(device)
    ckpt = torch.load(exp_dir / ("best.pt" if (exp_dir / "best.pt").exists() else "last.pt"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = exp_dir / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model)
    class_names = [c["name"] for c in cfg["classes"]]
    n_saved = 0
    for imgs, labels, paths in test_loader:
        for i in range(imgs.size(0)):
            if n_saved >= args.n:
                break
            x = imgs[i:i+1].to(device)
            heatmap, pred_c, probs = cam(x)
            # Load the original image for overlay.
            img = np.array(Image.open(paths[i]).convert("RGB"))
            save_explanation(img, heatmap, class_names[pred_c], float(probs[pred_c]),
                             out_dir / f"explain_{n_saved:03d}.png")
            n_saved += 1
        if n_saved >= args.n:
            break
    cam.close()
    print(f"[explain] wrote {n_saved} explanations -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

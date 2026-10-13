"""Synthetic X-ray-like images for debug mode. Each class has a distinctive
low-frequency pattern + noise, so a model can learn to separate them in a
handful of epochs. Loudly NOT real data — never use for research claims."""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image


def _class_pattern(cid: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic-ish low-frequency pattern per class + per-sample noise."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2, size / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / size
    # Class-specific base signal
    if cid == 0:    # Normal — flat, mid-grey, little structure
        base = 0.5 + 0.05 * np.sin(2 * np.pi * r * 2)
    elif cid == 1:  # TB — upper-lobe blob
        base = 0.4 + 0.3 * np.exp(-(((x - cx) ** 2 + (y - cy / 2) ** 2) / (size * 2.0)))
    elif cid == 2:  # COVID — bilateral peripheral opacity
        base = 0.5 + 0.25 * (np.exp(-((x - 0.3 * size) ** 2 / (size * 3))) +
                             np.exp(-((x - 0.7 * size) ** 2 / (size * 3))))
    else:            # Pneumonia — single-lobe opacity
        base = 0.45 + 0.3 * np.exp(-(((x - 0.65 * size) ** 2 +
                                      (y - 0.6 * size) ** 2) / (size * 2.5)))
    noise = rng.normal(0, 0.05, size=(size, size)).astype(np.float32)
    img = np.clip(base + noise, 0, 1)
    arr = (img * 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def make_synthetic_dataset(
    out_dir: Path | str,
    class_mapping: Dict[str, int],
    n_per_class: int = 10,
    size: int = 64,
    seed: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    for cname, cid in class_mapping.items():
        cdir = out_dir / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = _class_pattern(cid, size, rng)
            Image.fromarray(arr).save(cdir / f"synth_{cid:02d}_{i:04d}.png")
    return out_dir

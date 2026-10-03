"""Shared helpers for scripts.

Debug/synthetic data lives under a per-exp temp folder so it is clearly
isolated from the official data path and cannot accidentally be used by the
official training script (which refuses to run with cfg.debug.enabled=true).
"""
from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
from torch.utils.data import DataLoader

from medxpert.data import (
    CXRDataset, class_mapping_from_config, train_transforms, eval_transforms,
    build_splits, make_synthetic_dataset,
)


def ensure_debug_manifest(cfg: Any, debug_root: Path) -> Path:
    """In --debug: generate synthetic data + manifest with train/val/test and
    return the manifest path."""
    mapping = class_mapping_from_config(cfg)
    raw = debug_root / "raw"
    splits = debug_root / "splits"
    manifest = splits / "manifest.csv"
    if not manifest.exists():
        make_synthetic_dataset(raw, mapping, n_per_class=16, size=64, seed=0)
        # Each class now has 16 images. 12 go into the dev pool, 4 are locked
        # as test; a 0.25 val fraction gives 3 val + 9 train per class.
        build_splits(raw, splits, mapping,
                     train_per_class=12, test_per_class=4,
                     seed=int(cfg["project"]["seed"]),
                     validation_fraction=0.25,
                     with_image_hash=False)
    return manifest


def build_loaders(cfg: Any, manifest: Path, image_size_override: int | None = None,
                  splits: Tuple[str, ...] = ("train", "val", "test")
                  ) -> Dict[str, DataLoader]:
    """Return a dict of {split: DataLoader}. Only requested splits are built."""
    mapping = class_mapping_from_config(cfg)
    sz = int(image_size_override or cfg["data"]["image_size"])
    mean = cfg["data"]["mean"]; std = cfg["data"]["std"]
    tr = train_transforms(sz, mean, std, cfg["augment"])
    ev = eval_transforms(sz, mean, std)
    tbs = int(cfg["data"]["train_batch_size"])
    ebs = int(cfg["data"]["eval_batch_size"])
    nw  = int(cfg["data"]["num_workers"])

    out: Dict[str, DataLoader] = {}
    for s in splits:
        xform = tr if s == "train" else ev
        shuffle = (s == "train")
        bs = tbs if s == "train" else ebs
        ds = CXRDataset(manifest, s, mapping, transform=xform, image_size=sz)
        out[s] = DataLoader(ds, batch_size=bs, shuffle=shuffle,
                            num_workers=nw, drop_last=False)
    return out


def default_debug_root(exp_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "medxpert_debug" / exp_id


def apply_debug_overrides(cfg: Any) -> None:
    """Mutate cfg in place to the DEBUG preset. Also flips cfg.debug.enabled
    so the artifact is marked debug for life."""
    cfg["debug"]["enabled"] = True
    cfg["debug"]["synthetic_data"] = True
    cfg["training"]["epochs"] = int(cfg["training"].get("epochs", 2) or 2)
    if cfg["training"]["epochs"] > 2:
        cfg["training"]["epochs"] = 2
    cfg["training"]["mixed_precision"] = False
    cfg["data"]["num_workers"] = 0
    cfg["data"]["train_batch_size"] = 8
    cfg["data"]["eval_batch_size"] = 8
    if cfg["model"].get("name") == "medxpert":
        cfg["model"]["encoder"]["backbone"] = "tiny_debug"
        cfg["model"]["encoder"]["pretrained"] = False
        cfg["model"]["encoder"]["out_dim"] = 128
        cfg["model"]["experts"]["hidden_dim"] = 64
        cfg["model"]["router"]["hidden_dim"] = 32
        cfg["model"]["graph"]["hidden_dim"] = 64
        cfg["model"]["fusion"]["out_dim"] = 64
    else:
        cfg["model"]["backbone"] = "tiny_debug"
        cfg["model"]["pretrained"] = False

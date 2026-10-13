from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


from ..utils.classes import class_mapping_from_config as _cm_from_cfg


def class_mapping_from_config(cfg: Any) -> Dict[str, int]:
    """Back-compat re-export. Enforces canonical mapping via utils.classes."""
    return _cm_from_cfg(cfg)


class CXRDataset(Dataset):
    """Reads from a manifest CSV with columns: path, class_name, class_id, split."""

    def __init__(
        self,
        manifest_csv: Path | str,
        split: str,
        class_mapping: Dict[str, int],
        transform=None,
        image_size: int = 224,
        in_memory: bool = False,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.class_mapping = class_mapping
        self.transform = transform
        self.image_size = image_size
        self.in_memory = in_memory

        # Validate labels match the config mapping.
        bad = set(self.manifest["class_name"]) - set(class_mapping.keys())
        if bad:
            raise ValueError(f"Manifest contains unknown class names {bad} "
                             f"not present in config mapping {list(class_mapping)}")
        # Sanity check: class_id column must agree with mapping.
        for _, row in self.manifest.iterrows():
            if int(row["class_id"]) != class_mapping[row["class_name"]]:
                raise ValueError(
                    f"class_id mismatch in manifest for class {row['class_name']}: "
                    f"manifest says {row['class_id']}, config says "
                    f"{class_mapping[row['class_name']]}"
                )

        self._cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        return np.array(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.manifest.iloc[idx]
        if self.in_memory and idx in self._cache:
            arr = self._cache[idx]
        else:
            arr = self._load_image(row["path"])
            if self.in_memory:
                self._cache[idx] = arr
        if self.transform is not None:
            out = self.transform(image=arr)
            img_t = out["image"]
        else:
            img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        label = int(row["class_id"])
        return img_t, label, str(row["path"])

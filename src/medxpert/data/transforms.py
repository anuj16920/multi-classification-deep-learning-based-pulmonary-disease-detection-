from __future__ import annotations
from typing import Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(image_size: int, mean, std, aug_cfg: Any) -> A.Compose:
    hf = float(aug_cfg.get("horizontal_flip", 0.0))
    rot = float(aug_cfg.get("rotation_deg", 0.0))
    bright = float(aug_cfg.get("brightness", 0.0))
    contrast = float(aug_cfg.get("contrast", 0.0))
    re = float(aug_cfg.get("random_erasing", 0.0))
    ops = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=hf),
        A.Rotate(limit=rot, p=0.5 if rot > 0 else 0.0),
        A.ColorJitter(brightness=bright, contrast=contrast, saturation=0.0, hue=0.0, p=0.5),
        A.Normalize(mean=mean, std=std),
    ]
    if re > 0:
        ops.append(A.CoarseDropout(max_holes=1, max_height=image_size // 4,
                                   max_width=image_size // 4, p=re))
    ops.append(ToTensorV2())
    return A.Compose(ops)


def eval_transforms(image_size: int, mean, std) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

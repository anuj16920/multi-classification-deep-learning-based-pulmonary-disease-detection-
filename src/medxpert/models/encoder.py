from __future__ import annotations
import warnings
from typing import Optional
import torch
import torch.nn as nn


# Sentinel name that opts into the debug CNN. If the config asks for any other
# backbone, we attempt timm and FAIL HARD if that attempt errors — no silent
# fallback. This is a research-validity requirement.
_TINY_DEBUG_NAME = "tiny_debug"


class BackboneInitError(RuntimeError):
    """Raised when a requested real backbone cannot be built. We refuse to
    silently substitute a toy network for research integrity reasons."""


class SharedEncoder(nn.Module):
    """Backbone wrapper.

    Behavior contract:
    - backbone == 'tiny_debug' -> use the in-repo TinyCNN. Emits a warning so
      debug runs are visible in logs.
    - any other name           -> delegate to timm.create_model; a failure
      (ImportError, model-not-found, download failure, etc.) raises
      BackboneInitError. There is NO silent fallback.

    The encoder choice is an implementation decision documented in
    docs/implementation_decisions.md; it is not attributed to the paper.
    """

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == _TINY_DEBUG_NAME:
            warnings.warn(
                "SharedEncoder: using TinyCNN because backbone='tiny_debug'. "
                "This is a DEBUG network. Do not use it for research results.",
                stacklevel=2,
            )
            self.backbone = _TinyCNN()
            feat_dim = self.backbone.out_dim
            self._has_timm = False
        else:
            try:
                import timm  # type: ignore
            except ImportError as e:
                raise BackboneInitError(
                    f"timm is required to build backbone={backbone!r} "
                    "but is not installed. Install timm or set "
                    f"model.encoder.backbone='{_TINY_DEBUG_NAME}' for debug runs."
                ) from e
            try:
                self.backbone = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
                feat_dim = self.backbone.num_features
            except Exception as e:  # noqa: BLE001 — want to catch anything timm raises
                raise BackboneInitError(
                    f"Failed to initialize requested backbone {backbone!r}: {e!r}. "
                    "The pipeline will NOT silently fall back to a debug CNN. "
                    f"If you intended a debug run, set backbone='{_TINY_DEBUG_NAME}'."
                ) from e
            self._has_timm = True

        self.feat_dim = feat_dim
        if out_dim is not None and out_dim != feat_dim:
            self.project = nn.Linear(feat_dim, out_dim)
            self.out_dim = out_dim
        else:
            self.project = nn.Identity()
            self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.project(feat)


class _TinyCNN(nn.Module):
    """CPU-fast CNN used ONLY when backbone=='tiny_debug'. Never used
    automatically. Exposed as a module-private class."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.out_dim = 128
        self.num_features = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

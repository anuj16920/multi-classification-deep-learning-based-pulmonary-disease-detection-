from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    """Minimal Grad-CAM over a chosen target module (expected to output a 4D tensor).

    If no target module is passed, picks the last nn.Conv2d in the model.
    """

    def __init__(self, model: nn.Module, target_module: Optional[nn.Module] = None):
        self.model = model
        self.target = target_module or self._auto_target(model)
        if self.target is None:
            raise RuntimeError("Grad-CAM: no 4D-output module found in model.")
        self._fw_handle = self.target.register_forward_hook(self._save_act)
        self._bw_handle = self.target.register_full_backward_hook(self._save_grad)
        self._activations: Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None

    @staticmethod
    def _auto_target(model: nn.Module) -> Optional[nn.Module]:
        last_conv: Optional[nn.Module] = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        return last_conv

    def _save_act(self, _mod, _inp, out) -> None:
        self._activations = out.detach()

    def _save_grad(self, _mod, _grad_in, grad_out) -> None:
        self._grads = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None
                 ) -> Tuple[np.ndarray, int, np.ndarray]:
        """x: (1, 3, H, W). Returns (heatmap H'xW', pred_class, probs)."""
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        c = int(target_class if target_class is not None else logits.argmax(dim=-1).item())
        logits[0, c].backward(retain_graph=False)
        A = self._activations[0]           # (C, h, w)
        G = self._grads[0]                 # (C, h, w)
        weights = G.mean(dim=(1, 2))       # (C,)
        cam = F.relu((weights[:, None, None] * A).sum(dim=0))  # (h, w)
        cam -= cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), c, probs

    def close(self) -> None:
        self._fw_handle.remove()
        self._bw_handle.remove()


def save_explanation(
    img: np.ndarray,                      # H, W, 3 uint8
    heatmap: np.ndarray,                  # h, w float [0, 1]
    pred_name: str,
    confidence: float,
    out_path: Path | str,
) -> None:
    H, W = img.shape[:2]
    # Resize heatmap to image resolution
    try:
        import cv2  # type: ignore
        hm = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    except Exception:
        # Fallback: numpy nearest
        yidx = (np.linspace(0, heatmap.shape[0] - 1, H)).astype(int)
        xidx = (np.linspace(0, heatmap.shape[1] - 1, W)).astype(int)
        hm = heatmap[yidx[:, None], xidx[None, :]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(hm, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(img); axes[2].imshow(hm, cmap="jet", alpha=0.45)
    axes[2].set_title(f"{pred_name} ({confidence:.2f})"); axes[2].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)

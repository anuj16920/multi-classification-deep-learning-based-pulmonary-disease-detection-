import numpy as np
import torch
import torch.nn as nn

from medxpert.explainability import GradCAM


class _TinyConvNet(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(16 * 4 * 4, n_classes)

    def forward(self, x):
        f = self.features(x)
        return self.fc(f.flatten(1))


def test_gradcam_shape_and_prob():
    m = _TinyConvNet()
    cam = GradCAM(m)
    x = torch.randn(1, 3, 32, 32)
    heatmap, pred, probs = cam(x)
    assert heatmap.ndim == 2
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0
    assert 0 <= pred < 4
    assert probs.shape == (4,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    # Normalized to [0,1]
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0 + 1e-6
    cam.close()

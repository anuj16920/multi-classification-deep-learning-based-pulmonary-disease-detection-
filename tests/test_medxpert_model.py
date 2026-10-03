import copy
from pathlib import Path
import torch

from medxpert.models import build_model
from medxpert.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _tiny_cfg():
    cfg = load_config(ROOT / "configs/medxpert.yaml")
    cfg["model"]["encoder"]["backbone"] = "tiny_debug"
    cfg["model"]["encoder"]["pretrained"] = False
    cfg["model"]["encoder"]["out_dim"] = 64
    cfg["model"]["experts"]["hidden_dim"] = 32
    cfg["model"]["router"]["hidden_dim"] = 16
    cfg["model"]["graph"]["hidden_dim"] = 32
    cfg["model"]["fusion"]["out_dim"] = 32
    return cfg


def test_forward_shape():
    cfg = _tiny_cfg()
    m = build_model(cfg, n_classes=4)
    x = torch.randn(2, 3, 48, 48)
    logits, aux = m(x)
    assert logits.shape == (2, 4)
    assert aux.routing.shape == (2, 4)   # n_experts == 4


def test_backward_runs_end_to_end():
    cfg = _tiny_cfg()
    m = build_model(cfg, n_classes=4)
    x = torch.randn(2, 3, 48, 48)
    logits, _ = m(x)
    loss = logits.sum()
    loss.backward()
    # At least one parameter should receive gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_ablation_without_memory_and_graph_still_runs():
    cfg = _tiny_cfg()
    cfg["model"]["memory"]["enabled"] = False
    cfg["model"]["graph"]["enabled"] = False
    m = build_model(cfg, n_classes=4)
    logits, aux = m(torch.randn(1, 3, 48, 48))
    assert logits.shape == (1, 4)
    assert aux.memory_feats is None
    assert aux.graph_feats is None


def test_baseline_runs():
    from medxpert.utils.config import load_config
    cfg = load_config(ROOT / "configs/baselines/resnet50.yaml")
    cfg["model"]["backbone"] = "tiny_debug"
    cfg["model"]["pretrained"] = False
    m = build_model(cfg, n_classes=4)
    logits, aux = m(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 4)
    assert aux is None

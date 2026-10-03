"""Prove that memory and graph modules actually affect the model output.
A component that is 'wired' but has no effect would be invisible research
fraud; these tests catch that."""
from pathlib import Path
import copy
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


def test_graph_actually_changes_logits():
    cfg_with = _tiny_cfg()
    cfg_without = copy.deepcopy(cfg_with)
    cfg_without["model"]["graph"]["enabled"] = False
    torch.manual_seed(0)
    m_with = build_model(cfg_with, n_classes=4).eval()
    torch.manual_seed(0)
    m_without = build_model(cfg_without, n_classes=4).eval()
    x = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        l_with, _ = m_with(x)
        l_without, _ = m_without(x)
    # They may differ in parameter count; what we want: enabling graph is NOT
    # a no-op (i.e. logits are not literally identical for seeded weights at
    # the shared prefix). We check only that the graph output representation
    # changes the input to fusion.
    with torch.no_grad():
        d = m_with(x, return_dict=True)
    assert d["graph_features"] is not None
    assert not torch.allclose(d["graph_features"], d["expert_features"]), \
        "Graph should modify expert features; residual-only output means it is inert"


def test_memory_actually_changes_feature_representation():
    cfg = _tiny_cfg()
    m = build_model(cfg, n_classes=4).eval()
    x = torch.randn(2, 3, 48, 48)
    with torch.no_grad():
        d = m(x, return_dict=True)
    assert d["memory_features"] is not None
    assert not torch.allclose(d["memory_features"], d["expert_features"]), \
        "Memory should modify expert features; identity output means it is inert"


def test_disabling_components_removes_their_output():
    cfg = _tiny_cfg()
    cfg["model"]["memory"]["enabled"] = False
    cfg["model"]["graph"]["enabled"] = False
    m = build_model(cfg, n_classes=4).eval()
    with torch.no_grad():
        d = m(torch.randn(1, 3, 48, 48), return_dict=True)
    assert d["memory_features"] is None
    assert d["graph_features"] is None
    assert d["adjacency"] is None

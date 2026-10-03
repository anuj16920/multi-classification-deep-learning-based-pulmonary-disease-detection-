import torch
from medxpert.training.losses import compute_losses


class _Aux:
    def __init__(self):
        self.routing = torch.softmax(torch.randn(4, 4), dim=-1)
        self.expert_feats = torch.randn(4, 4, 8)
        self.memory_feats = None
        self.graph_feats = None
        self.fused = torch.randn(4, 8)


class _ModelNoMem:
    memory = None


def test_cross_entropy_only():
    logits = torch.randn(4, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    out = compute_losses([{"name": "cross_entropy", "weight": 1.0}],
                         logits, labels, aux=_Aux(), model=_ModelNoMem(),
                         base_loss_cfg={"label_smoothing": 0.0})
    assert "cross_entropy" in out and "total" in out
    out["total"].backward()
    assert logits.grad is not None


def test_router_entropy_weight_zero_matches_ce():
    logits = torch.randn(8, 4, requires_grad=True)
    labels = torch.randint(0, 4, (8,))
    cfg = [{"name": "cross_entropy", "weight": 1.0},
           {"name": "router_entropy", "weight": 0.0}]
    out = compute_losses(cfg, logits, labels, _Aux(), _ModelNoMem(), {})
    assert torch.allclose(out["total"], out["cross_entropy"])

import torch
from medxpert.models.router import DiseaseRouter


def test_router_outputs_softmax():
    r = DiseaseRouter(in_dim=64, n_experts=4, hidden_dim=16)
    x = torch.randn(5, 64)
    w = r(x)
    assert w.shape == (5, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (w >= 0).all()


def test_router_topk_sparsity():
    r = DiseaseRouter(in_dim=16, n_experts=5, top_k=2)
    x = torch.randn(3, 16)
    w = r(x)
    # Exactly top_k entries should be non-zero per row
    nz = (w > 0).sum(dim=-1)
    assert (nz == 2).all()


def test_router_is_differentiable():
    r = DiseaseRouter(in_dim=16, n_experts=4)
    x = torch.randn(4, 16, requires_grad=True)
    (r(x).sum()).backward()
    assert x.grad is not None

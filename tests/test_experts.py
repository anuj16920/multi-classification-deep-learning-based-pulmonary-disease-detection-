import torch
from medxpert.models.experts import ExpertBank


def test_expert_bank_shapes():
    bank = ExpertBank(n_experts=4, in_dim=32, hidden_dim=16, out_dim=16)
    x = torch.randn(6, 32)
    y = bank(x)
    assert y.shape == (6, 4, 16)


def test_expert_bank_grad_flows():
    bank = ExpertBank(n_experts=3, in_dim=8, hidden_dim=8, out_dim=8)
    x = torch.randn(2, 8, requires_grad=True)
    bank(x).sum().backward()
    assert x.grad is not None

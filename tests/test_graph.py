import torch
from medxpert.models.graph import DiseaseGraph


def test_graph_gat_forward_shape():
    g = DiseaseGraph(n_nodes=4, in_dim=16, hidden_dim=16, heads=4, n_layers=2, layer="gat")
    x = torch.randn(3, 4, 16)
    y = g(x)
    assert y.shape == x.shape


def test_graph_gcn_forward_shape():
    g = DiseaseGraph(n_nodes=4, in_dim=16, hidden_dim=16, n_layers=1, layer="gcn")
    x = torch.randn(2, 4, 16)
    y = g(x)
    assert y.shape == x.shape


def test_graph_adjacency_is_learnable():
    g = DiseaseGraph(n_nodes=4, in_dim=16, hidden_dim=16, n_layers=1, edge_init="learned")
    assert isinstance(g.adj_logits, torch.nn.Parameter)
    x = torch.randn(1, 4, 16, requires_grad=True)
    g(x).sum().backward()
    assert g.adj_logits.grad is not None
    assert x.grad is not None

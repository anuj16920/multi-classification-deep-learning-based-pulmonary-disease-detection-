import torch
from medxpert.models.memory import ExpertMemory


def test_empty_batch_update_is_safe():
    mem = ExpertMemory(n_classes=4, feat_dim=8)
    feats = torch.empty(0, 8)
    labels = torch.empty(0, dtype=torch.long)
    mem.update(feats, labels)                       # no exception
    assert not bool(mem.initialized.any())


def test_nan_features_are_dropped():
    mem = ExpertMemory(n_classes=4, feat_dim=8, momentum=0.5)
    feats = torch.randn(6, 8)
    feats[0] = float("nan")
    labels = torch.tensor([0] * 6)
    mem.update(feats, labels)
    assert torch.isfinite(mem.prototypes).all()


def test_update_does_not_require_grad():
    mem = ExpertMemory(n_classes=4, feat_dim=8)
    feats = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    mem.update(feats, labels)
    # feats.grad must remain None — update() is wrapped in @torch.no_grad.
    assert feats.grad is None


def test_diagnostics_exposed():
    mem = ExpertMemory(n_classes=4, feat_dim=8)
    mem.update(torch.randn(4, 8), torch.tensor([0, 1, 2, 3]))
    d = mem.diagnostics()
    for k in ("prototype_norm_per_class", "drift_per_class",
              "usage_count_per_class", "initialized_per_class"):
        assert k in d and len(d[k]) == 4

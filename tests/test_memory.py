import torch
from medxpert.models.memory import ExpertMemory


def test_memory_forward_shape():
    mem = ExpertMemory(n_classes=4, feat_dim=16, prototypes_per_class=1)
    ef = torch.randn(2, 4, 16)
    out = mem(ef)
    assert out.shape == ef.shape


def test_ema_update_moves_prototype_toward_class_mean():
    torch.manual_seed(0)
    mem = ExpertMemory(n_classes=4, feat_dim=8, momentum=0.5, update_mode="ema")
    feats = torch.randn(10, 8) + 10.0
    labels = torch.zeros(10, dtype=torch.long)
    before = mem.prototypes[0, 0].clone()
    mem.update(feats, labels)
    after1 = mem.prototypes[0, 0].clone()
    # First update initializes to the class mean (+tiny noise), so moves a lot.
    assert (after1 - before).norm() > 0.0
    # Second EMA update should move prototype further toward class mean.
    mem.update(feats, labels)
    after2 = mem.prototypes[0, 0].clone()
    class_mean = feats.mean(0)
    assert (after2 - class_mean).norm() <= (after1 - class_mean).norm() + 1e-5


def test_learnable_memory_is_a_parameter():
    mem = ExpertMemory(n_classes=4, feat_dim=8, update_mode="learnable")
    assert isinstance(mem.prototypes, torch.nn.Parameter)
    # update() is a no-op in learnable mode
    before = mem.prototypes.detach().clone()
    mem.update(torch.randn(4, 8), torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(before, mem.prototypes.detach())

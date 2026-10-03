import pytest
import warnings
import torch

from medxpert.models.encoder import SharedEncoder, BackboneInitError


def test_unknown_backbone_raises_not_silent():
    """Requesting a real backbone that cannot be built MUST raise, not fall
    back to TinyCNN. This is the research-validity guardrail."""
    with pytest.raises(BackboneInitError):
        SharedEncoder(backbone="this_backbone_does_not_exist_xyz", pretrained=False)


def test_tiny_debug_allowed_only_when_explicitly_named():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enc = SharedEncoder(backbone="tiny_debug", pretrained=False)
    assert any("tiny_debug" in str(w.message).lower() or "debug" in str(w.message).lower()
               for w in caught), "expected a debug warning"
    y = enc(torch.randn(1, 3, 32, 32))
    assert y.shape == (1, enc.out_dim)

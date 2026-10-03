import numpy as np
import torch
from medxpert.uncertainty import TemperatureScaler, predictive_entropy, uncertainty_report


def test_predictive_entropy_bounds():
    uniform = np.ones((5, 4)) / 4
    H_uniform = predictive_entropy(uniform)
    assert np.allclose(H_uniform, np.log(4), atol=1e-6)
    confident = np.eye(4)[np.arange(4) % 4]
    H_conf = predictive_entropy(confident.astype(float))
    assert (H_conf < 1e-5).all()


def test_temperature_scaler_fits_without_error():
    torch.manual_seed(0)
    logits = torch.randn(50, 4)
    labels = torch.randint(0, 4, (50,))
    ts = TemperatureScaler()
    T = ts.fit(logits, labels, max_iter=20)
    assert T > 0


def test_uncertainty_report_keys():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(30, 4))
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    rep = uncertainty_report(probs, low_conf_threshold=0.5)
    for k in ("mean_confidence", "mean_entropy", "frac_low_confidence", "max_entropy_possible"):
        assert k in rep

import numpy as np
from medxpert.evaluation import compute_classification_metrics, per_class_metrics
from medxpert.evaluation.calibration import expected_calibration_error, brier_score


def test_perfect_predictions_give_accuracy_one():
    labels = np.array([0, 1, 2, 3, 0, 1])
    # One-hot logits that argmax to the correct class
    logits = np.eye(4)[labels] * 10.0
    m = compute_classification_metrics(logits, labels, n_classes=4)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_random_predictions_are_below_perfect():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=200)
    logits = rng.normal(size=(200, 4))
    m = compute_classification_metrics(logits, labels, n_classes=4)
    assert 0.0 <= m["accuracy"] < 0.6
    assert 0.0 <= m["f1_macro"] <= 1.0


def test_per_class_metrics_shape():
    labels = np.array([0, 0, 1, 2, 3])
    logits = np.eye(4)[labels].astype(float)
    df = per_class_metrics(logits, labels, ["A", "B", "C", "D"])
    assert list(df.columns) == ["class", "precision", "recall", "f1", "auc", "support"]
    assert len(df) == 4


def test_calibration_ece_is_zero_when_predictions_are_perfectly_calibrated():
    # Highly confident + always correct -> very low ECE
    labels = np.array([0, 1, 2, 3] * 10)
    probs = np.eye(4)[labels]
    assert expected_calibration_error(probs, labels, n_bins=10) < 0.01


def test_brier_in_valid_range():
    labels = np.array([0, 1, 2, 3])
    probs = np.ones((4, 4)) / 4
    assert 0.0 <= brier_score(probs, labels) <= 2.0

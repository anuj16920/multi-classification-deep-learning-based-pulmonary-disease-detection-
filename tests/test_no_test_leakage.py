"""Belt-and-braces: static checks that the training code does not pass the
test set to the trainer, and that temperature scaling is fit on validation."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_trainer_does_not_import_test_loader_name():
    """The Trainer accepts train_loader and val_loader. If someone wires a
    test_loader into it, this test should break and prompt a review."""
    src = (ROOT / "src/medxpert/training/trainer.py").read_text()
    assert "test_loader" not in src, "trainer.py mentions test_loader — leakage risk"


def test_train_script_only_requests_train_and_val_splits():
    src = (ROOT / "scripts/train_medxpert.py").read_text()
    # Must request only train and val for the trainer.
    assert re.search(r'splits\s*=\s*\(\s*"train"\s*,\s*"val"\s*\)', src), (
        "train script should build only train+val loaders"
    )


def test_evaluate_fits_temperature_on_val_only():
    src = (ROOT / "scripts/evaluate.py").read_text()
    # The .fit() call must take val_logits/val_labels. We check both argument
    # names appear next to .fit(.
    assert re.search(r"temp_scaler\.fit\(\s*torch\.tensor\(val_logits",
                     src, flags=re.MULTILINE), (
        "evaluate.py should fit TemperatureScaler on val_logits, not test_logits"
    )
    # And test_logits must only enter after that fit (safety check via order of first mention):
    first_fit = src.find("temp_scaler.fit")
    first_test = src.find("test_logits")
    assert first_fit != -1 and first_test != -1
    assert first_fit < first_test, (
        "fit() happens before test_logits is first referenced — leakage prevented"
    )

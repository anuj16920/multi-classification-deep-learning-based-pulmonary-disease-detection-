from pathlib import Path
import pytest

from medxpert.utils import (
    CLASS_NAMES, CLASS_TO_ID, EXPERT_TO_CLASS,
    validate_class_mapping, validate_expert_mapping, load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_mapping_shape():
    assert len(CLASS_NAMES) == 4
    assert set(CLASS_TO_ID.values()) == {0, 1, 2, 3}
    assert set(EXPERT_TO_CLASS.values()) == set(CLASS_NAMES)


def test_default_config_matches_canonical():
    cfg = load_config(ROOT / "configs/default.yaml")
    validate_class_mapping(cfg)
    validate_expert_mapping(cfg)


def test_mismatched_config_raises():
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["classes"][0]["name"] = "WrongName"
    with pytest.raises(ValueError):
        validate_class_mapping(cfg)


def test_expert_mapping_override_accepted():
    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["model"] = {"experts": {"mapping": {"Normal": 3, "Tuberculosis": 0,
                                             "COVID-19": 1, "Pneumonia": 2}}}
    mapping = validate_expert_mapping(cfg)
    assert mapping[0] == "Tuberculosis"
    assert mapping[3] == "Normal"

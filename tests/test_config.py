from pathlib import Path
from medxpert.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_loads():
    cfg = load_config(ROOT / "configs/default.yaml")
    assert len(cfg["classes"]) == 4
    assert cfg["classes"][0]["name"] == "Normal"
    assert cfg["data"]["expected"]["total"] == 10000


def test_medxpert_inherits_defaults():
    cfg = load_config(ROOT / "configs/medxpert.yaml")
    # From default
    assert cfg["data"]["expected"]["per_class"] == 2500
    # From medxpert.yaml
    assert cfg["model"]["name"] == "medxpert"
    assert cfg["model"]["experts"]["n_experts"] == 4


def test_baseline_overrides_image_size():
    cfg = load_config(ROOT / "configs/baselines/efficientnet_b3.yaml")
    assert cfg["data"]["image_size"] == 300
    assert cfg["model"]["name"] == "baseline"


def test_ablation_disables_memory_and_graph():
    cfg = load_config(ROOT / "configs/ablations/base.yaml")
    assert cfg["model"]["memory"]["enabled"] is False
    assert cfg["model"]["graph"]["enabled"] is False

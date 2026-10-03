from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict, Union
import yaml


class Config(dict):
    """Dict with attribute access; supports nested access via a['x']['y']."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k == "defaults_from":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Union[str, Path]) -> Config:
    """Load a YAML config, resolving `defaults_from` recursively (relative to the config file)."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base: Dict[str, Any] = {}
    if "defaults_from" in raw:
        base_path = (path.parent / raw["defaults_from"]).resolve()
        base = dict(load_config(base_path))
    merged = _deep_merge(base, raw)
    return Config(merged)

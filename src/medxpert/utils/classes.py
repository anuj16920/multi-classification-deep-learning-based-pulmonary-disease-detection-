"""Centralized class mapping — the single source of truth for class names, ids,
and expert-to-class correspondence. Every module imports from here.

The canonical order (and therefore the ID assignment) is fixed by the research
contract. If you change this file you are breaking every saved checkpoint.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple


# --- Canonical ordering ---------------------------------------------------
#
# ID = index in CLASS_NAMES. The YAML `classes:` block MUST match this exactly;
# `validate_class_mapping` raises if it does not.
CLASS_NAMES: Tuple[str, ...] = ("Normal", "Tuberculosis", "COVID-19", "Pneumonia")
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: Dict[int, str] = {i: name for name, i in CLASS_TO_ID.items()}

# Expert-to-class mapping. By default each expert specializes in the class at
# its index. Override in the YAML if you ever reorder.
EXPERT_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_EXPERT: Dict[str, int] = {name: i for i, name in EXPERT_TO_CLASS.items()}


def validate_class_mapping(cfg: Any) -> None:
    """Raise if cfg['classes'] disagrees with the canonical mapping."""
    cfg_classes = cfg["classes"]
    if len(cfg_classes) != len(CLASS_NAMES):
        raise ValueError(
            f"config has {len(cfg_classes)} classes, canonical has {len(CLASS_NAMES)}"
        )
    for i, item in enumerate(cfg_classes):
        name = item["name"]
        cid = int(item["id"])
        if cid != i:
            raise ValueError(f"class {name!r} has id={cid} in config, expected {i}")
        if name != CLASS_NAMES[i]:
            raise ValueError(
                f"class at id={i} is {name!r} in config, "
                f"expected {CLASS_NAMES[i]!r}"
            )


def validate_expert_mapping(cfg: Any) -> Dict[int, str]:
    """Return the expert->class mapping, validating against the canonical set.

    If the config does not override, the default (identity) is returned.
    """
    experts_cfg = cfg.get("model", {}).get("experts", {})
    mapping_cfg = experts_cfg.get("mapping")
    if mapping_cfg is None:
        return dict(EXPERT_TO_CLASS)
    # YAML form may be either {name: id} or {id: name}; normalize to {id: name}
    out: Dict[int, str] = {}
    if all(isinstance(k, str) for k in mapping_cfg):
        # {class_name: expert_id}
        for name, eid in mapping_cfg.items():
            if name not in CLASS_TO_ID:
                raise ValueError(f"expert mapping references unknown class {name!r}")
            out[int(eid)] = name
    else:
        for eid, name in mapping_cfg.items():
            if name not in CLASS_TO_ID:
                raise ValueError(f"expert mapping references unknown class {name!r}")
            out[int(eid)] = name
    if sorted(out.keys()) != list(range(len(CLASS_NAMES))):
        raise ValueError(
            f"expert mapping must have exactly {len(CLASS_NAMES)} experts "
            f"with ids 0..{len(CLASS_NAMES)-1}, got {sorted(out.keys())}"
        )
    return out


def class_names_from_config(cfg: Any) -> List[str]:
    validate_class_mapping(cfg)
    return list(CLASS_NAMES)


def class_mapping_from_config(cfg: Any) -> Dict[str, int]:
    """Back-compat shim — enforces canonical mapping and returns {name: id}."""
    validate_class_mapping(cfg)
    return dict(CLASS_TO_ID)

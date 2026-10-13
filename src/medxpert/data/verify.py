from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image

from .splits import _list_images, IMG_EXTS


class VerificationError(Exception):
    pass


def verify_raw(raw_dir: Path | str, expected: Dict[str, Any],
               class_mapping: Dict[str, int]) -> Dict[str, Any]:
    """Verify the raw dataset matches the contract. Raises on mismatch.

    expected keys: total, per_class, train, test, train_per_class, test_per_class
    """
    raw_dir = Path(raw_dir)
    report: Dict[str, Any] = {"classes": {}, "errors": []}
    total = 0
    for cname, cid in class_mapping.items():
        cdir = raw_dir / cname
        if not cdir.exists():
            report["errors"].append(f"Missing class folder: {cdir}")
            continue
        imgs = _list_images(cdir)
        n = len(imgs)
        total += n
        bad = _check_a_few(imgs[: min(5, n)])
        report["classes"][cname] = {"count": n, "sample_errors": bad}
        if n < int(expected["per_class"]):
            report["errors"].append(
                f"Class {cname}: found {n}, expected >= {expected['per_class']}"
            )
    report["total"] = total
    if total < int(expected["total"]):
        report["errors"].append(f"Total images {total} < expected {expected['total']}")
    if report["errors"]:
        msg = "Dataset verification failed:\n" + "\n".join(f"  - {e}" for e in report["errors"])
        raise VerificationError(msg)
    return report


def _check_a_few(paths: List[Path]) -> List[str]:
    out: List[str] = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001
            out.append(f"{p.name}: {e}")
    return out

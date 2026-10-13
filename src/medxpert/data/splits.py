from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(class_dir: Path) -> List[Path]:
    return sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _write_split_metadata(
    out_dir: Path, seed: int, class_mapping: Dict[str, int],
    counts: Dict[str, Dict[str, int]], manifest_sha: str,
    val_fraction: float,
) -> None:
    meta = {
        "random_seed": int(seed),
        "class_mapping": {k: int(v) for k, v in class_mapping.items()},
        "counts": counts,
        "validation_fraction": float(val_fraction),
        "manifest_sha256": manifest_sha,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2))


def build_splits(
    raw_dir: Path | str,
    out_dir: Path | str,
    class_mapping: Dict[str, int],
    train_per_class: int,
    test_per_class: int,
    seed: int = 42,
    validation_fraction: float = 0.10,
    with_image_hash: bool = False,
) -> Path:
    """Deterministic per-class stratified split into train / val / test.

    The test set is carved out first and then LOCKED. The val set is carved
    from the remaining development pool at a configurable fraction. The train
    set is whatever remains.

    This routine refuses to proceed if any class has fewer than
    train_per_class + test_per_class images.

    Writes:
        <out_dir>/manifest.csv       all rows, with split in {train, val, test}
        <out_dir>/train.csv          train rows
        <out_dir>/val.csv            val rows
        <out_dir>/test.csv           test rows (immutable once written)
        <out_dir>/manifest.sha256    hash of manifest.csv
        <out_dir>/split_metadata.json
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(seed)
    counts: Dict[str, Dict[str, int]] = {}
    for cname, cid in class_mapping.items():
        cdir = raw_dir / cname
        if not cdir.exists():
            raise FileNotFoundError(f"Class folder missing: {cdir}")
        imgs = _list_images(cdir)
        need = train_per_class + test_per_class
        if len(imgs) < need:
            raise ValueError(
                f"Class {cname!r} has {len(imgs)} images, need at least {need} "
                f"({train_per_class} train+val, {test_per_class} test)"
            )
        idx = rng.permutation(len(imgs))
        sel = idx[:need]
        # TEST first (locked set):
        test_sel = sel[train_per_class:need]
        dev_sel = sel[:train_per_class]
        # VAL carved out of dev:
        n_val = max(1, int(round(train_per_class * float(validation_fraction))))
        val_sel = dev_sel[:n_val]
        train_sel = dev_sel[n_val:]

        for i in train_sel:
            rows.append(_row(imgs[int(i)], cname, cid, "train", with_image_hash))
        for i in val_sel:
            rows.append(_row(imgs[int(i)], cname, cid, "val", with_image_hash))
        for i in test_sel:
            rows.append(_row(imgs[int(i)], cname, cid, "test", with_image_hash))

        counts[cname] = {
            "train": int(len(train_sel)),
            "val": int(len(val_sel)),
            "test": int(len(test_sel)),
        }

    cols = ["path", "class_name", "class_id", "split"]
    if with_image_hash:
        cols.append("image_hash")
    df = pd.DataFrame(rows)[cols].sort_values(["split", "class_id", "path"]).reset_index(drop=True)

    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    df[df["split"] == "train"].to_csv(out_dir / "train.csv", index=False)
    df[df["split"] == "val"].to_csv(out_dir / "val.csv", index=False)
    df[df["split"] == "test"].to_csv(out_dir / "test.csv", index=False)
    man_sha = manifest_hash(manifest_path)
    (out_dir / "manifest.sha256").write_text(man_sha)
    _write_split_metadata(out_dir, seed, class_mapping, counts, man_sha,
                          val_fraction=validation_fraction)
    return manifest_path


def _row(path: Path, cname: str, cid: int, split: str, with_image_hash: bool) -> Dict[str, object]:
    r: Dict[str, object] = {
        "path": str(path.resolve()),
        "class_name": cname,
        "class_id": int(cid),
        "split": split,
    }
    if with_image_hash:
        try:
            r["image_hash"] = _sha256_file(path)
        except OSError:
            r["image_hash"] = ""
    return r


def load_split(manifest_csv: Path | str, split: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    return df[df["split"] == split].reset_index(drop=True)


def manifest_hash(manifest_csv: Path | str) -> str:
    h = hashlib.sha256()
    with Path(manifest_csv).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_split_disjoint(manifest_csv: Path | str) -> Dict[str, int]:
    """Return a report dict with overlap counts between splits. Zero overlaps
    means the official split is sound."""
    df = pd.read_csv(manifest_csv)
    sets = {s: set(df[df["split"] == s]["path"]) for s in ("train", "val", "test") if s in set(df["split"])}
    out: Dict[str, int] = {}
    keys = list(sets.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[f"{a}_vs_{b}"] = len(sets[a] & sets[b])
    return out


def duplicate_report(manifest_csv: Path | str, out_csv: Optional[Path | str] = None
                     ) -> pd.DataFrame:
    """Flag identical images via sha256. Requires image_hash column; if absent,
    compute it on the fly from paths."""
    df = pd.read_csv(manifest_csv)
    if "image_hash" not in df.columns:
        df = df.copy()
        df["image_hash"] = [_sha256_file(Path(p)) for p in df["path"]]
    dup = df[df.duplicated(subset=["image_hash"], keep=False)].sort_values("image_hash")
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        dup.to_csv(out_csv, index=False)
    return dup

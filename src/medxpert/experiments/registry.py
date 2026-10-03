from __future__ import annotations
import csv, json, subprocess, sys, platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


REGISTRY_COLUMNS = [
    "experiment_id", "model", "seed", "dataset_version",
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "auc_ovr_macro", "ece", "brier",
    "checkpoint", "git_commit", "timestamp",
]


def git_commit(repo_dir: Optional[Path] = None) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=str(repo_dir or Path.cwd()),
                                      stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return "not-a-git-repo"


class ExperimentRegistry:
    def __init__(self, csv_path: Path | str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(REGISTRY_COLUMNS)

    def append(self, row: Dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in REGISTRY_COLUMNS}
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([clean[k] for k in REGISTRY_COLUMNS])


def write_run_bundle(out_dir: Path | str, cfg: Any,
                     manifest_hash_str: Optional[str] = None,
                     hardware: Optional[Dict[str, Any]] = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)
    (out_dir / "git_commit.txt").write_text(git_commit())
    (out_dir / "seed.txt").write_text(str(cfg.get("project", {}).get("seed", "")))
    if manifest_hash_str:
        (out_dir / "manifest_hash.txt").write_text(manifest_hash_str)
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    if hardware:
        (out_dir / "hardware.json").write_text(json.dumps(hardware, indent=2))

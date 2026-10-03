"""Thin wrapper around train_medxpert for the 5 baseline configs."""
from __future__ import annotations
from train_medxpert import main as train_main

if __name__ == "__main__":
    raise SystemExit(train_main())

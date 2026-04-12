#!/usr/bin/env python
"""CLI: generate subject-level train/val/test splits and write dream_splits.json.

Usage:
    python scripts/make_splits.py --config configs/make_splits.yaml
"""
import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import yaml

from src.data.make_splits import discover_subject_ids, make_splits, save_splits
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate subject-level dataset splits")
    parser.add_argument(
        "--config",
        default="configs/make_splits.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    epochs_dir = Path(cfg["epochs_dir"])
    output_path = Path(cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subject_ids = discover_subject_ids(epochs_dir)

    if not subject_ids:
        logger.warning(
            "No preprocessed subjects found in %s — writing placeholder splits.", epochs_dir
        )
        placeholder = {
            "train": [],
            "val": [],
            "test": [],
            "note": (
                "No preprocessed subjects found — rerun after Plan 01-02 "
                "on a machine with raw data"
            ),
        }
        with output_path.open("w") as f:
            json.dump(placeholder, f, indent=2)
        return 0

    splits = make_splits(
        subject_ids,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        seed=int(cfg["seed"]),
    )
    save_splits(splits, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

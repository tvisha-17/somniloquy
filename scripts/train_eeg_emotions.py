"""CLI entry point for EEG emotion training."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_eeg_emotions import train_eeg_emotions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EEG emotion classifier.")
    parser.add_argument("--config", type=pathlib.Path, default=PROJECT_ROOT / "configs" / "eeg_emotions.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    train_eeg_emotions(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

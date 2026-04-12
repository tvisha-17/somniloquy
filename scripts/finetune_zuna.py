"""CLI entry point for Phase 2 fine-tuning."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.finetune_zuna import train_model
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the ZUNA speech decoder head.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PROJECT_ROOT / "configs" / "finetune_zuna.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    result = train_model(config)
    logger.info("Training finished: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

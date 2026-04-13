"""CLI pipeline for EEG emotion classification."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eeg_emotions_dataset import inspect_and_cache_eeg_emotions
from src.evaluation.evaluate_eeg_emotions import evaluate_eeg_emotions
from src.training.train_eeg_emotions import train_eeg_emotions
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEG emotion classification baseline.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PROJECT_ROOT / "configs" / "eeg_emotions.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    processed_dir = pathlib.Path(config["processed_dir"])
    if bool(config.get("force_rebuild_cache", False)) or not (processed_dir / "index.json").exists():
        inspect_and_cache_eeg_emotions(config)

    train_result = train_eeg_emotions(config)
    eval_config = dict(config)
    eval_config["checkpoint_path"] = train_result["best_checkpoint_path"]
    metrics = evaluate_eeg_emotions(eval_config)

    logger.info(
        "EEG emotion classification complete: split=%s balanced_accuracy=%.4f macro_f1=%.4f p_bal_acc=%.4f p_macro_f1=%.4f",
        metrics["split_mode"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        metrics["permutation_balanced_accuracy"]["empirical_p_value"],
        metrics["permutation_macro_f1"]["empirical_p_value"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

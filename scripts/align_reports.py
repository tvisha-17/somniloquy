#!/usr/bin/env python
"""CLI: align dream reports to REM EEG epochs for all subjects.

Usage:
    python scripts/align_reports.py --config configs/align_reports.yaml
"""
import argparse
import sys
from pathlib import Path

# Allow running as: python scripts/align_reports.py from project root
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import yaml

from src.data.align_reports import align_subject, save_target_embeddings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Align dream reports to REM epochs")
    parser.add_argument(
        "--config",
        default="configs/align_reports.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    epochs_dir = Path(cfg["epochs_dir"])
    reports_root = Path(cfg["reports_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all preprocessed epoch files
    epoch_files = sorted(epochs_dir.glob("sub-*_epochs.npz"))
    if not epoch_files:
        logger.warning(
            "No epoch files found in %s — run Plan 01-02 preprocessing first.", epochs_dir
        )
        return 0

    # Load model once
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence transformer: %s", cfg["embedding_model"])
    model = SentenceTransformer(cfg["embedding_model"])

    for epoch_path in epoch_files:
        # Extract subject ID from filename
        name = epoch_path.stem  # sub-<id>_epochs
        subject_id = name.split("_")[0].replace("sub-", "")
        report_path = reports_root / f"sub-{subject_id}_dream.txt"

        result = align_subject(
            subject_id=subject_id,
            epochs_path=epoch_path,
            report_path=report_path,
            model=model,
            cfg=cfg,
        )
        if result is not None:
            out_path = output_dir / f"sub-{subject_id}_target_embeddings.npz"
            save_target_embeddings(result, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

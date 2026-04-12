#!/usr/bin/env python
"""CLI entry point for dream-report → EEG epoch alignment.

Uses src/data/align_dream_reports.py which reads Reports.csv (shipped with
the Zhang & Wamsley 2019 dataset) to match dream reports to EDF segments by
filename, rather than searching for per-subject .txt files.

Usage:
    python scripts/align_reports.py
    python scripts/align_reports.py --config configs/align_reports.yaml
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import yaml

from src.data.align_dream_reports import (
    load_reports_csv,
    align_subject,
    save_target_embeddings,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Align dream reports to EEG epochs (CSV-based edition)."
    )
    parser.add_argument(
        "--config",
        default="configs/align_reports.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = _project_root / args.config
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    epochs_dir = _project_root / cfg["epochs_dir"]
    reports_csv = _project_root / cfg["reports_csv"]
    output_dir = _project_root / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the report index once (maps EDF basename → report text)
    report_index = load_reports_csv(reports_csv)
    if not report_index:
        logger.error("Reports CSV is empty or unreadable: %s", reports_csv)
        return 1

    # Find all preprocessed epoch files
    epoch_files = sorted(epochs_dir.glob("sub-*_epochs.npz"))
    if not epoch_files:
        logger.warning(
            "No epoch files found in %s — run scripts/preprocess_dream.py first.",
            epochs_dir,
        )
        return 0

    # Load sentence transformer once
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence transformer: %s", cfg["embedding_model"])
    model = SentenceTransformer(cfg["embedding_model"])

    n_aligned = 0
    n_skipped = 0

    for epoch_path in epoch_files:
        subject_id = epoch_path.stem.split("_")[0].replace("sub-", "")
        logger.info("Aligning subject %s", subject_id)

        result = align_subject(
            subject_id=subject_id,
            epochs_npz_path=epoch_path,
            report_index=report_index,
            model=model,
            cfg=cfg,
        )
        if result is None:
            n_skipped += 1
            continue

        out_path = output_dir / f"sub-{subject_id}_target_embeddings.npz"
        save_target_embeddings(result, out_path)
        n_aligned += 1

    logger.info("Done.  aligned=%d  skipped=%d", n_aligned, n_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())

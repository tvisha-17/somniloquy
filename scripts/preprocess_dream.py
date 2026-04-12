#!/usr/bin/env python
"""CLI entry point for DREAM EEG preprocessing.

Uses src/data/preprocess_dream_eeg.py which infers sleep-stage labels from
EDF filenames (subject<NNN>_REM.edf, _NREM.edf, _Morning.edf, _SO1.edf …)
rather than from raw.annotations.

Usage:
    python scripts/preprocess_dream.py
    python scripts/preprocess_dream.py --config configs/preprocess_dream.yaml
"""

import argparse
import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.data.preprocess_dream_eeg import (
    discover_subject_edfs,
    preprocess_subject_all_segments,
    save_subject_npz,
    save_subject_summary,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DREAM EEG dataset (filename-aware edition)."
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess_dream.yaml",
        help="Path to preprocessing config YAML.",
    )
    args = parser.parse_args()

    config_path = _PROJECT_ROOT / args.config
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    raw_root = _PROJECT_ROOT / cfg["raw_root"]
    if not raw_root.exists():
        logger.error("raw_root not found: %s — check configs/preprocess_dream.yaml", raw_root)
        sys.exit(1)

    output_dir = _PROJECT_ROOT / cfg["output_dir"]
    summary_dir = _PROJECT_ROOT / cfg.get("summary_dir", str(output_dir / "summaries"))

    # Group all EDF files by subject ID
    subject_map = discover_subject_edfs(raw_root)
    if not subject_map:
        logger.error("No recognisable EDF files found under %s", raw_root)
        sys.exit(1)

    # Optionally filter to a subset of subjects
    requested = cfg.get("subjects", "all")
    if requested != "all" and isinstance(requested, list):
        requested_set = {str(s).zfill(3) for s in requested}
        subject_map = {k: v for k, v in subject_map.items() if k in requested_set}
        logger.info("Filtered to %d requested subjects", len(subject_map))

    n_processed = 0
    n_skipped = 0
    total_epochs = 0

    for subject_id, edf_paths in sorted(subject_map.items()):
        logger.info(
            "Processing subject %s  (%d EDF files)", subject_id, len(edf_paths)
        )
        result = preprocess_subject_all_segments(subject_id, edf_paths, cfg)
        if result is None:
            n_skipped += 1
            continue

        save_subject_npz(result, output_dir)
        save_subject_summary(result, summary_dir)
        total_epochs += int(result["data"].shape[0])
        n_processed += 1

    logger.info(
        "Done.  processed=%d  skipped=%d  total_epochs=%d",
        n_processed, n_skipped, total_epochs,
    )


if __name__ == "__main__":
    main()

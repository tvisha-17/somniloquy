#!/usr/bin/env python
"""CLI entry point for DREAM EEG preprocessing (Agent 1A).

Usage:
    python scripts/preprocess_dream.py --config configs/preprocess_dream.yaml
"""

import argparse
import json
import sys
import pathlib

# Ensure project root is importable regardless of invocation directory
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.data.preprocess_dream import preprocess_subject, save_subject_npz
from src.utils.logging import get_logger

logger = get_logger(__name__)

_EEG_EXTENSIONS = {".edf", ".set", ".fif", ".vhdr"}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DREAM EEG dataset (Agent 1A)."
    )
    parser.add_argument(
        "--config",
        default="configs/preprocess_dream.yaml",
        help="Path to preprocessing config YAML.",
    )
    args = parser.parse_args()

    # Load config
    config_path = pathlib.Path(args.config)
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    # Check inspection report
    report_path = _PROJECT_ROOT / "data" / "processed" / "dream" / "inspection_report.json"
    if report_path.exists():
        with report_path.open() as fh:
            report = json.load(fh)
        logger.info("Inspection report loaded: %d known issues", len(report.get("known_issues", [])))
    else:
        logger.warning(
            "inspection_report.json not found at %s — run Plan 01-01 (scripts/inspect_dream.py) first.",
            report_path,
        )

    # Resolve raw root
    raw_root = _PROJECT_ROOT / cfg["raw_root"]
    if not raw_root.exists():
        logger.warning(
            "raw_root not found: skipping preprocessing (Plan 01-01 must run first on a machine with raw data)"
        )
        sys.exit(0)

    # Discover subject EEG files
    subject_files = [
        p for p in raw_root.rglob("*") if p.suffix.lower() in _EEG_EXTENSIONS
    ]
    logger.info("Discovered %d EEG files under %s", len(subject_files), raw_root)

    # Prepare output directory
    output_dir = _PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_skipped = 0
    total_epochs = 0

    for path in sorted(subject_files):
        logger.info("Processing %s", path.name)
        result = preprocess_subject(path, cfg)
        if result is None:
            n_skipped += 1
            continue
        save_subject_npz(result, output_dir)
        total_epochs += result["data"].shape[0]
        n_processed += 1

    logger.info(
        "Done. processed=%d skipped=%d total_epochs=%d",
        n_processed,
        n_skipped,
        total_epochs,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()

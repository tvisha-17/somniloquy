"""CLI entry point: run DREAM dataset inspection and produce DATASET_CARD.md.

Usage:
    python scripts/inspect_dream.py [--config configs/inspect_dream.yaml]

The script runs even if data/raw/dream/ does not exist — in that case the
output files will reflect empty inspection results and note the missing raw
data under Known Issues.

Exit codes:
    0 — success (output files written)
    1 — unexpected error
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when this script is run directly
# (i.e., `python scripts/inspect_dream.py` from the project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml

from src.data.inspect_dream import run_inspection, write_dataset_card, write_inspection_report
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect the DREAM dataset and produce a dataset card + inspection report."
    )
    parser.add_argument(
        "--config",
        default="configs/inspect_dream.yaml",
        help="Path to the inspection YAML config (default: configs/inspect_dream.yaml)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    logger.info("Loading config from %s", config_path)
    with config_path.open() as fh:
        config = yaml.safe_load(fh)

    logger.info("Starting inspection with config: %s", config)
    inspection = run_inspection(config)

    # Ensure output directories exist
    card_path = Path(config["output_card"])
    report_path = Path(config["output_report"])
    card_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    write_dataset_card(inspection, card_path)
    write_inspection_report(inspection, report_path)

    logger.info("Dataset card written to: %s", card_path)
    logger.info("Inspection report written to: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

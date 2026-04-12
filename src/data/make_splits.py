"""Subject-level train/val/test split generator for the DREAM dataset."""

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def discover_subject_ids(epochs_dir: Path) -> List[str]:
    """Return sorted list of subject IDs from sub-<id>_epochs.npz filenames.

    Args:
        epochs_dir: Directory containing epoch .npz files.

    Returns:
        Sorted list of subject ID strings.
    """
    epochs_dir = Path(epochs_dir)
    pattern = re.compile(r"^sub-([A-Za-z0-9]+)_epochs\.npz$")
    ids = []
    if epochs_dir.is_dir():
        for p in epochs_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                ids.append(m.group(1))
    return sorted(ids)


def make_splits(
    subject_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    """Partition subjects into disjoint train/val/test sets.

    Args:
        subject_ids: List of subject ID strings.
        train_ratio: Fraction for training (0 < x < 1).
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to sorted ID lists.

    Raises:
        ValueError: If ratios do not sum to 1.0 (±1e-6).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total:.6f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    rng = np.random.default_rng(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val :])

    assert set(train).isdisjoint(val), "train/val overlap"
    assert set(train).isdisjoint(test), "train/test overlap"
    assert set(val).isdisjoint(test), "val/test overlap"

    logger.info(
        "Splits: train=%d val=%d test=%d (seed=%d)",
        len(train), len(val), len(test), seed,
    )
    return {"train": train, "val": val, "test": test}


def save_splits(splits: Dict[str, List[str]], output_path: Path) -> None:
    """Write splits to a JSON file with indent=2.

    Args:
        splits: Dict with 'train', 'val', 'test' keys.
        output_path: Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(splits, f, indent=2)
    logger.info("Saved splits to %s", output_path)

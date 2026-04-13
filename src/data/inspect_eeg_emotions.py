"""Inspect the EEG emotion dataset and write a dataset card."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

import numpy as np
import scipy.io

from src.utils.logging import get_logger

logger = get_logger(__name__)

FILENAME_RE = re.compile(r"^G_(S\d+)_M(\d+)_E(\d+)_R(\d+)_([^_]+)_raw_ref\.mat$")


def parse_filename_metadata(path: Path) -> dict[str, Any]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized EEG emotion filename: {path.name}")
    subject_id, night, emotion_code, report_index, stage = match.groups()
    return {
        "subject_id": subject_id,
        "night": int(night),
        "emotion_code": int(emotion_code),
        "report_index": int(report_index),
        "stage": stage,
    }


def _sample_mat_properties(path: Path) -> dict[str, Any]:
    payload = scipy.io.loadmat(path)
    if "Data" not in payload:
        raise KeyError(f"{path.name} is missing Data")
    data = np.asarray(payload["Data"], dtype=np.float32)
    return {
        "shape": tuple(data.shape),
        "dtype": str(data.dtype),
        "mean": float(data.mean()),
        "std": float(data.std()),
    }


def inspect_eeg_emotions_dataset(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"EEG emotion dataset directory not found: {data_dir}")

    files = sorted(path for path in data_dir.rglob("*") if path.is_file())
    ext_counts = Counter(path.suffix.lower() for path in files)
    data_files = sorted(path for path in files if path.suffix.lower() == ".mat")

    subject_counts: Counter[str] = Counter()
    emotion_counts: Counter[int] = Counter()
    stage_counts: Counter[str] = Counter()
    sample_properties = []
    lengths = []
    channel_counts = []

    for path in data_files:
        meta = parse_filename_metadata(path)
        subject_counts[meta["subject_id"]] += 1
        emotion_counts[meta["emotion_code"]] += 1
        stage_counts[meta["stage"]] += 1

    for path in data_files[: min(10, len(data_files))]:
        props = _sample_mat_properties(path)
        sample_properties.append({"file": path.name, **props})

    for path in data_files:
        data = scipy.io.loadmat(path)["Data"]
        channel_counts.append(int(data.shape[0]))
        lengths.append(int(data.shape[1]))

    inferred_sfreq = None
    if lengths:
        rounded = [length for length in lengths if length % 200 == 0]
        if len(rounded) >= len(lengths) * 0.8:
            inferred_sfreq = 200

    summary = {
        "data_dir": str(data_dir),
        "n_files": len(files),
        "file_types": dict(ext_counts),
        "n_mat_files": len(data_files),
        "n_subjects": len(subject_counts),
        "subject_counts": dict(subject_counts),
        "emotion_counts": dict(sorted(emotion_counts.items())),
        "stage_counts": dict(stage_counts),
        "channel_counts": sorted(set(channel_counts)),
        "min_length": min(lengths) if lengths else None,
        "median_length": int(np.median(lengths)) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "inferred_sfreq": inferred_sfreq,
        "sample_properties": sample_properties,
    }

    card_lines = [
        "# EEG Emotions Dataset Card",
        "",
        f"- Data directory: `{data_dir}`",
        f"- File types: `{dict(ext_counts)}`",
        f"- Number of `.mat` clips: `{len(data_files)}`",
        f"- Subjects detected: `{len(subject_counts)}`",
        f"- Emotion code counts: `{dict(sorted(emotion_counts.items()))}`",
        f"- Sleep stage tag counts: `{dict(stage_counts)}`",
        f"- Channel counts observed: `{sorted(set(channel_counts))}`",
        f"- Clip length range (samples): `{min(lengths) if lengths else 'n/a'}` to `{max(lengths) if lengths else 'n/a'}`",
        f"- Median clip length (samples): `{int(np.median(lengths)) if lengths else 'n/a'}`",
        f"- Inferred sampling rate: `{inferred_sfreq if inferred_sfreq is not None else 'unknown'}`",
        "",
        "## Sample Files",
        "",
    ]
    for sample in sample_properties:
        card_lines.append(
            f"- `{sample['file']}`: shape `{sample['shape']}`, dtype `{sample['dtype']}`, "
            f"mean `{sample['mean']:.4f}`, std `{sample['std']:.4f}`"
        )

    (output_dir / "DATASET_CARD.md").write_text("\n".join(card_lines) + "\n")
    logger.info(
        "inspect_eeg_emotions_dataset n_files=%d n_subjects=%d emotion_counts=%s",
        len(data_files),
        len(subject_counts),
        dict(sorted(emotion_counts.items())),
    )
    return summary


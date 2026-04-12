"""DREAM EEG Preprocessing Pipeline — Agent 1A.

Implements filter, epoch, reject, normalize stages using MNE-Python.
Outputs per-subject .npz files under data/processed/dream/eeg/.
"""

import re
from pathlib import Path
from typing import Tuple

import mne
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sleep-stage mapping
# ---------------------------------------------------------------------------

def map_sleep_stages(annotations) -> np.ndarray:
    """Map annotation label strings to integer sleep-stage codes.

    Mapping (case-insensitive):
        W / wake  -> 0
        N1 / 1   -> 1
        N2 / 2   -> 2
        N3 / 3   -> 3
        R / REM / 4 -> 4
        anything else -> -1

    Args:
        annotations: Iterable of annotation label strings.

    Returns:
        np.ndarray of int with same length as input.
    """
    codes = []
    for label in annotations:
        lbl = label.strip().upper()
        if lbl in ("W", "WAKE", "0"):
            codes.append(0)
        elif lbl in ("N1", "1", "NREM1"):
            codes.append(1)
        elif lbl in ("N2", "2", "NREM2"):
            codes.append(2)
        elif lbl in ("N3", "3", "NREM3", "SWS"):
            codes.append(3)
        elif lbl in ("R", "REM", "4", "NREM4"):
            codes.append(4)
        else:
            codes.append(-1)
    return np.array(codes, dtype=int)


# ---------------------------------------------------------------------------
# Filter / resample
# ---------------------------------------------------------------------------

def bandpass_notch_resample(
    raw: mne.io.BaseRaw,
    l_freq: float,
    h_freq: float,
    notch_freqs,
    target_sfreq: float,
) -> mne.io.BaseRaw:
    """Apply bandpass filter -> notch filter -> resample in that order.

    Args:
        raw: MNE raw object (modified in-place and returned).
        l_freq: High-pass frequency in Hz.
        h_freq: Low-pass frequency in Hz.
        notch_freqs: Frequency or list of frequencies to notch out.
        target_sfreq: Target sampling frequency in Hz.

    Returns:
        The modified raw object.
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=notch_freqs, verbose=False)
    raw.resample(target_sfreq, verbose=False)
    return raw


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------

def _stage_at_time(raw_annotations: mne.Annotations, t: float) -> int:
    """Return the sleep-stage code active at time t (seconds), or -1 if none."""
    for onset, duration, description in zip(
        raw_annotations.onset, raw_annotations.duration, raw_annotations.description
    ):
        if onset <= t < onset + duration:
            codes = map_sleep_stages([description])
            return int(codes[0])
    return -1


def epoch_data(
    raw: mne.io.BaseRaw,
    sleep_stages: np.ndarray,
    epoch_duration: float,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Epoch raw EEG into fixed-length windows.

    Args:
        raw: MNE raw object.
        sleep_stages: Ignored (kept for API consistency); stages are derived
            from raw.annotations at each epoch's start time.
        epoch_duration: Length of each epoch in seconds.
        overlap: Fractional overlap (0 = no overlap, 0.5 = 50%).

    Returns:
        data: np.ndarray shape (n_epochs, n_channels, n_timepoints)
        stages: np.ndarray of int sleep-stage codes per epoch
        start_times: np.ndarray of float epoch start times in seconds
    """
    overlap_sec = epoch_duration * overlap
    epochs_mne = mne.make_fixed_length_epochs(
        raw, duration=epoch_duration, overlap=overlap_sec, preload=True, verbose=False
    )
    data = epochs_mne.get_data()  # (n_epochs, n_channels, n_timepoints)

    # Derive start times
    n_epochs = data.shape[0]
    step = epoch_duration - overlap_sec
    start_times = np.array([i * step for i in range(n_epochs)], dtype=np.float64)

    # Map sleep stages from annotations at each epoch start
    stages = np.array(
        [_stage_at_time(raw.annotations, t) for t in start_times], dtype=int
    )

    return data, stages, start_times


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------

def reject_by_peak_to_peak(
    data: np.ndarray,
    stages: np.ndarray,
    start_times: np.ndarray,
    threshold_v: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop epochs where peak-to-peak amplitude on any channel exceeds threshold.

    Args:
        data: shape (n_epochs, n_channels, n_timepoints)
        stages: shape (n_epochs,)
        start_times: shape (n_epochs,)
        threshold_v: Peak-to-peak threshold in Volts.

    Returns:
        Filtered (data, stages, start_times) with rejected epochs removed.
    """
    ptp = data.max(axis=-1) - data.min(axis=-1)  # (n_epochs, n_channels)
    keep = ptp.max(axis=-1) <= threshold_v        # (n_epochs,)
    return data[keep], stages[keep], start_times[keep]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def zscore_per_channel(data: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel across all epochs and time points.

    Args:
        data: np.ndarray shape (n_epochs, n_channels, n_timepoints), float32.

    Returns:
        Normalized array of same shape and dtype float32.
    """
    mean = data.mean(axis=(0, 2), keepdims=True)
    std = data.std(axis=(0, 2), keepdims=True) + 1e-8
    return ((data - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Raw loading helper (monkeypatchable in tests)
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> mne.io.BaseRaw:
    """Load a raw EEG file using MNE auto-dispatch on file extension.

    Supports: .edf, .set, .fif, .vhdr
    """
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    elif suffix == ".set":
        return mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
    elif suffix == ".fif":
        return mne.io.read_raw_fif(str(path), preload=True, verbose=False)
    elif suffix == ".vhdr":
        return mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG format: {suffix}")


# ---------------------------------------------------------------------------
# Subject preprocessor
# ---------------------------------------------------------------------------

def preprocess_subject(raw_path: Path, cfg: dict) -> "dict | None":
    """Run the full preprocessing pipeline on one subject.

    Args:
        raw_path: Path to the raw EEG file.
        cfg: Config dict with keys: l_freq, h_freq, notch_freqs,
             epoch_duration, overlap, reject_threshold, target_sfreq, min_epochs.

    Returns:
        Dict with keys data, sleep_stages, subject_id, sfreq, ch_names,
        epoch_times_s; or None if fewer than min_epochs survive.
    """
    # Extract subject ID from filename
    match = re.search(r"sub-([A-Za-z0-9]+)", raw_path.name)
    subject_id = match.group(1) if match else raw_path.stem

    # Load
    raw = _load_raw(raw_path)
    raw.pick_types(eeg=True, verbose=False)
    logger.info("loaded shape=%s subject=%s", raw.get_data().shape, subject_id)

    # Filter + resample
    raw = bandpass_notch_resample(
        raw,
        l_freq=cfg["l_freq"],
        h_freq=cfg["h_freq"],
        notch_freqs=cfg["notch_freqs"],
        target_sfreq=cfg["target_sfreq"],
    )
    logger.info("post-filter shape=%s", raw.get_data().shape)

    # Epoch
    data, stages, start_times = epoch_data(
        raw,
        sleep_stages=np.array([], dtype=int),  # derived from annotations internally
        epoch_duration=cfg["epoch_duration"],
        overlap=cfg["overlap"],
    )
    logger.info("post-epoch shape=%s", data.shape)

    # Reject
    data, stages, start_times = reject_by_peak_to_peak(
        data, stages, start_times, threshold_v=cfg["reject_threshold"]
    )
    logger.info("post-reject n_epochs=%d", data.shape[0])

    # Skip if too few epochs
    if data.shape[0] < cfg["min_epochs"]:
        logger.warning(
            "Subject %s has only %d epochs (< min_epochs=%d) — skipping.",
            subject_id,
            data.shape[0],
            cfg["min_epochs"],
        )
        return None

    # Normalize
    data = zscore_per_channel(data)
    logger.info(
        "post-normalize mean=%.4f std=%.4f", float(data.mean()), float(data.std())
    )

    # Output validation
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    assert not np.isnan(data).any(), "NaN detected after normalization"
    assert abs(data.mean()) < 0.1, f"Mean not near zero: {data.mean()}"
    assert abs(data.std() - 1.0) < 0.2, f"Std not near 1: {data.std()}"

    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    return {
        "data": data,
        "sleep_stages": stages,
        "subject_id": subject_id,
        "sfreq": sfreq,
        "ch_names": ch_names,
        "epoch_times_s": start_times,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_subject_npz(subject_dict: dict, output_dir: Path) -> Path:
    """Save subject data to a .npz file.

    Args:
        subject_dict: Dict with keys data, sleep_stages, subject_id, sfreq,
                      ch_names, epoch_times_s.
        output_dir: Directory where the file will be written.

    Returns:
        Path to the written .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sub-{subject_dict['subject_id']}_epochs.npz"
    np.savez(
        str(path),
        data=subject_dict["data"],
        sleep_stages=subject_dict["sleep_stages"],
        subject_id=subject_dict["subject_id"],
        sfreq=subject_dict["sfreq"],
        ch_names=np.array(subject_dict["ch_names"]),
        epoch_times_s=subject_dict["epoch_times_s"],
    )
    return path

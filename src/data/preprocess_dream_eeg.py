"""DREAM EEG Preprocessing Pipeline — filename-aware edition.

Designed for the Zhang & Wamsley 2019 dataset where recordings are already
split into per-subject, per-condition EDF files:

    subject<NNN>_REM.edf
    subject<NNN>_NREM.edf
    subject<NNN>_Morning.edf
    subject<NNN>_SO1.edf  …  subject<NNN>_SO10.edf

Pipeline (per subject)
----------------------
1. Discover all EDF files that belong to the subject.
2. For each EDF segment:
   a. Load with MNE; pick EEG channels (fall back to all if none typed EEG).
   b. Apply bandpass filter (l_freq–h_freq).
   c. Apply notch filter (50 or 60 Hz).
   d. Resample to target_sfreq.
   e. Create fixed-length epochs with configurable overlap.
   f. Assign a single sleep-stage label to every epoch (from filename).
   g. Record per-epoch metadata: segment_type, source_file, segment_index,
      epoch start time (relative to EDF start), annotation descriptions.
3. Concatenate all segment epochs for the subject.
4. Reject epochs where any channel exceeds the peak-to-peak threshold.
5. Skip subjects with fewer than min_epochs retained epochs.
6. Z-score normalise each channel across ALL retained epochs.
7. Validate dtype, shape, NaN-freedom, and normalisation quality.
8. Save a .npz and a JSON summary.

Output .npz keys
-----------------
data              float32  (n_epochs, n_channels, n_timepoints)
sleep_stages      int32    (n_epochs,)
segment_types     str      (n_epochs,)    e.g. "REM", "NREM", "SO"
source_files      str      (n_epochs,)    EDF basename
segment_indices   int32    (n_epochs,)    SO index or -1
epoch_times_s     float64  (n_epochs,)    start time within source EDF
subject_id        str      scalar
sfreq             float64  scalar
ch_names          str      (n_channels,)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Filename parser  (mirrors inspect_dream_dataset.parse_dream_filename)
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"^subject(\d+)_(REM|NREM|Morning|SO(\d+))(?:\.edf)?$",
    re.IGNORECASE,
)


def parse_dream_filename(path: "Path | str") -> Optional[Dict]:
    """Parse a DREAM EDF filename into its semantic components.

    Args:
        path: File path or bare filename (with or without ``.edf``).

    Returns:
        Dict with ``subject_id`` (str), ``segment_type`` (str: REM / NREM /
        Morning / SO), ``segment_index`` (int: SO number or -1).
        Returns ``None`` if the name does not match the expected convention.
    """
    stem = Path(path).stem
    m = _FNAME_RE.match(stem)
    if m is None:
        return None

    subject_id = m.group(1)
    raw_seg = m.group(2)

    if raw_seg.upper().startswith("SO"):
        segment_type = "SO"
        segment_index = int(m.group(3))
    elif raw_seg.upper() == "REM":
        segment_type = "REM"
        segment_index = -1
    elif raw_seg.upper() == "NREM":
        segment_type = "NREM"
        segment_index = -1
    else:
        segment_type = raw_seg.capitalize()   # Morning
        segment_index = -1

    return {
        "subject_id": subject_id,
        "segment_type": segment_type,
        "segment_index": segment_index,
    }


# ---------------------------------------------------------------------------
# EDF discovery
# ---------------------------------------------------------------------------

def discover_subject_edfs(raw_root: Path) -> Dict[str, List[Path]]:
    """Scan *raw_root* and group EDF paths by subject ID.

    Args:
        raw_root: Directory containing EDF files (searched recursively).

    Returns:
        Dict mapping subject_id (str) → sorted list of matching Path objects.
        EDFs with unrecognised names are logged and skipped.
    """
    raw_root = Path(raw_root)
    mapping: Dict[str, List[Path]] = {}

    for p in sorted(raw_root.rglob("*.edf")):
        parsed = parse_dream_filename(p)
        if parsed is None:
            logger.warning("discover_subject_edfs: skipping unrecognised file %s", p.name)
            continue
        sid = parsed["subject_id"]
        mapping.setdefault(sid, []).append(p)

    logger.info(
        "discover_subject_edfs: root=%s  n_subjects=%d  total_edfs=%d",
        raw_root, len(mapping), sum(len(v) for v in mapping.values()),
    )
    return mapping


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def label_from_segment_type(
    segment_type: str,
    segment_index: int,
    cfg: dict,
) -> int:
    """Map a segment type to an integer sleep-stage label.

    Label source is ``cfg["segment_labels"]`` (a dict).  Falls back to -1
    for unknown types.

    Args:
        segment_type: One of ``"REM"``, ``"NREM"``, ``"Morning"``, ``"SO"``.
        segment_index: SO index (1–10) or -1; not used for label but kept for
            extensibility.
        cfg: Config dict (from ``configs/preprocess_dream.yaml``).

    Returns:
        Integer label, e.g. 4 for REM.
    """
    labels = cfg.get("segment_labels", {})
    label = labels.get(segment_type, -1)
    if label == -1:
        logger.warning(
            "label_from_segment_type: no label for segment_type=%s — assigning -1",
            segment_type,
        )
    return int(label)


# ---------------------------------------------------------------------------
# EEG channel picking
# ---------------------------------------------------------------------------

def pick_eeg_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Pick EEG-typed channels; fall back to all channels if none are typed EEG.

    Some EDFs mark every channel as 'misc' or use custom type strings.  This
    helper logs a warning and proceeds with all channels rather than raising.

    Args:
        raw: Loaded MNE raw object (modified in-place and returned).

    Returns:
        The raw object with only EEG (or all) channels retained.
    """
    n_before = len(raw.ch_names)
    eeg_picks = mne.pick_types(raw.info, eeg=True)

    if len(eeg_picks) == 0:
        logger.warning(
            "pick_eeg_channels: no EEG-typed channels found in %s — retaining all %d channels",
            raw.filenames[0] if hasattr(raw, "filenames") else "unknown",
            n_before,
        )
        return raw   # return as-is

    raw.pick(eeg_picks)
    logger.info(
        "pick_eeg_channels: %d → %d channels selected", n_before, len(raw.ch_names)
    )
    return raw


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
    """Apply bandpass → notch → resample in that order.

    Args:
        raw: MNE raw object (modified in-place).
        l_freq: High-pass cutoff in Hz.
        h_freq: Low-pass cutoff in Hz.
        notch_freqs: Single frequency or list of frequencies to notch.
        target_sfreq: Target sampling rate in Hz.

    Returns:
        The modified raw object.
    """
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    raw.notch_filter(freqs=notch_freqs, verbose=False)
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    logger.info(
        "bandpass_notch_resample: shape=%s  sfreq=%.1f",
        raw.get_data().shape, raw.info["sfreq"],
    )
    return raw


# ---------------------------------------------------------------------------
# Epoch rejection
# ---------------------------------------------------------------------------

def reject_by_peak_to_peak(
    data: np.ndarray,
    sleep_stages: np.ndarray,
    start_times: np.ndarray,
    segment_types: np.ndarray,
    source_files: np.ndarray,
    segment_indices: np.ndarray,
    threshold_v: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop epochs where peak-to-peak amplitude on any channel exceeds threshold.

    Args:
        data: shape (n_epochs, n_channels, n_timepoints)
        sleep_stages: shape (n_epochs,)
        start_times: shape (n_epochs,)
        segment_types: shape (n_epochs,)  string array
        source_files: shape (n_epochs,)   string array
        segment_indices: shape (n_epochs,)
        threshold_v: Peak-to-peak limit in Volts (e.g. 200e-6).

    Returns:
        Tuple of six arrays with the same structure, with rejected epochs
        removed.
    """
    ptp = data.max(axis=-1) - data.min(axis=-1)   # (n_epochs, n_channels)
    keep = ptp.max(axis=-1) <= threshold_v          # (n_epochs,)
    n_rejected = int((~keep).sum())

    logger.info(
        "reject_by_peak_to_peak: threshold=%.1f µV  kept=%d/%d  rejected=%d",
        threshold_v * 1e6, int(keep.sum()), len(keep), n_rejected,
    )
    return (
        data[keep],
        sleep_stages[keep],
        start_times[keep],
        segment_types[keep],
        source_files[keep],
        segment_indices[keep],
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def zscore_per_channel(data: np.ndarray) -> np.ndarray:
    """Z-score normalise each channel across all epochs and time points.

    Normalisation is computed over the flattened (epoch, time) axes so that
    every channel has approximately zero mean and unit standard deviation
    across the entire subject recording.

    Args:
        data: float array of shape (n_epochs, n_channels, n_timepoints).

    Returns:
        Normalised float32 array of the same shape.
    """
    mean = data.mean(axis=(0, 2), keepdims=True)        # (1, n_ch, 1)
    std = data.std(axis=(0, 2), keepdims=True) + 1e-8   # (1, n_ch, 1)
    normalised = ((data - mean) / std).astype(np.float32)
    logger.info(
        "zscore_per_channel: input shape=%s  out mean=%.4f  out std=%.4f",
        data.shape, float(normalised.mean()), float(normalised.std()),
    )
    return normalised


# ---------------------------------------------------------------------------
# Raw loader  (monkeypatchable in tests)
# ---------------------------------------------------------------------------

def _load_raw(path: Path) -> mne.io.BaseRaw:
    """Load an EDF file with MNE (data preloaded into memory).

    Args:
        path: Path to ``.edf`` file.

    Returns:
        Preloaded MNE raw object.
    """
    return mne.io.read_raw_edf(str(path), preload=True, verbose=False)


# ---------------------------------------------------------------------------
# Single-segment preprocessor
# ---------------------------------------------------------------------------

def preprocess_segment(
    edf_path: Path,
    parsed: Dict,
    cfg: dict,
    reference_ch_names: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Load one EDF segment and return epoched arrays.

    Called once per EDF file before aggregating across segments.

    Args:
        edf_path: Path to the EDF file.
        parsed: Result of ``parse_dream_filename(edf_path)`` (cached by
            caller to avoid re-parsing).
        cfg: Config dict from ``configs/preprocess_dream.yaml``.
        reference_ch_names: If provided, the returned data will only contain
            channels present in this list (intersection), in the same order.
            Used to enforce channel consistency across segments for a subject.

    Returns:
        Dict with keys: ``data``, ``sleep_stages``, ``start_times``,
        ``segment_types``, ``source_files``, ``segment_indices``,
        ``ch_names``, ``sfreq``; or ``None`` when the segment is too short
        to yield at least one epoch.
    """
    segment_type = parsed["segment_type"]
    segment_index = parsed["segment_index"]
    source_file = edf_path.name

    # ---------- load ----------
    try:
        raw = _load_raw(edf_path)
    except Exception as exc:
        logger.warning("preprocess_segment: cannot load %s — %s", edf_path, exc)
        return None

    logger.info(
        "preprocess_segment: loaded %s  shape=%s  sfreq=%.1f",
        source_file, raw.get_data().shape, raw.info["sfreq"],
    )

    # Preserve annotations before any channel dropping
    ann_descs = [str(a["description"]) for a in raw.annotations]
    if ann_descs:
        logger.info(
            "preprocess_segment: %s has %d annotation(s): %s",
            source_file, len(ann_descs), ann_descs[:5],
        )

    # ---------- channel selection ----------
    raw = pick_eeg_channels(raw)

    # Align to reference channel layout if supplied
    if reference_ch_names is not None:
        available = set(raw.ch_names)
        target = [c for c in reference_ch_names if c in available]
        if len(target) == 0:
            logger.warning(
                "preprocess_segment: %s shares no channels with reference — skipping",
                source_file,
            )
            return None
        if len(target) < len(reference_ch_names):
            missing = [c for c in reference_ch_names if c not in available]
            logger.warning(
                "preprocess_segment: %s missing channels %s — using intersection (%d ch)",
                source_file, missing, len(target),
            )
        raw.pick(target)

    # ---------- filter / resample ----------
    raw = bandpass_notch_resample(
        raw,
        l_freq=float(cfg["l_freq"]),
        h_freq=float(cfg["h_freq"]),
        notch_freqs=cfg["notch_freqs"],
        target_sfreq=float(cfg["target_sfreq"]),
    )

    # ---------- epoch ----------
    epoch_duration = float(cfg["epoch_duration"])
    overlap_sec = epoch_duration * float(cfg["overlap"])
    step_sec = epoch_duration - overlap_sec

    # MNE requires duration > 0 and the recording to be long enough for ≥1 epoch
    if raw.times[-1] < epoch_duration:
        logger.warning(
            "preprocess_segment: %s is too short (%.1f s < %.1f s epoch) — skipping",
            source_file, raw.times[-1], epoch_duration,
        )
        return None

    try:
        epochs_mne = mne.make_fixed_length_epochs(
            raw,
            duration=epoch_duration,
            overlap=overlap_sec,
            preload=True,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("preprocess_segment: epoching failed for %s — %s", source_file, exc)
        return None

    data = epochs_mne.get_data()   # (n_epochs, n_channels, n_timepoints)
    n_epochs = data.shape[0]

    if n_epochs == 0:
        logger.warning("preprocess_segment: %s produced 0 epochs — skipping", source_file)
        return None

    # ---------- derive epoch start times ----------
    start_times = np.array(
        [i * step_sec for i in range(n_epochs)], dtype=np.float64
    )

    # ---------- assign label ----------
    label = label_from_segment_type(segment_type, segment_index, cfg)
    sleep_stages = np.full(n_epochs, label, dtype=np.int32)

    # ---------- per-epoch metadata ----------
    segment_types_arr = np.array([segment_type] * n_epochs, dtype=object)
    source_files_arr = np.array([source_file] * n_epochs, dtype=object)
    segment_indices_arr = np.full(n_epochs, segment_index, dtype=np.int32)

    logger.info(
        "preprocess_segment: %s → %d epochs  label=%d  ch=%s  shape=%s",
        source_file, n_epochs, label, raw.ch_names, data.shape,
    )

    return {
        "data": data.astype(np.float32),
        "sleep_stages": sleep_stages,
        "start_times": start_times,
        "segment_types": segment_types_arr,
        "source_files": source_files_arr,
        "segment_indices": segment_indices_arr,
        "ch_names": list(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]),
    }


# ---------------------------------------------------------------------------
# Subject-level orchestrator
# ---------------------------------------------------------------------------

def preprocess_subject_all_segments(
    subject_id: str,
    edf_paths: List[Path],
    cfg: dict,
) -> Optional[Dict]:
    """Run the full preprocessing pipeline for one subject.

    Iterates over all EDF files for the subject, preprocesses each segment,
    concatenates, rejects bad epochs globally, and z-score normalises.

    Args:
        subject_id: Subject identifier string (e.g. ``"010"``).
        edf_paths: List of Path objects pointing to this subject's EDF files.
        cfg: Config dict from ``configs/preprocess_dream.yaml``.

    Returns:
        Dict ready for :func:`save_subject_npz` and
        :func:`save_subject_summary`, or ``None`` if fewer than
        ``cfg["min_epochs"]`` epochs survive preprocessing.

        Keys: ``data``, ``sleep_stages``, ``epoch_times_s``,
        ``segment_types``, ``source_files``, ``segment_indices``,
        ``subject_id``, ``sfreq``, ``ch_names``,
        ``n_rejected``, ``epochs_per_segment_type``,
        ``duration_per_source_file``, ``total_epochs_before_rejection``.
    """
    morning_action = cfg.get("morning_action", "include").lower()
    min_epochs = int(cfg.get("min_epochs", 100))

    all_segments: List[Dict] = []
    skipped_segments: List[str] = []

    # Sort for deterministic ordering: REM, NREM, Morning, SO1…SO10
    def _sort_key(p: Path) -> Tuple:
        parsed = parse_dream_filename(p)
        if parsed is None:
            return ("ZZZ", 999)
        order = {"REM": 0, "NREM": 1, "Morning": 2, "SO": 3}
        return (order.get(parsed["segment_type"], 9), parsed["segment_index"])

    edf_paths_sorted = sorted(edf_paths, key=_sort_key)

    for edf_path in edf_paths_sorted:
        parsed = parse_dream_filename(edf_path)
        if parsed is None:
            logger.warning(
                "preprocess_subject_all_segments: cannot parse %s — skipping", edf_path.name
            )
            skipped_segments.append(edf_path.name)
            continue

        # Apply morning_action filter
        if parsed["segment_type"] == "Morning" and morning_action == "exclude":
            logger.info(
                "preprocess_subject_all_segments: skipping Morning segment (morning_action=exclude)"
            )
            skipped_segments.append(edf_path.name)
            continue

        # Each segment is processed independently with its own channel set.
        # Channel intersection across segments is resolved after collection.
        seg_result = preprocess_segment(edf_path, parsed, cfg, reference_ch_names=None)

        if seg_result is None:
            skipped_segments.append(edf_path.name)
            continue

        all_segments.append(seg_result)

    # Compute the channel intersection across all collected segments and trim
    # data arrays so every segment has identical channel layout.
    reference_ch_names: Optional[List[str]] = None
    if all_segments:
        common_set = set(all_segments[0]["ch_names"])
        for seg in all_segments[1:]:
            common_set &= set(seg["ch_names"])
        # Preserve the order from the first segment
        reference_ch_names = [c for c in all_segments[0]["ch_names"] if c in common_set]
        logger.info(
            "preprocess_subject_all_segments: subject=%s  common channels=%d",
            subject_id, len(reference_ch_names),
        )
        for seg in all_segments:
            if seg["ch_names"] != reference_ch_names:
                keep_idx = [seg["ch_names"].index(c) for c in reference_ch_names]
                seg["data"] = seg["data"][:, keep_idx, :]
                seg["ch_names"] = reference_ch_names

    if not all_segments:
        logger.warning(
            "preprocess_subject_all_segments: subject %s has no usable segments — skipping",
            subject_id,
        )
        return None

    # ---------- concatenate ----------
    data = np.concatenate([s["data"] for s in all_segments], axis=0)
    sleep_stages = np.concatenate([s["sleep_stages"] for s in all_segments])
    start_times = np.concatenate([s["start_times"] for s in all_segments])
    segment_types = np.concatenate([s["segment_types"] for s in all_segments])
    source_files = np.concatenate([s["source_files"] for s in all_segments])
    segment_indices = np.concatenate([s["segment_indices"] for s in all_segments])

    total_before = data.shape[0]
    logger.info(
        "preprocess_subject_all_segments: subject=%s  concat shape=%s",
        subject_id, data.shape,
    )

    # ---------- NaN check before rejection ----------
    if np.isnan(data).any():
        logger.warning(
            "preprocess_subject_all_segments: subject %s has NaNs after epoching — "
            "dropping NaN epochs",
            subject_id,
        )
        nan_mask = ~np.isnan(data).any(axis=(1, 2))
        data = data[nan_mask]
        sleep_stages = sleep_stages[nan_mask]
        start_times = start_times[nan_mask]
        segment_types = segment_types[nan_mask]
        source_files = source_files[nan_mask]
        segment_indices = segment_indices[nan_mask]

    # ---------- global epoch rejection ----------
    (
        data, sleep_stages, start_times,
        segment_types, source_files, segment_indices,
    ) = reject_by_peak_to_peak(
        data, sleep_stages, start_times,
        segment_types, source_files, segment_indices,
        threshold_v=float(cfg["reject_threshold"]),
    )

    n_rejected = total_before - data.shape[0]

    # ---------- min-epochs guard ----------
    if data.shape[0] < min_epochs:
        logger.warning(
            "preprocess_subject_all_segments: subject %s has only %d epochs "
            "(< min_epochs=%d) — skipping",
            subject_id, data.shape[0], min_epochs,
        )
        return None

    # ---------- z-score normalisation ----------
    data = zscore_per_channel(data)

    # ---------- validation ----------
    assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"
    assert data.ndim == 3, f"Expected 3-D array, got ndim={data.ndim}"
    assert not np.isnan(data).any(), "NaN detected after normalisation"
    assert len(sleep_stages) == data.shape[0], "Label / epoch count mismatch"
    assert abs(float(data.mean())) < 0.1, f"Mean not near zero: {data.mean()}"
    assert abs(float(data.std()) - 1.0) < 0.2, f"Std not near 1: {data.std()}"

    # ---------- per-segment-type epoch counts (for summary) ----------
    epochs_per_type: Dict[str, int] = {}
    for st in np.unique(segment_types):
        epochs_per_type[str(st)] = int((segment_types == st).sum())

    # ---------- duration per source file (for summary) ----------
    duration_per_file: Dict[str, float] = {}
    epoch_dur = float(cfg["epoch_duration"])
    step = epoch_dur * (1.0 - float(cfg["overlap"]))
    for sf in np.unique(source_files):
        mask = source_files == sf
        n = int(mask.sum())
        duration_per_file[str(sf)] = round(n * step, 2)

    sfreq = float(all_segments[0]["sfreq"])

    logger.info(
        "preprocess_subject_all_segments: subject=%s  final shape=%s  sfreq=%.1f  "
        "rejected=%d  skipped_segments=%d",
        subject_id, data.shape, sfreq, n_rejected, len(skipped_segments),
    )

    return {
        "data": data,
        "sleep_stages": sleep_stages.astype(np.int32),
        "epoch_times_s": start_times,
        "segment_types": segment_types,
        "source_files": source_files,
        "segment_indices": segment_indices.astype(np.int32),
        "subject_id": subject_id,
        "sfreq": sfreq,
        "ch_names": reference_ch_names or [],
        # Summary metadata (not saved to .npz, used for JSON summary)
        "n_rejected": n_rejected,
        "total_epochs_before_rejection": total_before,
        "epochs_per_segment_type": epochs_per_type,
        "duration_per_source_file": duration_per_file,
        "skipped_segments": skipped_segments,
    }


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_subject_npz(subject_dict: dict, output_dir: Path) -> Path:
    """Save the preprocessed subject data to a ``.npz`` file.

    File is written to ``output_dir/sub-<subject_id>_epochs.npz``.

    .npz contents
    -------------
    data              float32  (n_epochs, n_channels, n_timepoints)
    sleep_stages      int32    (n_epochs,)
    segment_types     object   (n_epochs,)  string labels
    source_files      object   (n_epochs,)  EDF basenames
    segment_indices   int32    (n_epochs,)  SO index or -1
    epoch_times_s     float64  (n_epochs,)
    subject_id        str      scalar
    sfreq             float64  scalar
    ch_names          object   (n_channels,)

    Args:
        subject_dict: Dict returned by :func:`preprocess_subject_all_segments`.
        output_dir: Directory where the file will be written.

    Returns:
        Path to the written ``.npz`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sid = subject_dict["subject_id"]
    out_path = output_dir / f"sub-{sid}_epochs.npz"

    np.savez(
        str(out_path),
        data=subject_dict["data"],
        sleep_stages=subject_dict["sleep_stages"],
        segment_types=np.array(subject_dict["segment_types"], dtype=object),
        source_files=np.array(subject_dict["source_files"], dtype=object),
        segment_indices=subject_dict["segment_indices"],
        epoch_times_s=subject_dict["epoch_times_s"],
        subject_id=np.array(subject_dict["subject_id"]),
        sfreq=np.array(subject_dict["sfreq"]),
        ch_names=np.array(subject_dict["ch_names"], dtype=object),
    )

    logger.info(
        "save_subject_npz: wrote %s  data=%s  dtype=%s",
        out_path, subject_dict["data"].shape, subject_dict["data"].dtype,
    )
    return out_path


def save_subject_summary(subject_dict: dict, summary_dir: Path) -> Path:
    """Write a per-subject processing summary to a JSON file.

    File is written to ``summary_dir/sub-<subject_id>_summary.json``.

    JSON keys
    ---------
    subject_id
    total_epochs
    epochs_per_segment_type
    epochs_rejected
    total_epochs_before_rejection
    duration_per_source_file
    skipped_segments
    ch_names
    sfreq

    Args:
        subject_dict: Dict returned by :func:`preprocess_subject_all_segments`.
        summary_dir: Directory where the JSON file will be written.

    Returns:
        Path to the written JSON file.
    """
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    sid = subject_dict["subject_id"]
    out_path = summary_dir / f"sub-{sid}_summary.json"

    summary = {
        "subject_id": sid,
        "total_epochs": int(subject_dict["data"].shape[0]),
        "epochs_per_segment_type": subject_dict.get("epochs_per_segment_type", {}),
        "epochs_rejected": subject_dict.get("n_rejected", 0),
        "total_epochs_before_rejection": subject_dict.get(
            "total_epochs_before_rejection", 0
        ),
        "duration_per_source_file": subject_dict.get("duration_per_source_file", {}),
        "skipped_segments": subject_dict.get("skipped_segments", []),
        "ch_names": subject_dict.get("ch_names", []),
        "sfreq": subject_dict.get("sfreq", None),
    }

    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("save_subject_summary: wrote %s", out_path)
    return out_path

"""Dream Report Aligner — Agent 1B.

Encodes free-text dream reports into 384-dim sentence embeddings and aligns
them to REM EEG epochs that fall within a time window before awakening.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

# sentence_transformers is used by callers who pass a SentenceTransformer as `model`.
# Imported here so the dependency is explicit; falls back gracefully in offline tests.
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_report(text: str, model) -> np.ndarray:
    """Encode a single dream report text into a 384-dim float32 embedding.

    Args:
        text: Raw dream report string.
        model: A SentenceTransformer (or compatible stub) with .encode().

    Returns:
        np.ndarray of shape (384,) and dtype float32.
    """
    embedding = model.encode(text, convert_to_numpy=True).astype(np.float32)
    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    return embedding


def encode_reports_batch(texts: List[str], model) -> np.ndarray:
    """Encode a batch of dream report texts.

    Args:
        texts: List of report strings.
        model: A SentenceTransformer (or compatible stub).

    Returns:
        np.ndarray of shape (len(texts), 384) and dtype float32.
    """
    result = np.stack([encode_report(t, model) for t in texts]).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Awakening detection
# ---------------------------------------------------------------------------

_AWAKENING_KEYWORDS = ("awaken", "wake_up", "arousal_end")


def find_awakening_times(
    annotations: List[dict], recording_end_s: float
) -> List[float]:
    """Return onset times of annotations that match awakening keywords.

    Each entry in annotations must have keys 'onset' (float) and
    'description' (str). If no annotations match, returns [recording_end_s].

    Args:
        annotations: List of annotation dicts.
        recording_end_s: Duration of recording in seconds (fallback).

    Returns:
        List of float awakening times, sorted ascending.
    """
    times = []
    for ann in annotations:
        desc = ann.get("description", "").lower()
        if any(kw in desc for kw in _AWAKENING_KEYWORDS):
            times.append(float(ann["onset"]))
    if not times:
        return [recording_end_s]
    return sorted(times)


# ---------------------------------------------------------------------------
# REM epoch selection
# ---------------------------------------------------------------------------

def select_rem_epochs_before_awakening(
    sleep_stages: np.ndarray,
    epoch_times_s: np.ndarray,
    awakening_time_s: float,
    window_s: float,
) -> np.ndarray:
    """Return indices of REM epochs within window_s seconds before awakening.

    Args:
        sleep_stages: Integer array of sleep stage codes (4 = REM).
        epoch_times_s: Float array of epoch start times in seconds.
        awakening_time_s: Time of awakening in seconds.
        window_s: Look-back window in seconds.

    Returns:
        int64 array of matching epoch indices.
    """
    mask = (
        (sleep_stages == 4)
        & (epoch_times_s >= awakening_time_s - window_s)
        & (epoch_times_s < awakening_time_s)
    )
    return np.where(mask)[0].astype(np.int64)


# ---------------------------------------------------------------------------
# Subject epoch loading
# ---------------------------------------------------------------------------

def load_subject_epochs(subject_npz_path: Path) -> dict:
    """Load epoch metadata from a preprocessed .npz file.

    Args:
        subject_npz_path: Path to sub-<id>_epochs.npz.

    Returns:
        Dict with keys: sleep_stages, epoch_times_s, subject_id, annotations.
    """
    data = np.load(str(subject_npz_path), allow_pickle=True)
    result = {
        "sleep_stages": data["sleep_stages"],
        "epoch_times_s": data["epoch_times_s"],
        "subject_id": str(data["subject_id"]),
    }
    # annotations may or may not be present
    if "annotations" in data:
        result["annotations"] = data["annotations"].tolist()
    else:
        result["annotations"] = []
    return result


# ---------------------------------------------------------------------------
# Per-subject alignment
# ---------------------------------------------------------------------------

def align_subject(
    subject_id: str,
    epochs_path: Path,
    report_path: Path,
    model,
    cfg: dict,
) -> Optional[dict]:
    """Align dream report embedding to REM epochs for one subject.

    Args:
        subject_id: Subject identifier string.
        epochs_path: Path to sub-<id>_epochs.npz.
        report_path: Path to sub-<id>_dream.txt.
        model: SentenceTransformer (or stub).
        cfg: Config dict with keys time_alignment_window, embedding_dim.

    Returns:
        Dict with epoch_indices, target_embeddings, report_text; or None if
        no REM epochs fall within the alignment window.
    """
    # Load report
    if not report_path.exists():
        logger.warning("Report file not found for subject %s: %s", subject_id, report_path)
        return None

    report_text = report_path.read_text().strip()

    # Load epochs
    epoch_data = load_subject_epochs(epochs_path)
    sleep_stages = epoch_data["sleep_stages"]
    epoch_times_s = epoch_data["epoch_times_s"]
    annotations = epoch_data["annotations"]

    # Determine recording end (last epoch time)
    recording_end_s = float(epoch_times_s[-1]) if len(epoch_times_s) > 0 else 0.0

    # Find awakening times
    awakening_times = find_awakening_times(annotations, recording_end_s)

    # Encode report once
    embedding = encode_report(report_text, model)

    # Collect REM epoch indices across all awakenings
    all_indices = []
    window_s = float(cfg["time_alignment_window"])
    for awake_t in awakening_times:
        idxs = select_rem_epochs_before_awakening(
            sleep_stages, epoch_times_s, awake_t, window_s
        )
        all_indices.append(idxs)

    if all_indices:
        combined = np.unique(np.concatenate(all_indices)).astype(np.int64)
    else:
        combined = np.array([], dtype=np.int64)

    if len(combined) == 0:
        logger.warning(
            "Subject %s: no REM epochs found in alignment window — skipping.", subject_id
        )
        return None

    # Broadcast single embedding to all selected epochs
    target_embeddings = np.broadcast_to(
        embedding, (len(combined), len(embedding))
    ).copy().astype(np.float32)

    logger.info(
        "Subject %s: %d REM epochs selected for alignment.", subject_id, len(combined)
    )

    return {
        "epoch_indices": combined,
        "target_embeddings": target_embeddings,
        "report_text": report_text,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_target_embeddings(result: dict, output_path: Path) -> None:
    """Save alignment result to a .npz file.

    Args:
        result: Dict with epoch_indices, target_embeddings, report_text.
        output_path: Destination .npz path.
    """
    np.savez(
        str(output_path),
        epoch_indices=result["epoch_indices"].astype(np.int64),
        target_embeddings=result["target_embeddings"].astype(np.float32),
        report_text=np.array(result["report_text"]),
    )
    logger.info("Saved target embeddings to %s", output_path)

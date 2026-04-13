"""Dream Report Aligner — DREAM dataset edition.

Aligns free-text dream reports to EEG epoch arrays produced by
``preprocess_dream_eeg.py``.

Dataset-specific design
-----------------------
The Zhang & Wamsley 2019 dataset ships a ``Reports.csv`` file whose
``Filename`` column maps each dream report to its source EDF basename
(e.g. ``subject010_REM.edf``).  Because the EEG data is already split per
condition, there is **no awakening time to detect**: every retained epoch
from a given EDF segment simply receives the same report embedding.

Alignment procedure (per subject)
-----------------------------------
1. Load the subject's preprocessed ``.npz`` (written by
   ``preprocess_dream_eeg.py``).  Read the ``source_files`` array (one
   basename per epoch).
2. For each unique source basename, look up the matching row in the
   ``Reports.csv`` index.  If no match, skip that segment silently.
3. Encode the report text with SentenceTransformer
   (``all-MiniLM-L6-v2`` → 384 dims).
4. Broadcast the embedding to all epochs whose ``source_files`` entry
   matches the basename.
5. Save a ``.npz`` file that contains the epoch indices, the embedding
   matrix, and metadata.

Output ``.npz`` keys
----------------------
epoch_indices           int64    (n_aligned_epochs,)
target_embeddings       float32  (n_aligned_epochs, 384)
segment_types           object   (n_aligned_epochs,)  e.g. "REM"
source_files            object   (n_aligned_epochs,)  EDF basenames
report_texts            object   (n_aligned_epochs,)  full report per epoch

Manual confirmation required
-----------------------------
- Column names in ``Reports.csv`` are assumed to be exactly:
  ``Subject ID``, ``Case ID``, ``Filename``, ``Text of Report``.
  Adjust ``_CSV_COL_FILENAME`` / ``_CSV_COL_TEXT`` if they differ.
- ``Filename`` values must match EDF basenames exactly (including
  capitalisation and the ``.edf`` extension).
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CSV column names  ← confirm against actual Reports.csv header
# ---------------------------------------------------------------------------
_CSV_COL_FILENAME = "Filename"
_CSV_COL_TEXT = "Text of Report"
_DEFAULT_FILLER_WORDS = ("umm", "um", "uh", "uhh", "uhm", "erm", "hmm", "mm")
_DEFAULT_DROP_PATTERNS = ("n/a", "na", "none")

# ---------------------------------------------------------------------------
# Filename parser  (mirrors parse_dream_filename in other modules)
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"^subject(\d+)_(REM|NREM|Morning|SO(\d+))(?:\.edf)?$",
    re.IGNORECASE,
)


def _parse_dream_filename(name: str) -> Optional[Dict]:
    """Return (subject_id, segment_type, segment_index) from an EDF basename."""
    stem = Path(name).stem
    m = _FNAME_RE.match(stem)
    if m is None:
        return None
    subject_id = m.group(1)
    raw_seg = m.group(2)
    if raw_seg.upper().startswith("SO"):
        return {
            "subject_id": subject_id,
            "segment_type": "SO",
            "segment_index": int(m.group(3)),
        }
    if raw_seg.upper() == "REM":
        return {"subject_id": subject_id, "segment_type": "REM", "segment_index": -1}
    if raw_seg.upper() == "NREM":
        return {"subject_id": subject_id, "segment_type": "NREM", "segment_index": -1}
    return {"subject_id": subject_id, "segment_type": raw_seg.capitalize(), "segment_index": -1}


# ---------------------------------------------------------------------------
# CSV report index
# ---------------------------------------------------------------------------

def load_reports_csv(reports_csv_path: "Path | str") -> Dict[str, str]:
    """Load ``Reports.csv`` and return a mapping from EDF basename → report text.

    The ``Filename`` column is used as the key.  Rows with empty report text
    are silently omitted.

    Args:
        reports_csv_path: Path to the CSV file.

    Returns:
        Dict mapping EDF basename (e.g. ``"subject010_REM.edf"``) to the
        dream report string.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        KeyError: if expected column names are absent.
    """
    reports_csv_path = Path(reports_csv_path)
    if not reports_csv_path.exists():
        raise FileNotFoundError(f"Reports CSV not found: {reports_csv_path}")

    index: Dict[str, str] = {}
    with reports_csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        if _CSV_COL_FILENAME not in fieldnames:
            raise KeyError(
                f"Column '{_CSV_COL_FILENAME}' not found in {reports_csv_path}. "
                f"Available columns: {fieldnames}"
            )
        if _CSV_COL_TEXT not in fieldnames:
            raise KeyError(
                f"Column '{_CSV_COL_TEXT}' not found in {reports_csv_path}. "
                f"Available columns: {fieldnames}"
            )

        for row in reader:
            filename = row[_CSV_COL_FILENAME].strip()
            text = row[_CSV_COL_TEXT].strip()
            if filename and text:
                index[filename] = text

    logger.info(
        "load_reports_csv: loaded %d report entries from %s",
        len(index), reports_csv_path,
    )
    return index


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_report(text: str, model) -> np.ndarray:
    """Encode a single dream report into a 384-dim float32 embedding.

    Args:
        text: Raw dream report string.
        model: A SentenceTransformer (or test stub) with a ``.encode()``
            method.

    Returns:
        np.ndarray of shape ``(384,)`` and dtype ``float32``.

    Raises:
        AssertionError: if the output shape is unexpected.
    """
    embedding = model.encode(text, convert_to_numpy=True).astype(np.float32)
    assert embedding.shape == (384,), (
        f"Expected embedding shape (384,), got {embedding.shape}"
    )
    return embedding


def encode_reports_batch(texts: List[str], model) -> np.ndarray:
    """Encode a list of dream reports as a float32 matrix.

    Args:
        texts: List of report strings.
        model: A SentenceTransformer (or compatible stub).

    Returns:
        np.ndarray of shape ``(len(texts), 384)`` and dtype ``float32``.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    embeddings = np.stack([encode_report(t, model) for t in texts]).astype(np.float32)
    logger.info(
        "encode_reports_batch: encoded %d texts  output shape=%s",
        len(texts), embeddings.shape,
    )
    return embeddings


def clean_report_text(text: str, cfg: Optional[dict] = None) -> str:
    """Normalize report text and remove lightweight filler words."""
    cfg = cfg or {}
    filler_words = [str(word).strip() for word in cfg.get("filler_words", list(_DEFAULT_FILLER_WORDS))]
    cleaned = " ".join(str(text).replace("\n", " ").split())
    for filler in filler_words:
        if filler:
            cleaned = re.sub(rf"\b{re.escape(filler)}\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
    return cleaned.strip()


def is_report_usable(text: str, cfg: Optional[dict] = None) -> bool:
    """Return True when a cleaned report contains enough signal to train on."""
    cfg = cfg or {}
    cleaned = clean_report_text(text, cfg)
    min_report_chars = int(cfg.get("min_report_chars", 20))
    min_report_alpha_chars = int(cfg.get("min_report_alpha_chars", 10))
    drop_patterns = {str(item).strip().lower() for item in cfg.get("drop_report_patterns", list(_DEFAULT_DROP_PATTERNS))}
    alpha_chars = sum(1 for char in cleaned if char.isalpha())

    if len(cleaned) < min_report_chars:
        return False
    if alpha_chars < min_report_alpha_chars:
        return False
    if cleaned.lower() in drop_patterns:
        return False
    return True


# ---------------------------------------------------------------------------
# Subject epoch loading
# ---------------------------------------------------------------------------

def load_subject_epochs(subject_npz_path: Path) -> dict:
    """Load the arrays needed for alignment from a preprocessed ``.npz`` file.

    Args:
        subject_npz_path: Path to ``sub-<id>_epochs.npz``.

    Returns:
        Dict with keys: ``subject_id``, ``source_files``, ``segment_types``,
        ``segment_indices``, ``sleep_stages``, ``epoch_times_s``.
    """
    d = np.load(str(subject_npz_path), allow_pickle=True)
    result = {
        "subject_id": str(d["subject_id"]),
        "source_files": d["source_files"],
        "segment_types": d["segment_types"],
        "segment_indices": d["segment_indices"],
        "sleep_stages": d["sleep_stages"],
        "epoch_times_s": d["epoch_times_s"],
    }
    logger.info(
        "load_subject_epochs: subject=%s  n_epochs=%d",
        result["subject_id"], len(result["sleep_stages"]),
    )
    return result


# ---------------------------------------------------------------------------
# Per-subject alignment
# ---------------------------------------------------------------------------

def align_subject(
    subject_id: str,
    epochs_npz_path: Path,
    report_index: Dict[str, str],
    model,
    cfg: dict,
) -> Optional[Dict]:
    """Align dream report embeddings to EEG epochs for one subject.

    For each unique EDF source file in the epochs NPZ, the function looks up
    the matching report in *report_index* (keyed by EDF basename), encodes it,
    and assigns the embedding to every epoch from that source file.

    Args:
        subject_id: Subject identifier string.
        epochs_npz_path: Path to ``sub-<id>_epochs.npz``.
        report_index: Dict mapping EDF basename → report text, as returned by
            :func:`load_reports_csv`.
        model: SentenceTransformer (or test stub).
        cfg: Config dict from ``configs/align_reports.yaml``.  Used for
            ``align_segment_types`` and ``embedding_dim``.

    Returns:
        Dict with keys ``epoch_indices``, ``target_embeddings``,
        ``segment_types``, ``source_files``, ``report_texts``; or ``None``
        if no epochs could be aligned (missing reports for all segments).
    """
    if not epochs_npz_path.exists():
        logger.warning(
            "align_subject: epochs file not found for subject %s: %s",
            subject_id, epochs_npz_path,
        )
        return None

    epoch_data = load_subject_epochs(epochs_npz_path)
    source_files: np.ndarray = epoch_data["source_files"]
    segment_types: np.ndarray = epoch_data["segment_types"]

    # Determine which segment types to align
    align_types = cfg.get("align_segment_types", "all")
    if align_types != "all" and isinstance(align_types, list):
        align_types_set = set(align_types)
    else:
        align_types_set = None   # align everything

    matched_indices: List[int] = []
    matched_embeddings: List[np.ndarray] = []
    matched_seg_types: List[str] = []
    matched_src_files: List[str] = []
    matched_report_texts: List[str] = []

    for unique_source in np.unique(source_files):
        source_name = str(unique_source)
        epoch_mask = source_files == unique_source
        epoch_idxs = np.where(epoch_mask)[0]

        # Check segment-type filter
        seg_type_for_source = str(segment_types[epoch_mask][0]) if epoch_mask.any() else ""
        if align_types_set is not None and seg_type_for_source not in align_types_set:
            logger.info(
                "align_subject: subject %s  skipping source %s (segment_type=%s not in align list)",
                subject_id, source_name, seg_type_for_source,
            )
            continue

        # Look up report
        if source_name not in report_index:
            logger.info(
                "align_subject: subject %s  no report for source %s — skipping",
                subject_id, source_name,
            )
            continue

        report_text = report_index[source_name]
        cleaned_report_text = clean_report_text(report_text, cfg)
        if not is_report_usable(cleaned_report_text, cfg):
            logger.info(
                "align_subject: subject %s  skipping source %s because cleaned report is too weak: %r",
                subject_id,
                source_name,
                cleaned_report_text,
            )
            continue

        # Encode (cache in caller for large batches; here we encode per source)
        try:
            embedding = encode_report(cleaned_report_text, model)
        except Exception as exc:
            logger.warning(
                "align_subject: subject %s  encoding failed for %s — %s",
                subject_id, source_name, exc,
            )
            continue

        n = len(epoch_idxs)
        matched_indices.extend(epoch_idxs.tolist())
        matched_embeddings.extend([embedding] * n)
        matched_seg_types.extend([seg_type_for_source] * n)
        matched_src_files.extend([source_name] * n)
        matched_report_texts.extend([cleaned_report_text] * n)

        logger.info(
            "align_subject: subject %s  source=%s  n_epochs=%d  report_len=%d",
            subject_id, source_name, n, len(report_text),
        )

    if not matched_indices:
        logger.warning(
            "align_subject: subject %s — no epochs aligned (missing reports for all segments)",
            subject_id,
        )
        return None

    epoch_indices_arr = np.array(matched_indices, dtype=np.int64)
    target_embeddings_arr = np.stack(matched_embeddings).astype(np.float32)

    logger.info(
        "align_subject: subject %s  total aligned epochs=%d  embeddings shape=%s",
        subject_id, len(epoch_indices_arr), target_embeddings_arr.shape,
    )

    return {
        "epoch_indices": epoch_indices_arr,
        "target_embeddings": target_embeddings_arr,
        "segment_types": np.array(matched_seg_types, dtype=object),
        "source_files": np.array(matched_src_files, dtype=object),
        "report_texts": np.array(matched_report_texts, dtype=object),
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_target_embeddings(result: dict, output_path: Path) -> None:
    """Save alignment output to a ``.npz`` file.

    Args:
        result: Dict returned by :func:`align_subject`.
        output_path: Destination ``.npz`` path (parent directory must exist or
            will be created).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(output_path),
        epoch_indices=result["epoch_indices"].astype(np.int64),
        target_embeddings=result["target_embeddings"].astype(np.float32),
        segment_types=np.array(result["segment_types"], dtype=object),
        source_files=np.array(result["source_files"], dtype=object),
        report_texts=np.array(result["report_texts"], dtype=object),
    )
    logger.info(
        "save_target_embeddings: wrote %s  n=%d  embeddings=%s",
        output_path,
        len(result["epoch_indices"]),
        result["target_embeddings"].shape,
    )

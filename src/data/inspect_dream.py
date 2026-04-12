"""DREAM dataset inspection module.

Implements the Dataset Inspection Protocol from AGENTS.md (Steps 1-6):
- Directory and file audit
- EEG format identification and signal probing (via MNE)
- Dream report structure inspection
- Subject/session structure discovery
- Dataset card and machine-readable report generation

All functions use the project logger (INFO level) and log tensor/data shapes
as required by AGENTS.md rule #6.
"""

import collections
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EEG_SUFFIXES = {".edf", ".set", ".fif", ".vhdr"}


def _load_raw(path: Path):
    """Load a raw EEG file using the appropriate MNE reader."""
    import mne  # deferred import — MNE not needed for non-EEG operations

    suffix = path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose=False)
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose=False)
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(str(path), preload=False, verbose=False)
    elif suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG file extension: {suffix}")
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def audit_directory(root: Path) -> Dict:
    """Walk root and return a directory tree + extension counts.

    Args:
        root: Path to the directory to audit.

    Returns:
        dict with keys:
            - ``tree``: list of strings, each entry represents a file/dir
              indented by depth (capped at depth 3).
            - ``ext_counts``: Counter of file extensions.
            - ``total_files``: total number of files found.
    """
    root = Path(root)
    tree: List[str] = []
    ext_counts: Dict[str, int] = collections.Counter()
    total_files = 0

    for p in sorted(root.rglob("*")):
        try:
            depth = len(p.relative_to(root).parts)
        except ValueError:
            continue
        if depth > 3:
            continue
        indent = "  " * (depth - 1)
        if p.is_dir():
            tree.append(f"{indent}{p.name}/")
        else:
            size_kb = p.stat().st_size // 1024
            tree.append(f"{indent}{p.name} [{size_kb} KB]")
            ext_counts[p.suffix] += 1
            total_files += 1

    logger.info("audit_directory: root=%s total_files=%d ext_counts=%s", root, total_files, dict(ext_counts))
    return {"tree": tree, "ext_counts": dict(ext_counts), "total_files": total_files}


def probe_eeg_file(path: Path) -> Dict:
    """Probe a single EEG file and return signal properties.

    Args:
        path: Path to the EEG file (must exist).

    Returns:
        dict with keys: path, sfreq, n_channels, ch_names, ch_types,
        duration_s, annotations.

    Raises:
        FileNotFoundError: if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")

    raw = _load_raw(path)
    sfreq = raw.info["sfreq"]
    ch_names = list(raw.ch_names)
    ch_types = list(raw.get_channel_types())
    n_channels = len(ch_names)
    duration_s = raw.times[-1] if len(raw.times) else 0.0

    annotations = []
    for ann in raw.annotations:
        annotations.append({
            "onset": float(ann["onset"]),
            "duration": float(ann["duration"]),
            "description": str(ann["description"]),
        })

    logger.info(
        "probe_eeg_file: path=%s sfreq=%.1f n_channels=%d duration_s=%.1f n_annotations=%d",
        path, sfreq, n_channels, duration_s, len(annotations),
    )
    return {
        "path": str(path),
        "sfreq": sfreq,
        "n_channels": n_channels,
        "ch_names": ch_names,
        "ch_types": ch_types,
        "duration_s": duration_s,
        "annotations": annotations,
    }


def probe_report_files(root: Path, pattern: str = "sub-*_dream.txt") -> Dict:
    """Inspect dream report files under root.

    Looks for files matching common patterns. Returns stats about the reports
    found.

    Args:
        root: Directory to search under.
        pattern: Primary glob pattern (default: 'sub-*_dream.txt').

    Returns:
        dict with keys: n_reports, example_path, avg_length_chars,
        example_snippet.
    """
    root = Path(root)
    patterns = [pattern, "sub-*_report.txt", "*.tsv", "*.csv", "*.json"]

    matches: List[Path] = []
    for pat in patterns:
        found = sorted(root.rglob(pat))
        if found:
            matches = found
            break

    if not matches:
        logger.info("probe_report_files: no report files found under %s", root)
        return {
            "n_reports": 0,
            "example_path": None,
            "avg_length_chars": 0.0,
            "example_snippet": None,
        }

    lengths = []
    for f in matches:
        try:
            text = f.read_text(errors="replace")
            lengths.append(len(text))
        except Exception:
            lengths.append(0)

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    example_path = str(matches[0])
    try:
        snippet = matches[0].read_text(errors="replace")[:200]
    except Exception:
        snippet = None

    logger.info(
        "probe_report_files: n_reports=%d avg_length_chars=%.0f example=%s",
        len(matches), avg_len, example_path,
    )
    return {
        "n_reports": len(matches),
        "example_path": example_path,
        "avg_length_chars": avg_len,
        "example_snippet": snippet,
    }


def discover_subjects(root: Path) -> List[str]:
    """Return sorted list of unique subject IDs under root.

    Subject IDs are parsed from filenames matching the pattern ``sub-<id>``.

    Args:
        root: Directory to search under.

    Returns:
        Sorted list of subject ID strings (e.g. ["01", "02"]).
    """
    root = Path(root)
    sub_re = re.compile(r"sub-([A-Za-z0-9]+)")
    ids = set()
    for p in root.rglob("*"):
        m = sub_re.search(p.name)
        if m:
            ids.add(m.group(1))
    result = sorted(ids)
    logger.info("discover_subjects: root=%s n_subjects=%d", root, len(result))
    return result


def run_inspection(config: dict) -> Dict:
    """Orchestrate a full dataset inspection.

    Runs audit, EEG probing, report probing, and subject discovery.
    Individual probe failures are caught and recorded in ``known_issues``.

    Args:
        config: dict with keys ``raw_root`` (str) and
            ``max_files_to_probe`` (int, default 3).

    Returns:
        dict with keys: audit, eeg_sample, reports, subjects, known_issues.
    """
    raw_root = Path(config["raw_root"])
    max_probe = int(config.get("max_files_to_probe", 3))
    known_issues: List[str] = []

    # Handle missing raw_root gracefully
    if not raw_root.exists():
        logger.info("run_inspection: raw_root does not exist: %s", raw_root)
        known_issues.append(f"raw_root_missing: {raw_root}")
        return {
            "audit": {"tree": [], "ext_counts": {}, "total_files": 0},
            "eeg_sample": None,
            "reports": {
                "n_reports": 0,
                "example_path": None,
                "avg_length_chars": 0.0,
                "example_snippet": None,
            },
            "subjects": [],
            "known_issues": known_issues,
        }

    # Step 1: Directory audit
    audit = audit_directory(raw_root)

    # Step 2: Probe EEG files (first max_probe by sorted path)
    eeg_files = sorted(
        [p for p in raw_root.rglob("*") if p.suffix.lower() in _EEG_SUFFIXES]
    )[:max_probe]

    eeg_sample = None
    for eeg_path in eeg_files:
        try:
            eeg_sample = probe_eeg_file(eeg_path)
            break  # use first successful probe
        except Exception as exc:
            issue = f"eeg_probe_failed:{eeg_path.name}:{exc}"
            known_issues.append(issue)
            logger.warning("EEG probe failed for %s: %s", eeg_path, exc)

    # Step 3: Dream report probing
    try:
        reports = probe_report_files(raw_root)
    except Exception as exc:
        reports = {
            "n_reports": 0,
            "example_path": None,
            "avg_length_chars": 0.0,
            "example_snippet": None,
        }
        known_issues.append(f"report_probe_failed:{exc}")
        logger.warning("Report probe failed: %s", exc)

    # Step 4: Subject discovery
    subjects = discover_subjects(raw_root)

    logger.info(
        "run_inspection complete: total_files=%d subjects=%d eeg_sample=%s known_issues=%d",
        audit["total_files"], len(subjects), bool(eeg_sample), len(known_issues),
    )
    return {
        "audit": audit,
        "eeg_sample": eeg_sample,
        "reports": reports,
        "subjects": subjects,
        "known_issues": known_issues,
    }


def write_dataset_card(inspection: dict, output_path: Path) -> None:
    """Write a DREAM dataset card to output_path.

    The card contains exactly these H2 sections in order:
    Format, Signal Properties, Subject and Session Counts, Label Structure,
    Known Issues.

    Args:
        inspection: dict returned by run_inspection().
        output_path: Path where the Markdown file will be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit = inspection.get("audit", {})
    eeg = inspection.get("eeg_sample") or {}
    reports = inspection.get("reports", {})
    subjects = inspection.get("subjects", [])
    known_issues = inspection.get("known_issues", [])

    lines = [
        "# DREAM Dataset Card",
        "",
        "## Format",
        "",
    ]

    # Format section
    ext_counts = audit.get("ext_counts", {})
    if ext_counts:
        lines.append("File extensions found:")
        for ext, count in sorted(ext_counts.items()):
            lines.append(f"- `{ext}`: {count} files")
    else:
        lines.append("No files found in raw directory.")
    lines.append("")

    # Signal Properties
    lines += [
        "## Signal Properties",
        "",
    ]
    if eeg:
        lines.append(f"sfreq: {eeg.get('sfreq', 'unknown')} Hz")
        lines.append(f"n_channels: {eeg.get('n_channels', 'unknown')}")
        lines.append(f"duration_s: {eeg.get('duration_s', 'unknown')}")
        ann_count = len(eeg.get("annotations", []))
        lines.append(f"annotations: {ann_count} annotation(s) found")
        ch_types = list(set(eeg.get("ch_types", [])))
        lines.append(f"channel types: {ch_types}")
    else:
        lines.append("sfreq: not available (no EEG file probed)")
        lines.append("n_channels: not available")
        lines.append("duration_s: not available")
        lines.append("annotations: not available")
    lines.append("")

    # Subject and Session Counts
    n_subjects = len(subjects)
    lines += [
        "## Subject and Session Counts",
        "",
        f"n_subjects: {n_subjects}",
    ]
    if subjects:
        lines.append(f"subject IDs: {subjects[:10]}" + (" ..." if len(subjects) > 10 else ""))
    else:
        lines.append("No subjects detected (raw data may not be downloaded).")
    lines.append("")

    # Label Structure
    lines += [
        "## Label Structure",
        "",
    ]
    n_reports = reports.get("n_reports", 0)
    if n_reports > 0:
        lines.append(f"- {n_reports} dream report file(s) found.")
        if reports.get("example_path"):
            lines.append(f"- Example: `{reports['example_path']}`")
        avg_len = reports.get("avg_length_chars", 0.0)
        lines.append(f"- Average report length: {avg_len:.0f} characters.")
        lines.append("- Reports are free-text; semantic alignment will be used (not word-level CTC).")
        if reports.get("example_snippet"):
            lines.append(f"- Snippet: `{reports['example_snippet'][:100]}...`")
    else:
        lines.append("No dream report files found.")
        lines.append("Expected pattern: `sub-*_dream.txt` or similar.")
        lines.append("Semantic targets will be generated via sentence-transformer embeddings.")
    lines.append("")

    # Known Issues
    lines += [
        "## Known Issues",
        "",
    ]
    if known_issues:
        for issue in known_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("None identified.")
    lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content)
    logger.info("write_dataset_card: written to %s (%d bytes)", output_path, len(content))


def write_inspection_report(inspection: dict, output_path: Path) -> None:
    """Write inspection results as JSON to output_path.

    Args:
        inspection: dict returned by run_inspection().
        output_path: Path where the JSON file will be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(inspection, indent=2, default=str)
    output_path.write_text(content)
    logger.info("write_inspection_report: written to %s (%d bytes)", output_path, len(content))

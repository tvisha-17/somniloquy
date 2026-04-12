"""DREAM dataset inspection module — filename-aware edition.

Adapted for the Zhang & Wamsley 2019 release in which EEG recordings are
already split into per-subject, per-condition EDF files:

    subject<NNN>_REM.edf
    subject<NNN>_NREM.edf
    subject<NNN>_Morning.edf
    subject<NNN>_SO1.edf  …  subject<NNN>_SO10.edf

Primary responsibilities
------------------------
1. Recursively scan a raw directory and parse every EDF filename into
   (subject_id, segment_type, segment_index).
2. Compute dataset-level statistics (subjects, files per type, per subject).
3. Flag missing or irregular segment patterns.
4. Probe a sample of EDF files with MNE (sfreq, channels, duration,
   annotations).
5. Write a human-readable DATASET_CARD.md and a machine-readable
   inspection_report.json.

All functions log data shapes at INFO level as required by the project
logging policy.
"""

import collections
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex for the DREAM filename convention.
# Captures:
#   group 1  — subject number string (e.g. "010")
#   group 2  — full segment token  (e.g. "REM", "NREM", "Morning", "SO3")
#   group 3  — SO index digit(s) when present, else None
_FNAME_RE = re.compile(
    r"^subject(\d+)_(REM|NREM|Morning|SO(\d+))(?:\.edf)?$",
    re.IGNORECASE,
)

# Expected segment types that should be present for a "complete" subject.
_EXPECTED_BASE_TYPES = {"REM", "NREM", "Morning"}
_EXPECTED_SO_COUNT = 10   # SO1 through SO10


# ---------------------------------------------------------------------------
# Filename parser  (shared with other modules)
# ---------------------------------------------------------------------------

def parse_dream_filename(path: "Path | str") -> Optional[Dict]:
    """Parse a DREAM EDF filename into its semantic components.

    Supports both bare names (``subject010_REM``) and full paths.

    Args:
        path: A Path or string pointing to (or named like) a DREAM EDF file.

    Returns:
        Dict with keys:

        - ``subject_id`` (str): zero-padded subject number, e.g. ``"010"``.
        - ``segment_type`` (str): one of ``"REM"``, ``"NREM"``,
          ``"Morning"``, ``"SO"``.
        - ``segment_index`` (int): SO index (1–10) or ``-1`` when not
          applicable.

        Returns ``None`` if the filename does not match the expected pattern.
    """
    stem = Path(path).stem   # strip extension if present
    m = _FNAME_RE.match(stem)
    if m is None:
        return None

    subject_id = m.group(1)
    raw_seg = m.group(2)

    if raw_seg.upper().startswith("SO"):
        segment_type = "SO"
        segment_index = int(m.group(3))
    else:
        # Normalise capitalisation: REM, NREM, Morning
        if raw_seg.upper() == "REM":
            segment_type = "REM"
        elif raw_seg.upper() == "NREM":
            segment_type = "NREM"
        else:
            segment_type = raw_seg.capitalize()   # Morning
        segment_index = -1

    return {
        "subject_id": subject_id,
        "segment_type": segment_type,
        "segment_index": segment_index,
    }


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------

def scan_dream_directory(raw_root: Path) -> List[Dict]:
    """Recursively scan *raw_root* and parse every EDF filename.

    Args:
        raw_root: Root directory containing EDF files (may be nested).

    Returns:
        List of record dicts, one per recognised EDF.  Each record contains
        ``subject_id``, ``segment_type``, ``segment_index``, and ``path``
        (absolute Path to the file).  Files that do not match the expected
        naming pattern are logged at WARNING and omitted.
    """
    raw_root = Path(raw_root)
    records: List[Dict] = []
    unmatched: List[str] = []

    for p in sorted(raw_root.rglob("*.edf")):
        parsed = parse_dream_filename(p)
        if parsed is None:
            unmatched.append(p.name)
            logger.warning("scan_dream_directory: unrecognised filename %s", p.name)
            continue
        records.append({**parsed, "path": p})

    logger.info(
        "scan_dream_directory: root=%s  matched=%d  unmatched=%d",
        raw_root, len(records), len(unmatched),
    )
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_dataset_stats(records: List[Dict]) -> Dict:
    """Compute subject- and segment-level counts from a list of file records.

    Args:
        records: List of records as returned by :func:`scan_dream_directory`.

    Returns:
        Dict with keys:

        - ``n_subjects`` (int)
        - ``subject_ids`` (sorted list of str)
        - ``files_per_segment_type`` (dict: segment_type → count)
        - ``files_per_subject`` (dict: subject_id → count)
        - ``so_counts_per_subject`` (dict: subject_id → n_SO_files)
    """
    n_by_type: Dict[str, int] = collections.Counter()
    n_by_subject: Dict[str, int] = collections.Counter()
    so_by_subject: Dict[str, int] = collections.Counter()

    for r in records:
        n_by_type[r["segment_type"]] += 1
        n_by_subject[r["subject_id"]] += 1
        if r["segment_type"] == "SO":
            so_by_subject[r["subject_id"]] += 1

    subject_ids = sorted(n_by_subject.keys())

    logger.info(
        "compute_dataset_stats: n_subjects=%d  segment_types=%s",
        len(subject_ids), dict(n_by_type),
    )
    return {
        "n_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "files_per_segment_type": dict(n_by_type),
        "files_per_subject": dict(n_by_subject),
        "so_counts_per_subject": dict(so_by_subject),
    }


# ---------------------------------------------------------------------------
# Irregular-pattern checker
# ---------------------------------------------------------------------------

def check_irregular_patterns(records: List[Dict]) -> List[str]:
    """Detect missing or unusual segment patterns across subjects.

    Checks:

    - Each subject should have exactly one REM, NREM, and Morning file.
    - Each subject should ideally have SO1 through SO10 (warns on gaps).
    - Duplicate (subject, segment_type, segment_index) tuples are flagged.

    Args:
        records: List of records from :func:`scan_dream_directory`.

    Returns:
        List of human-readable issue strings.  Empty list means no issues.
    """
    issues: List[str] = []

    # Index records by (subject_id, segment_type, segment_index)
    seen: Dict[Tuple, int] = collections.Counter()
    by_subject: Dict[str, List[Dict]] = collections.defaultdict(list)

    for r in records:
        key = (r["subject_id"], r["segment_type"], r["segment_index"])
        seen[key] += 1
        by_subject[r["subject_id"]].append(r)

    # Duplicate check
    for key, count in seen.items():
        if count > 1:
            issues.append(
                f"duplicate: subject {key[0]} segment {key[1]}"
                + (f"[{key[2]}]" if key[2] != -1 else "")
                + f"  ({count} files)"
            )

    # Per-subject completeness
    for subject_id, recs in sorted(by_subject.items()):
        types_found = {r["segment_type"] for r in recs}

        # Check base types
        for bt in _EXPECTED_BASE_TYPES:
            if bt not in types_found:
                issues.append(f"missing_{bt}: subject {subject_id}")

        # Check SO sequence
        if "SO" in types_found:
            so_indices = sorted(
                r["segment_index"] for r in recs if r["segment_type"] == "SO"
            )
            expected = list(range(1, _EXPECTED_SO_COUNT + 1))
            missing_so = [i for i in expected if i not in so_indices]
            if missing_so:
                issues.append(
                    f"missing_SO_indices: subject {subject_id}  missing={missing_so}"
                )
        else:
            issues.append(f"missing_all_SO: subject {subject_id}")

    logger.info("check_irregular_patterns: %d issue(s) found", len(issues))
    return issues


# ---------------------------------------------------------------------------
# EDF sample probing
# ---------------------------------------------------------------------------

def probe_edf_sample(paths: List[Path], max_probe: int = 3) -> List[Dict]:
    """Probe up to *max_probe* EDF files with MNE and return their properties.

    Args:
        paths: List of Path objects to candidate EDF files.
        max_probe: Maximum number of files to actually open.

    Returns:
        List of probe dicts, one per successfully opened file.  Each dict
        contains ``path``, ``sfreq``, ``n_channels``, ``ch_names``,
        ``ch_types``, ``duration_s``, ``n_annotations``, and
        ``annotation_descriptions``.
    """
    import mne  # deferred — not needed for non-EEG operations

    results: List[Dict] = []
    for p in paths[:max_probe]:
        try:
            raw = mne.io.read_raw_edf(str(p), preload=False, verbose=False)
            ch_types = raw.get_channel_types()
            duration_s = raw.times[-1] if len(raw.times) > 0 else 0.0
            ann_descs = [str(a["description"]) for a in raw.annotations]

            info = {
                "path": str(p),
                "sfreq": raw.info["sfreq"],
                "n_channels": len(raw.ch_names),
                "ch_names": list(raw.ch_names),
                "ch_types": list(ch_types),
                "duration_s": float(duration_s),
                "n_annotations": len(ann_descs),
                "annotation_descriptions": ann_descs[:20],  # cap for readability
            }
            logger.info(
                "probe_edf_sample: %s  sfreq=%.1f  n_ch=%d  duration_s=%.1f  n_ann=%d",
                p.name, info["sfreq"], info["n_channels"],
                info["duration_s"], info["n_annotations"],
            )
            results.append(info)
        except Exception as exc:
            logger.warning("probe_edf_sample: failed to open %s — %s", p, exc)

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_inspection(config: dict) -> Dict:
    """Run a full dataset inspection for the DREAM dataset.

    Steps performed:

    1. Scan *raw_root* and parse all EDF filenames.
    2. Compute dataset statistics.
    3. Check for irregular / missing segment patterns.
    4. Probe a sample of EDF files with MNE.

    Args:
        config: Dict with keys:

            - ``raw_root`` (str | Path): directory to scan.
            - ``max_files_to_probe`` (int, default 3): EDF files to probe.
            - ``output_card`` (str | Path, optional): written by caller.
            - ``output_report`` (str | Path, optional): written by caller.

    Returns:
        Dict with keys ``records``, ``stats``, ``issues``, ``edf_probes``,
        ``known_errors``.
    """
    raw_root = Path(config["raw_root"])
    max_probe = int(config.get("max_files_to_probe", 3))
    known_errors: List[str] = []

    if not raw_root.exists():
        known_errors.append(f"raw_root_missing: {raw_root}")
        logger.warning("run_inspection: raw_root does not exist: %s", raw_root)
        return {
            "records": [],
            "stats": {
                "n_subjects": 0,
                "subject_ids": [],
                "files_per_segment_type": {},
                "files_per_subject": {},
                "so_counts_per_subject": {},
            },
            "issues": [],
            "edf_probes": [],
            "known_errors": known_errors,
        }

    # Step 1: scan + parse
    records = scan_dream_directory(raw_root)

    # Step 2: stats
    stats = compute_dataset_stats(records)

    # Step 3: pattern irregularities
    issues = check_irregular_patterns(records)

    # Step 4: EDF probing (sample first max_probe unique subjects)
    sample_paths = [r["path"] for r in records[:max_probe]]
    edf_probes = probe_edf_sample(sample_paths, max_probe)

    logger.info(
        "run_inspection: complete  n_subjects=%d  n_files=%d  issues=%d  probes=%d",
        stats["n_subjects"], len(records), len(issues), len(edf_probes),
    )
    return {
        "records": [
            {**r, "path": str(r["path"])} for r in records  # JSON-serialisable
        ],
        "stats": stats,
        "issues": issues,
        "edf_probes": edf_probes,
        "known_errors": known_errors,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_dataset_card(inspection: dict, output_path: Path) -> None:
    """Write a DREAM dataset card as Markdown to *output_path*.

    Sections: Format, Signal Properties, Subject and Session Counts,
    Label Structure, Known Issues.

    Args:
        inspection: Dict returned by :func:`run_inspection`.
        output_path: Destination path for the ``.md`` file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = inspection.get("stats", {})
    probes = inspection.get("edf_probes", [])
    issues = inspection.get("issues", [])
    errors = inspection.get("known_errors", [])
    records = inspection.get("records", [])

    lines: List[str] = [
        "# DREAM Dataset Card",
        "",
        "> Auto-generated by `src/data/inspect_dream_dataset.py`.",
        "",
        "## Format",
        "",
        "Files are pre-split per-subject, per-condition EDF recordings from the",
        "Zhang & Wamsley (2019) dataset.",
        "",
        "**Filename convention:** `subject<NNN>_<SegmentType>.edf`",
        "",
        "Supported segment types:",
        "- `REM` — isolated REM-sleep segment",
        "- `NREM` — isolated NREM-sleep segment",
        "- `Morning` — morning wakefulness / verbal recall",
        "- `SO1` … `SO10` — sleep-onset segments",
        "",
    ]

    # --------------- Signal Properties ---------------
    lines += ["## Signal Properties", ""]
    if probes:
        p = probes[0]
        lines += [
            f"Sample file: `{Path(p['path']).name}`",
            "",
            f"| Property | Value |",
            f"|---|---|",
            f"| sfreq | {p['sfreq']} Hz |",
            f"| n_channels | {p['n_channels']} |",
            f"| duration_s | {p['duration_s']:.1f} s |",
            f"| n_annotations | {p['n_annotations']} |",
            f"| channel types | {sorted(set(p['ch_types']))} |",
            "",
        ]
        if p.get("ch_names"):
            ch_preview = p["ch_names"][:10]
            lines.append(f"First channels: `{ch_preview}`")
            lines.append("")
    else:
        lines += ["Signal properties unavailable (no files probed).", ""]

    # --------------- Subject and Session Counts ---------------
    lines += ["## Subject and Session Counts", ""]
    n_sub = stats.get("n_subjects", 0)
    lines.append(f"**Total subjects:** {n_sub}")
    lines.append(f"**Total EDF files:** {len(records)}")
    lines.append("")

    by_type = stats.get("files_per_segment_type", {})
    if by_type:
        lines.append("**Files per segment type:**")
        lines.append("")
        lines.append("| Segment type | File count |")
        lines.append("|---|---|")
        for stype, cnt in sorted(by_type.items()):
            lines.append(f"| {stype} | {cnt} |")
        lines.append("")

    sub_ids = stats.get("subject_ids", [])
    if sub_ids:
        preview = sub_ids[:10]
        suffix = " …" if len(sub_ids) > 10 else ""
        lines.append(f"**Subject IDs:** {preview}{suffix}")
        lines.append("")

    # --------------- Label Structure ---------------
    lines += [
        "## Label Structure",
        "",
        "Sleep-stage labels are inferred from EDF filenames; raw annotations",
        "are preserved in metadata but are **not** the primary label source.",
        "",
        "| Segment type | Integer label |",
        "|---|---|",
        "| REM | 4 |",
        "| NREM | 2 (undifferentiated; N2 default) |",
        "| Morning | 5 |",
        "| SO1–SO10 | 6 |",
        "",
        "Dream reports are stored in `Reports.csv` and are linked to EDF files",
        "via the `Filename` column.",
        "",
    ]

    # --------------- Known Issues ---------------
    lines += ["## Known Issues", ""]
    all_issues = issues + errors
    if all_issues:
        for issue in all_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("None identified.")
    lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content)
    logger.info(
        "write_dataset_card: written to %s  (%d bytes)", output_path, len(content)
    )


def write_inspection_report(inspection: dict, output_path: Path) -> None:
    """Serialise the full inspection result to a JSON file.

    Args:
        inspection: Dict returned by :func:`run_inspection`.
        output_path: Destination ``.json`` path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(inspection, indent=2, default=str)
    output_path.write_text(content)
    logger.info(
        "write_inspection_report: written to %s  (%d bytes)",
        output_path, len(content),
    )

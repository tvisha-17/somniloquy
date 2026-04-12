"""Unit tests for src.data.inspect_dream — TDD RED phase."""

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. audit_directory — empty dir
# ---------------------------------------------------------------------------


def test_audit_directory_empty(tmp_path):
    from src.data.inspect_dream import audit_directory

    result = audit_directory(tmp_path)
    assert result["total_files"] == 0
    assert result["ext_counts"] == {}
    assert result["tree"] == []


# ---------------------------------------------------------------------------
# 2. audit_directory — counts extensions
# ---------------------------------------------------------------------------


def test_audit_directory_counts_extensions(tmp_path):
    from src.data.inspect_dream import audit_directory

    (tmp_path / "a.edf").write_text("x")
    (tmp_path / "b.edf").write_text("x")
    (tmp_path / "c.txt").write_text("x")

    result = audit_directory(tmp_path)
    assert result["ext_counts"] == {".edf": 2, ".txt": 1}
    assert result["total_files"] == 3


# ---------------------------------------------------------------------------
# 3. probe_eeg_file — missing file raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_probe_eeg_file_missing_raises(tmp_path):
    from src.data.inspect_dream import probe_eeg_file

    with pytest.raises(FileNotFoundError):
        probe_eeg_file(tmp_path / "nonexistent.edf")


# ---------------------------------------------------------------------------
# 4. discover_subjects — parses sub-<id> pattern
# ---------------------------------------------------------------------------


def test_discover_subjects_parses_sub_ids(tmp_path):
    from src.data.inspect_dream import discover_subjects

    (tmp_path / "sub-01_eeg.edf").write_text("x")
    (tmp_path / "sub-02_eeg.edf").write_text("x")
    (tmp_path / "sub-01_dream.txt").write_text("x")

    result = discover_subjects(tmp_path)
    assert result == ["01", "02"]


# ---------------------------------------------------------------------------
# 5. write_dataset_card — contains required H2 sections
# ---------------------------------------------------------------------------


def test_write_dataset_card_contains_required_sections(tmp_path):
    from src.data.inspect_dream import write_dataset_card

    inspection = {
        "audit": {"tree": [], "ext_counts": {}, "total_files": 0},
        "eeg_sample": None,
        "reports": {"n_reports": 0, "example_path": None, "avg_length_chars": 0.0, "example_snippet": None},
        "subjects": [],
        "known_issues": [],
    }
    out = tmp_path / "DATASET_CARD.md"
    write_dataset_card(inspection, out)

    content = out.read_text()
    assert "# DREAM Dataset Card" in content
    assert "## Format" in content
    assert "## Signal Properties" in content
    assert "## Subject and Session Counts" in content
    assert "## Label Structure" in content
    assert "## Known Issues" in content


# ---------------------------------------------------------------------------
# 6. write_inspection_report — valid JSON
# ---------------------------------------------------------------------------


def test_write_inspection_report_is_valid_json(tmp_path):
    from src.data.inspect_dream import write_inspection_report

    inspection = {
        "audit": {"tree": [], "ext_counts": {}, "total_files": 0},
        "eeg_sample": None,
        "reports": {"n_reports": 0, "example_path": None, "avg_length_chars": 0.0, "example_snippet": None},
        "subjects": [],
        "known_issues": [],
    }
    out = tmp_path / "report.json"
    write_inspection_report(inspection, out)

    loaded = json.loads(out.read_text())
    assert isinstance(loaded, dict)
    assert "audit" in loaded
    assert "known_issues" in loaded


# ---------------------------------------------------------------------------
# 7. run_inspection — records known_issues when _load_raw raises
# ---------------------------------------------------------------------------


def test_run_inspection_records_known_issues_on_probe_failure(tmp_path, monkeypatch):
    from src.data import inspect_dream

    # Create a fake EEG file so the audit finds something to probe
    eeg_file = tmp_path / "sub-01_eeg.edf"
    eeg_file.write_text("fake edf")

    # Force _load_raw to raise
    def _bad_load(path):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(inspect_dream, "_load_raw", _bad_load)

    config = {
        "raw_root": str(tmp_path),
        "max_files_to_probe": 3,
    }
    result = inspect_dream.run_inspection(config)
    assert len(result["known_issues"]) > 0

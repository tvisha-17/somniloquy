"""Tests for src/data/align_dream_reports.py."""

from pathlib import Path

import numpy as np


class DummyModel:
    def encode(self, text, convert_to_numpy=True):
        return np.ones(384, dtype=np.float32)


def test_clean_report_text_removes_fillers_and_normalizes_spaces():
    from src.data.align_dream_reports import clean_report_text

    cleaned = clean_report_text("umm  I was uh walking  home  ", {"filler_words": ["umm", "uh"]})
    assert cleaned == "I was walking home"


def test_is_report_usable_rejects_tiny_cleaned_text():
    from src.data.align_dream_reports import is_report_usable

    cfg = {"min_report_chars": 10, "min_report_alpha_chars": 5, "filler_words": ["umm"]}
    assert is_report_usable("umm umm", cfg) is False
    assert is_report_usable("I was walking outside", cfg) is True


def test_align_subject_skips_low_information_reports(tmp_path):
    from src.data.align_dream_reports import align_subject

    np.savez(
        tmp_path / "sub-01_epochs.npz",
        subject_id="01",
        source_files=np.array(["subject01_REM.edf", "subject01_REM.edf"], dtype=object),
        segment_types=np.array(["REM", "REM"], dtype=object),
        segment_indices=np.array([-1, -1], dtype=np.int64),
        sleep_stages=np.array([4, 4], dtype=np.int64),
        epoch_times_s=np.array([0.0, 2.0], dtype=np.float64),
    )

    result = align_subject(
        subject_id="01",
        epochs_npz_path=tmp_path / "sub-01_epochs.npz",
        report_index={"subject01_REM.edf": "umm umm"},
        model=DummyModel(),
        cfg={"min_report_chars": 10, "min_report_alpha_chars": 5, "filler_words": ["umm"]},
    )
    assert result is None


def test_align_subject_saves_cleaned_report_text(tmp_path):
    from src.data.align_dream_reports import align_subject

    np.savez(
        tmp_path / "sub-01_epochs.npz",
        subject_id="01",
        source_files=np.array(["subject01_REM.edf"], dtype=object),
        segment_types=np.array(["REM"], dtype=object),
        segment_indices=np.array([-1], dtype=np.int64),
        sleep_stages=np.array([4], dtype=np.int64),
        epoch_times_s=np.array([0.0], dtype=np.float64),
    )

    result = align_subject(
        subject_id="01",
        epochs_npz_path=tmp_path / "sub-01_epochs.npz",
        report_index={"subject01_REM.edf": "umm I was flying uh over water"},
        model=DummyModel(),
        cfg={"min_report_chars": 10, "min_report_alpha_chars": 5, "filler_words": ["umm", "uh"]},
    )
    assert result is not None
    assert result["report_texts"].tolist() == ["I was flying over water"]

"""Tests for src/data/align_reports.py — Agent 1B (Dream Report Aligner).

Uses a DummyModel stub instead of the real SentenceTransformer to avoid
any network access.
"""
from pathlib import Path

import numpy as np
import pytest


class DummyModel:
    """Stub for SentenceTransformer with a fixed zero embedding."""

    def encode(self, text, convert_to_numpy=True):
        return np.zeros(384, dtype=np.float32)


def test_encode_report_returns_384_float32():
    from src.data.align_reports import encode_report

    model = DummyModel()
    result = encode_report("I was flying over a city", model)
    assert result.shape == (384,), f"Expected shape (384,), got {result.shape}"
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_encode_reports_batch_shape():
    from src.data.align_reports import encode_reports_batch

    model = DummyModel()
    texts = ["dream one", "dream two", "dream three"]
    result = encode_reports_batch(texts, model)
    assert result.shape == (3, 384), f"Expected (3, 384), got {result.shape}"
    assert result.dtype == np.float32


def test_find_awakening_times_matches_keywords():
    from src.data.align_reports import find_awakening_times

    annotations = [
        {"onset": 100.0, "description": "awakening"},
        {"onset": 200.0, "description": "arousal_end"},
        {"onset": 300.0, "description": "other event"},
    ]
    result = find_awakening_times(annotations, recording_end_s=400.0)
    assert result == [100.0, 200.0], f"Got {result}"


def test_find_awakening_times_falls_back_to_recording_end():
    from src.data.align_reports import find_awakening_times

    annotations = [
        {"onset": 50.0, "description": "snore"},
        {"onset": 120.0, "description": "movement"},
    ]
    result = find_awakening_times(annotations, recording_end_s=300.0)
    assert result == [300.0], f"Got {result}"


def test_select_rem_epochs_before_awakening_boundaries():
    from src.data.align_reports import select_rem_epochs_before_awakening

    # window=50: [40, 90) -> t=50 (idx 3) and t=80 (idx 4) qualify with stage==4
    sleep_stages = np.array([0, 4, 4, 4, 4, 0], dtype=int)
    epoch_times = np.array([0.0, 10.0, 20.0, 50.0, 80.0, 100.0])
    indices = select_rem_epochs_before_awakening(
        sleep_stages, epoch_times, awakening_time_s=90.0, window_s=50.0
    )
    assert list(indices) == [3, 4], f"Got {list(indices)}"


def test_select_rem_epochs_empty_when_no_rem_in_window():
    from src.data.align_reports import select_rem_epochs_before_awakening

    sleep_stages = np.array([0, 0, 0, 0], dtype=int)
    epoch_times = np.array([0.0, 10.0, 20.0, 30.0])
    indices = select_rem_epochs_before_awakening(
        sleep_stages, epoch_times, awakening_time_s=50.0, window_s=30.0
    )
    assert indices.shape == (0,), f"Expected shape (0,), got {indices.shape}"
    assert indices.dtype == np.int64, f"Expected int64, got {indices.dtype}"


def test_save_target_embeddings_roundtrip(tmp_path):
    from src.data.align_reports import save_target_embeddings

    result = {
        "epoch_indices": np.array([0, 1, 2], dtype=np.int64),
        "target_embeddings": np.zeros((3, 384), dtype=np.float32),
        "report_text": "Flying over mountains",
    }
    out_path = tmp_path / "sub-01_target_embeddings.npz"
    save_target_embeddings(result, out_path)

    loaded = np.load(str(out_path), allow_pickle=True)
    assert "epoch_indices" in loaded
    assert "target_embeddings" in loaded
    assert "report_text" in loaded
    assert loaded["target_embeddings"].dtype == np.float32
    assert loaded["target_embeddings"].shape[1] == 384


def test_align_subject_returns_none_when_no_rem_in_window(tmp_path):
    from src.data.align_reports import align_subject

    n_epochs = 50
    np.savez(
        str(tmp_path / "sub-test01_epochs.npz"),
        data=np.zeros((n_epochs, 4, 512), dtype=np.float32),
        sleep_stages=np.zeros(n_epochs, dtype=int),
        subject_id="test01",
        sfreq=256.0,
        ch_names=np.array(["Fp1", "Fp2", "F3", "F4"]),
        epoch_times_s=np.arange(n_epochs, dtype=np.float64) * 2.0,
    )

    report_path = tmp_path / "sub-test01_dream.txt"
    report_path.write_text("I was in a dark forest")

    cfg = {
        "time_alignment_window": 30.0,
        "embedding_dim": 384,
    }

    model = DummyModel()
    result = align_subject(
        subject_id="test01",
        epochs_path=tmp_path / "sub-test01_epochs.npz",
        report_path=report_path,
        model=model,
        cfg=cfg,
    )
    assert result is None

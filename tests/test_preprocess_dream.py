"""Unit tests for src/data/preprocess_dream.py.

Tests use synthetic MNE RawArray data — no real files required.
"""

import sys
import pathlib

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import mne

from src.data.preprocess_dream import (
    map_sleep_stages,
    zscore_per_channel,
    reject_by_peak_to_peak,
    epoch_data,
    preprocess_subject,
    save_subject_npz,
    _load_raw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(n_channels: int = 4, sfreq: float = 256.0, duration: float = 20.0) -> mne.io.RawArray:
    """Return a synthetic mne.io.RawArray with EEG channel type."""
    n_times = int(sfreq * duration)
    data = np.random.randn(n_channels, n_times).astype(np.float64) * 20e-6
    ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


# ---------------------------------------------------------------------------
# Test 1: map_sleep_stages - known codes
# ---------------------------------------------------------------------------

def test_map_sleep_stages_known_codes():
    labels = ["Wake", "N1", "N2", "N3", "REM", "foo"]
    result = map_sleep_stages(labels)
    assert list(result) == [0, 1, 2, 3, 4, -1], f"Got {result}"


# ---------------------------------------------------------------------------
# Test 2: map_sleep_stages - case-insensitive
# ---------------------------------------------------------------------------

def test_map_sleep_stages_case_insensitive():
    labels = ["wake", "n1", "n2", "n3", "rem", "UNKNOWN"]
    result = map_sleep_stages(labels)
    assert list(result) == [0, 1, 2, 3, 4, -1], f"Got {result}"


# ---------------------------------------------------------------------------
# Test 3: zscore_per_channel - unit variance
# ---------------------------------------------------------------------------

def test_zscore_per_channel_produces_unit_variance():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 4, 100)).astype(np.float32) * 5.0
    out = zscore_per_channel(data)
    assert out.dtype == np.float32
    assert abs(out.mean()) < 0.1, f"Mean too large: {out.mean()}"
    assert abs(out.std() - 1.0) < 0.2, f"Std off: {out.std()}"


# ---------------------------------------------------------------------------
# Test 4: reject_by_peak_to_peak - drops high-ptp epochs
# ---------------------------------------------------------------------------

def test_reject_by_peak_to_peak_drops_high_ptp():
    n_epochs, n_ch, n_t = 5, 3, 64
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_epochs, n_ch, n_t)).astype(np.float32) * 10e-6  # small ptp
    # Force epoch 0 to have ptp=500e-6 on channel 0
    data[0, 0, :] = 0.0
    data[0, 0, 0] = 500e-6  # ptp = 500e-6
    stages = np.array([0, 1, 2, 3, 4])
    start_times = np.arange(n_epochs, dtype=np.float64)

    out_data, out_stages, out_starts = reject_by_peak_to_peak(
        data, stages, start_times, threshold_v=200e-6
    )
    assert out_data.shape[0] == n_epochs - 1, f"Expected 4 epochs, got {out_data.shape[0]}"
    assert 0 not in out_stages, "Epoch 0 (stage W) should be dropped"
    assert out_data.shape[0] == out_stages.shape[0] == out_starts.shape[0]


# ---------------------------------------------------------------------------
# Test 5: epoch_data - shape correct
# ---------------------------------------------------------------------------

def test_epoch_data_shape_correct():
    raw = _make_raw(n_channels=4, sfreq=256.0, duration=20.0)
    data, stages, start_times = epoch_data(
        raw,
        sleep_stages=np.zeros(100, dtype=int),  # dummy annotation stages
        epoch_duration=2.0,
        overlap=0.0,
    )
    # 20s / 2s = 10 epochs, 4 channels, 2s * 256 = 512 timepoints
    assert data.shape == (10, 4, 512), f"Unexpected shape: {data.shape}"


# ---------------------------------------------------------------------------
# Test 6: preprocess_subject skips if below min_epochs
# ---------------------------------------------------------------------------

def test_preprocess_subject_skips_if_below_min_epochs(monkeypatch, tmp_path):
    """Monkeypatch _load_raw to return a very short raw (< min_epochs after epoching)."""
    short_raw = _make_raw(n_channels=4, sfreq=256.0, duration=5.0)  # only ~2 epochs

    import src.data.preprocess_dream as module

    monkeypatch.setattr(module, "_load_raw", lambda path: short_raw)

    cfg = {
        "l_freq": 0.5,
        "h_freq": 40.0,
        "notch_freqs": [50.0],
        "epoch_duration": 2.0,
        "overlap": 0.0,
        "reject_threshold": 200e-6,
        "target_sfreq": 256,
        "min_epochs": 100,
    }
    dummy_path = tmp_path / "sub-01_eeg.edf"
    dummy_path.touch()
    result = preprocess_subject(dummy_path, cfg)
    assert result is None, f"Expected None for short raw, got {result}"


# ---------------------------------------------------------------------------
# Test 7: save_subject_npz roundtrip
# ---------------------------------------------------------------------------

def test_save_subject_npz_roundtrip(tmp_path):
    subject_dict = {
        "data": np.zeros((5, 4, 512), dtype=np.float32),
        "sleep_stages": np.array([0, 1, 2, 3, 4], dtype=int),
        "subject_id": "test01",
        "sfreq": 256.0,
        "ch_names": ["EEG001", "EEG002", "EEG003", "EEG004"],
        "epoch_times_s": np.linspace(0, 8, 5),
    }
    path = save_subject_npz(subject_dict, tmp_path)
    assert path.exists()
    loaded = np.load(str(path))
    for key in ["data", "sleep_stages", "subject_id", "sfreq", "ch_names", "epoch_times_s"]:
        assert key in loaded, f"Missing key: {key}"
    assert loaded["data"].dtype == np.float32


# ---------------------------------------------------------------------------
# Test 8: full preprocess_subject - output validation passes
# ---------------------------------------------------------------------------

def test_output_validation_passes_on_normalized(monkeypatch, tmp_path):
    """Run preprocess_subject on a synthetic raw with enough duration."""
    long_raw = _make_raw(n_channels=4, sfreq=256.0, duration=500.0)  # 250 epochs at 2s

    import src.data.preprocess_dream as module

    monkeypatch.setattr(module, "_load_raw", lambda path: long_raw)

    cfg = {
        "l_freq": 0.5,
        "h_freq": 40.0,
        "notch_freqs": [50.0],
        "epoch_duration": 2.0,
        "overlap": 0.0,
        "reject_threshold": 200e-6,
        "target_sfreq": 256,
        "min_epochs": 100,
    }
    dummy_path = tmp_path / "sub-42_eeg.edf"
    dummy_path.touch()
    result = preprocess_subject(dummy_path, cfg)
    assert result is not None, "Expected non-None result for long raw"
    data = result["data"]
    assert data.dtype == np.float32, f"dtype: {data.dtype}"
    assert not np.isnan(data).any(), "NaN found in data"
    assert abs(data.mean()) < 0.1, f"Mean not near zero: {data.mean()}"
    assert abs(data.std() - 1.0) < 0.2, f"Std not near 1: {data.std()}"

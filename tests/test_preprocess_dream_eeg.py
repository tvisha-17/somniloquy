"""Unit tests for src/data/preprocess_dream_eeg.py.

All tests use synthetic data — no real EDF files required.
MNE is a test dependency; tests are skipped automatically if it is absent.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

mne = pytest.importorskip("mne")

import src.data.preprocess_dream_eeg as module
from src.data.preprocess_dream_eeg import (
    parse_dream_filename,
    label_from_segment_type,
    pick_eeg_channels,
    bandpass_notch_resample,
    reject_by_peak_to_peak,
    zscore_per_channel,
    preprocess_subject_all_segments,
    save_subject_npz,
    save_subject_summary,
    discover_subject_edfs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(
    n_channels: int = 4,
    sfreq: float = 256.0,
    duration: float = 20.0,
    ch_type: str = "eeg",
) -> mne.io.RawArray:
    """Synthetic MNE RawArray with configurable channel type."""
    n_times = int(sfreq * duration)
    data = np.random.default_rng(42).standard_normal((n_channels, n_times)) * 20e-6
    ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=[ch_type] * n_channels
    )
    return mne.io.RawArray(data, info, verbose=False)


def _default_cfg() -> dict:
    return {
        "l_freq": 0.5,
        "h_freq": 40.0,
        "notch_freqs": [50.0],
        "epoch_duration": 2.0,
        "overlap": 0.0,
        "reject_threshold": 200e-6,
        "target_sfreq": 256,
        "min_epochs": 5,
        "morning_action": "include",
        "segment_labels": {
            "REM": 4,
            "NREM": 2,
            "Morning": 5,
            "SO": 6,
        },
    }


# ---------------------------------------------------------------------------
# 1. parse_dream_filename — valid patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fname, expected",
    [
        (
            "subject010_REM.edf",
            {"subject_id": "010", "segment_type": "REM", "segment_index": -1},
        ),
        (
            "subject010_NREM.edf",
            {"subject_id": "010", "segment_type": "NREM", "segment_index": -1},
        ),
        (
            "subject010_Morning.edf",
            {"subject_id": "010", "segment_type": "Morning", "segment_index": -1},
        ),
        (
            "subject010_SO1.edf",
            {"subject_id": "010", "segment_type": "SO", "segment_index": 1},
        ),
        (
            "subject010_SO10.edf",
            {"subject_id": "010", "segment_type": "SO", "segment_index": 10},
        ),
        # No extension
        (
            "subject026_REM",
            {"subject_id": "026", "segment_type": "REM", "segment_index": -1},
        ),
        # Different subject number
        (
            "subject249_SO5.edf",
            {"subject_id": "249", "segment_type": "SO", "segment_index": 5},
        ),
    ],
)
def test_parse_dream_filename_valid(fname, expected):
    result = parse_dream_filename(fname)
    assert result == expected, f"Got {result} for {fname!r}"


# ---------------------------------------------------------------------------
# 2. parse_dream_filename — unrecognised patterns return None
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fname",
    [
        "sub-010_eeg.edf",       # old sub- convention
        "sleep_recording.edf",   # generic name
        "subject_REM.edf",       # missing subject number
        "subject010.edf",        # missing segment type
        "",                      # empty string
    ],
)
def test_parse_dream_filename_invalid(fname):
    assert parse_dream_filename(fname) is None, f"Expected None for {fname!r}"


# ---------------------------------------------------------------------------
# 3. label_from_segment_type
# ---------------------------------------------------------------------------

def test_label_from_segment_type_known():
    cfg = _default_cfg()
    assert label_from_segment_type("REM", -1, cfg) == 4
    assert label_from_segment_type("NREM", -1, cfg) == 2
    assert label_from_segment_type("Morning", -1, cfg) == 5
    assert label_from_segment_type("SO", 3, cfg) == 6


def test_label_from_segment_type_unknown():
    cfg = _default_cfg()
    # unknown type should return -1 without raising
    assert label_from_segment_type("Unknown", -1, cfg) == -1


def test_label_from_segment_type_configurable():
    cfg = {**_default_cfg(), "segment_labels": {"REM": 1, "NREM": 0}}
    assert label_from_segment_type("REM", -1, cfg) == 1
    assert label_from_segment_type("NREM", -1, cfg) == 0


# ---------------------------------------------------------------------------
# 4. pick_eeg_channels — channels correctly typed
# ---------------------------------------------------------------------------

def test_pick_eeg_channels_eeg_type():
    raw = _make_raw(n_channels=4, ch_type="eeg")
    result = pick_eeg_channels(raw)
    assert len(result.ch_names) == 4


def test_pick_eeg_channels_misc_fallback():
    """When no EEG-typed channels exist, all channels are retained."""
    raw = _make_raw(n_channels=3, ch_type="misc")
    result = pick_eeg_channels(raw)
    assert len(result.ch_names) == 3   # fallback: keep all


# ---------------------------------------------------------------------------
# 5. bandpass_notch_resample — output sfreq and shape
# ---------------------------------------------------------------------------

def test_bandpass_notch_resample_changes_sfreq():
    raw = _make_raw(n_channels=2, sfreq=1000.0, duration=5.0)
    raw = bandpass_notch_resample(raw, 0.5, 40.0, [50.0], target_sfreq=256.0)
    assert raw.info["sfreq"] == 256.0
    assert raw.get_data().shape[0] == 2


def test_bandpass_notch_resample_same_sfreq_no_resample():
    raw = _make_raw(n_channels=2, sfreq=256.0, duration=5.0)
    raw = bandpass_notch_resample(raw, 0.5, 40.0, [50.0], target_sfreq=256.0)
    assert raw.info["sfreq"] == 256.0


# ---------------------------------------------------------------------------
# 6. reject_by_peak_to_peak
# ---------------------------------------------------------------------------

def test_reject_by_peak_to_peak_drops_bad_epoch():
    n_ep, n_ch, n_t = 6, 3, 64
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((n_ep, n_ch, n_t)) * 10e-6).astype(np.float32)
    # Inject artifact into epoch 2: ptp = 500 µV
    data[2, 0, :] = 0.0
    data[2, 0, 0] = 500e-6

    stages = np.arange(n_ep, dtype=np.int32)
    times = np.arange(n_ep, dtype=np.float64)
    seg_types = np.array(["REM"] * n_ep, dtype=object)
    src_files = np.array(["f.edf"] * n_ep, dtype=object)
    seg_idxs = np.full(n_ep, -1, dtype=np.int32)

    out = reject_by_peak_to_peak(
        data, stages, times, seg_types, src_files, seg_idxs, threshold_v=200e-6
    )
    out_data, out_stages, out_times, out_st, out_sf, out_si = out

    assert out_data.shape[0] == n_ep - 1, "Exactly one epoch should be dropped"
    assert 2 not in out_stages.tolist(), "Artifact epoch (stage index 2) should be removed"
    assert out_data.shape[0] == out_stages.shape[0] == out_times.shape[0]


def test_reject_by_peak_to_peak_keeps_all_clean():
    data = np.zeros((5, 2, 32), dtype=np.float32) + 1e-6
    stages = np.zeros(5, dtype=np.int32)
    times = np.arange(5, dtype=np.float64)
    seg_types = np.array(["REM"] * 5, dtype=object)
    src_files = np.array(["f.edf"] * 5, dtype=object)
    seg_idxs = np.full(5, -1, dtype=np.int32)

    out_data, *_ = reject_by_peak_to_peak(
        data, stages, times, seg_types, src_files, seg_idxs, threshold_v=200e-6
    )
    assert out_data.shape[0] == 5, "All clean epochs should be kept"


# ---------------------------------------------------------------------------
# 7. zscore_per_channel
# ---------------------------------------------------------------------------

def test_zscore_per_channel_unit_variance():
    rng = np.random.default_rng(7)
    data = (rng.standard_normal((20, 4, 100)) * 5.0).astype(np.float32)
    out = zscore_per_channel(data)
    assert out.dtype == np.float32
    assert abs(float(out.mean())) < 0.1, f"Mean not near zero: {out.mean()}"
    assert abs(float(out.std()) - 1.0) < 0.2, f"Std not near 1: {out.std()}"


def test_zscore_per_channel_no_nans():
    data = np.ones((5, 3, 50), dtype=np.float32)
    out = zscore_per_channel(data)
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# 8. preprocess_subject_all_segments — with monkeypatched _load_raw
# ---------------------------------------------------------------------------

def _make_long_raw(n_channels=4, sfreq=256.0, duration=50.0):
    return _make_raw(n_channels=n_channels, sfreq=sfreq, duration=duration)


def test_preprocess_subject_skips_below_min_epochs(monkeypatch, tmp_path):
    """Subjects with fewer than min_epochs retained epochs should return None."""
    short_raw = _make_raw(n_channels=4, sfreq=256.0, duration=3.0)  # ~1 epoch

    monkeypatch.setattr(module, "_load_raw", lambda path: short_raw)

    # Create dummy EDF path with valid name
    dummy = tmp_path / "subject001_REM.edf"
    dummy.touch()

    cfg = {**_default_cfg(), "min_epochs": 100}
    result = preprocess_subject_all_segments("001", [dummy], cfg)
    assert result is None


def test_preprocess_subject_succeeds_with_enough_data(monkeypatch, tmp_path):
    """Full pipeline should return a valid dict for a long recording."""
    long_raw = _make_long_raw(n_channels=4, sfreq=256.0, duration=300.0)

    monkeypatch.setattr(module, "_load_raw", lambda path: long_raw)

    dummy = tmp_path / "subject001_REM.edf"
    dummy.touch()

    cfg = {**_default_cfg(), "min_epochs": 5, "overlap": 0.0}
    result = preprocess_subject_all_segments("001", [dummy], cfg)

    assert result is not None
    data = result["data"]

    # dtype
    assert data.dtype == np.float32

    # 3-D shape
    assert data.ndim == 3, f"Expected ndim=3, got {data.ndim}"

    # label count matches epoch count
    assert len(result["sleep_stages"]) == data.shape[0]

    # no NaNs
    assert not np.isnan(data).any()

    # normalisation quality
    assert abs(float(data.mean())) < 0.1
    assert abs(float(data.std()) - 1.0) < 0.2

    # metadata arrays have correct length
    assert len(result["segment_types"]) == data.shape[0]
    assert len(result["source_files"]) == data.shape[0]
    assert len(result["segment_indices"]) == data.shape[0]
    assert len(result["epoch_times_s"]) == data.shape[0]


def test_preprocess_subject_labels_from_filename(monkeypatch, tmp_path):
    """Labels should be assigned from EDF filename, not from annotations."""
    rem_raw = _make_long_raw(duration=60.0)
    nrem_raw = _make_long_raw(duration=60.0)

    raw_map = {}

    def fake_load(path):
        return raw_map[str(path)]

    monkeypatch.setattr(module, "_load_raw", fake_load)

    rem_path = tmp_path / "subject001_REM.edf"
    nrem_path = tmp_path / "subject001_NREM.edf"
    rem_path.touch()
    nrem_path.touch()
    raw_map[str(rem_path)] = rem_raw
    raw_map[str(nrem_path)] = nrem_raw

    cfg = {**_default_cfg(), "min_epochs": 1, "overlap": 0.0}
    result = preprocess_subject_all_segments("001", [rem_path, nrem_path], cfg)

    assert result is not None
    stages = result["sleep_stages"]
    seg_types = result["segment_types"]

    rem_labels = stages[seg_types == "REM"]
    nrem_labels = stages[seg_types == "NREM"]

    assert (rem_labels == 4).all(), f"REM epochs should have label 4, got {np.unique(rem_labels)}"
    assert (nrem_labels == 2).all(), f"NREM epochs should have label 2, got {np.unique(nrem_labels)}"


def test_preprocess_subject_morning_exclude(monkeypatch, tmp_path):
    """morning_action=exclude should skip Morning EDF files."""
    raw = _make_long_raw(duration=60.0)
    monkeypatch.setattr(module, "_load_raw", lambda path: raw)

    rem_path = tmp_path / "subject001_REM.edf"
    morning_path = tmp_path / "subject001_Morning.edf"
    rem_path.touch()
    morning_path.touch()

    cfg = {**_default_cfg(), "morning_action": "exclude", "min_epochs": 1, "overlap": 0.0}
    result = preprocess_subject_all_segments("001", [rem_path, morning_path], cfg)

    assert result is not None
    seg_types = result["segment_types"]
    assert "Morning" not in seg_types.tolist(), "Morning epochs must not appear with morning_action=exclude"


def test_preprocess_subject_channel_intersection(monkeypatch, tmp_path):
    """When a later segment has different channels, only the intersection is kept."""
    # First segment: 4 channels EEG001–EEG004
    raw_a = _make_long_raw(n_channels=4, duration=60.0)

    # Second segment: 3 channels (EEG001–EEG003 only, missing EEG004)
    info_b = mne.create_info(
        ch_names=["EEG001", "EEG002", "EEG003"], sfreq=256.0, ch_types=["eeg"] * 3
    )
    n_t = int(256.0 * 60.0)
    raw_b = mne.io.RawArray(
        np.random.default_rng(1).standard_normal((3, n_t)) * 20e-6,
        info_b, verbose=False,
    )

    raw_map = {}

    def fake_load(path):
        return raw_map[str(path)]

    monkeypatch.setattr(module, "_load_raw", fake_load)

    path_a = tmp_path / "subject001_REM.edf"
    path_b = tmp_path / "subject001_NREM.edf"
    path_a.touch()
    path_b.touch()
    raw_map[str(path_a)] = raw_a
    raw_map[str(path_b)] = raw_b

    cfg = {**_default_cfg(), "min_epochs": 1, "overlap": 0.0}
    result = preprocess_subject_all_segments("001", [path_a, path_b], cfg)

    assert result is not None
    # All epochs must have the same channel count
    assert result["data"].shape[1] == len(result["ch_names"])
    # Channel count must be the intersection = 3
    assert result["data"].shape[1] == 3


# ---------------------------------------------------------------------------
# 9. save_subject_npz — roundtrip
# ---------------------------------------------------------------------------

def test_save_subject_npz_roundtrip(tmp_path):
    subject_dict = {
        "data": np.zeros((10, 4, 512), dtype=np.float32),
        "sleep_stages": np.array([4, 4, 2, 2, 5, 5, 6, 6, 6, 4], dtype=np.int32),
        "epoch_times_s": np.linspace(0.0, 18.0, 10),
        "segment_types": np.array(
            ["REM", "REM", "NREM", "NREM", "Morning", "Morning",
             "SO", "SO", "SO", "REM"],
            dtype=object,
        ),
        "source_files": np.array(["subject001_REM.edf"] * 10, dtype=object),
        "segment_indices": np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1, -1], dtype=np.int32),
        "subject_id": "001",
        "sfreq": 256.0,
        "ch_names": ["EEG001", "EEG002", "EEG003", "EEG004"],
        # summary fields (not saved to npz but must not break the function)
        "n_rejected": 2,
        "total_epochs_before_rejection": 12,
        "epochs_per_segment_type": {"REM": 3, "NREM": 2, "Morning": 2, "SO": 3},
        "duration_per_source_file": {"subject001_REM.edf": 18.0},
        "skipped_segments": [],
    }

    out_path = save_subject_npz(subject_dict, tmp_path)
    assert out_path.exists()

    loaded = np.load(str(out_path), allow_pickle=True)
    required_keys = [
        "data", "sleep_stages", "segment_types", "source_files",
        "segment_indices", "epoch_times_s", "subject_id", "sfreq", "ch_names",
    ]
    for k in required_keys:
        assert k in loaded, f"Missing key in .npz: {k}"

    assert loaded["data"].dtype == np.float32
    assert loaded["data"].shape == (10, 4, 512)
    assert loaded["sleep_stages"].shape == (10,)
    assert str(loaded["subject_id"]) == "001"


# ---------------------------------------------------------------------------
# 10. save_subject_summary — JSON content
# ---------------------------------------------------------------------------

def test_save_subject_summary_json_content(tmp_path):
    subject_dict = {
        "data": np.zeros((8, 4, 512), dtype=np.float32),
        "subject_id": "002",
        "sfreq": 256.0,
        "ch_names": ["EEG001", "EEG002"],
        "n_rejected": 3,
        "total_epochs_before_rejection": 11,
        "epochs_per_segment_type": {"REM": 4, "NREM": 4},
        "duration_per_source_file": {"subject002_REM.edf": 8.0},
        "skipped_segments": [],
    }

    out_path = save_subject_summary(subject_dict, tmp_path)
    assert out_path.exists()

    import json
    summary = json.loads(out_path.read_text())

    assert summary["subject_id"] == "002"
    assert summary["total_epochs"] == 8
    assert summary["epochs_rejected"] == 3
    assert "REM" in summary["epochs_per_segment_type"]


# ---------------------------------------------------------------------------
# 11. discover_subject_edfs — groups by subject
# ---------------------------------------------------------------------------

def test_discover_subject_edfs(tmp_path):
    # Create fake EDF files for two subjects
    for name in [
        "subject001_REM.edf",
        "subject001_NREM.edf",
        "subject001_Morning.edf",
        "subject002_REM.edf",
        "subject002_SO1.edf",
        "not_a_dream_file.edf",   # should be skipped
    ]:
        (tmp_path / name).touch()

    mapping = discover_subject_edfs(tmp_path)

    assert "001" in mapping
    assert "002" in mapping
    assert len(mapping["001"]) == 3
    assert len(mapping["002"]) == 2

    # Unrecognised file should not appear
    all_names = [p.name for paths in mapping.values() for p in paths]
    assert "not_a_dream_file.edf" not in all_names


# ---------------------------------------------------------------------------
# 12. Integration: full pipeline on two-segment synthetic subject
# ---------------------------------------------------------------------------

def test_integration_two_segments(monkeypatch, tmp_path):
    """End-to-end: two segments → concatenated, normalised .npz."""
    raw_rem = _make_long_raw(duration=200.0)
    raw_so = _make_long_raw(duration=100.0)

    raw_map = {}

    def fake_load(path):
        return raw_map[str(path)]

    monkeypatch.setattr(module, "_load_raw", fake_load)

    rem_path = tmp_path / "subject099_REM.edf"
    so_path = tmp_path / "subject099_SO3.edf"
    rem_path.touch()
    so_path.touch()
    raw_map[str(rem_path)] = raw_rem
    raw_map[str(so_path)] = raw_so

    cfg = {**_default_cfg(), "min_epochs": 5, "overlap": 0.0}
    result = preprocess_subject_all_segments("099", [rem_path, so_path], cfg)

    assert result is not None

    npz_path = save_subject_npz(result, tmp_path)
    loaded = np.load(str(npz_path), allow_pickle=True)

    # Validate .npz correctness
    data = loaded["data"]
    stages = loaded["sleep_stages"]
    seg_types = loaded["segment_types"]

    assert data.dtype == np.float32
    assert not np.isnan(data).any()
    assert data.shape[0] == stages.shape[0] == seg_types.shape[0]

    present_types = set(seg_types.tolist())
    assert "REM" in present_types
    assert "SO" in present_types

    rem_stages = stages[seg_types == "REM"]
    so_stages = stages[seg_types == "SO"]
    assert (rem_stages == 4).all()
    assert (so_stages == 6).all()

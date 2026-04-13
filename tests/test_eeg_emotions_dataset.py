"""Tests for EEG emotion dataset utilities."""

from pathlib import Path

import numpy as np
import scipy.io


def _write_clip(path: Path, shape=(6, 6400)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, {"Data": np.random.randn(*shape).astype(np.float32)})


def _base_config(tmp_path: Path) -> dict:
    return {
        "data_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "window_size_samples": 3200,
        "stride_size_samples": 3200,
        "target_sfreq": 200,
        "normalization_mode": "per_sample",
        "use_subject_split": True,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "random_seed": 13,
    }


def test_eeg_emotions_dataset_returns_expected_shapes(tmp_path):
    from src.data.eeg_emotions_dataset import EEGEmotionDataset, build_or_load_splits, inspect_and_cache_eeg_emotions

    raw_dir = tmp_path / "raw"
    _write_clip(raw_dir / "G_S0001_M1_E2_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0002_M1_E3_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0003_M1_E4_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0004_M1_E2_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0005_M1_E3_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0006_M1_E4_R1_N1_raw_ref.mat")

    config = _base_config(tmp_path)
    inspect_and_cache_eeg_emotions(config)
    splits = build_or_load_splits(config)
    dataset = EEGEmotionDataset(splits["train"], normalization_mode="per_sample")

    x, y, subject_id = dataset[0]
    assert tuple(x.shape) == (6, 3200)
    assert y in {0, 1, 2}
    assert subject_id.startswith("S")


def test_eeg_emotions_dataset_grouped_labels(tmp_path):
    from src.data.eeg_emotions_dataset import inspect_and_cache_eeg_emotions, load_cached_index

    raw_dir = tmp_path / "raw"
    _write_clip(raw_dir / "G_S0001_M1_E1_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0002_M1_E2_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0003_M1_E3_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0004_M1_E4_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0005_M1_E5_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0006_M1_E0_R1_N1_raw_ref.mat")

    config = _base_config(tmp_path)
    config["emotion_label_groups"] = {
        "negative": [1, 2],
        "neutral": [3],
        "positive": [4, 5],
    }
    inspect_and_cache_eeg_emotions(config)
    index = load_cached_index(tmp_path / "processed")

    assert index["label_names"] == ["negative", "neutral", "positive"]
    assert index["label_map"] == {"1": 0, "2": 0, "3": 1, "4": 2, "5": 2}
    assert all(sample["raw_emotion_code"] != 0 for sample in index["samples"])


def test_eeg_emotions_dataset_per_recording_normalization_runs(tmp_path):
    from src.data.eeg_emotions_dataset import EEGEmotionDataset, build_or_load_splits, inspect_and_cache_eeg_emotions

    raw_dir = tmp_path / "raw"
    _write_clip(raw_dir / "G_S0001_M1_E2_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0002_M1_E3_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0003_M1_E4_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0004_M1_E2_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0005_M1_E3_R1_N1_raw_ref.mat")
    _write_clip(raw_dir / "G_S0006_M1_E4_R1_N1_raw_ref.mat")

    config = _base_config(tmp_path)
    inspect_and_cache_eeg_emotions(config)
    splits = build_or_load_splits(config)
    dataset = EEGEmotionDataset(splits["train"], normalization_mode="per_recording")

    x, y, _subject_id = dataset[0]
    assert tuple(x.shape) == (6, 3200)
    assert np.isfinite(x.numpy()).all()
    assert y in {0, 1, 2}

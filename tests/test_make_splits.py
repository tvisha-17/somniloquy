"""Tests for src/data/make_splits.py — subject-level train/val/test split generator."""

import json
from pathlib import Path

import pytest


def test_make_splits_disjoint():
    from src.data.make_splits import make_splits

    subjects = [str(i).zfill(2) for i in range(1, 11)]
    splits = make_splits(subjects, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_make_splits_deterministic():
    from src.data.make_splits import make_splits

    subjects = [str(i).zfill(2) for i in range(1, 11)]
    s1 = make_splits(subjects, 0.7, 0.15, 0.15, seed=42)
    s2 = make_splits(subjects, 0.7, 0.15, 0.15, seed=42)
    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["test"] == s2["test"]


def test_make_splits_coverage():
    from src.data.make_splits import make_splits

    subjects = [str(i).zfill(2) for i in range(1, 11)]
    splits = make_splits(subjects, 0.7, 0.15, 0.15, seed=42)
    all_subjects = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert all_subjects == set(subjects)


def test_make_splits_ratios():
    from src.data.make_splits import make_splits

    subjects = [str(i).zfill(2) for i in range(1, 101)]
    splits = make_splits(subjects, 0.7, 0.15, 0.15, seed=42)
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15


def test_make_splits_invalid_ratios_raises():
    from src.data.make_splits import make_splits

    subjects = [str(i).zfill(2) for i in range(1, 11)]
    with pytest.raises(ValueError):
        make_splits(subjects, 0.5, 0.3, 0.3, seed=42)


def test_save_splits_writes_valid_json(tmp_path):
    from src.data.make_splits import make_splits, save_splits

    subjects = [str(i).zfill(2) for i in range(1, 11)]
    splits = make_splits(subjects, 0.7, 0.15, 0.15, seed=42)
    out_path = tmp_path / "dream_splits.json"
    save_splits(splits, out_path)

    with out_path.open() as f:
        loaded = json.load(f)

    assert "train" in loaded
    assert "val" in loaded
    assert "test" in loaded
    assert isinstance(loaded["train"], list)
    assert isinstance(loaded["val"], list)
    assert isinstance(loaded["test"], list)


def test_discover_subject_ids_filters_correctly(tmp_path):
    from src.data.make_splits import discover_subject_ids

    (tmp_path / "sub-01_epochs.npz").touch()
    (tmp_path / "sub-02_epochs.npz").touch()
    (tmp_path / "sub-01_target_embeddings.npz").touch()
    (tmp_path / "foo.txt").touch()

    result = discover_subject_ids(tmp_path)
    assert result == ["01", "02"], f"Got {result}"

"""Tests for EEG emotion model/training pipeline."""

from pathlib import Path

import numpy as np
import scipy.io
import torch


def _write_clip(path: Path, shape=(6, 6400)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, {"Data": np.random.randn(*shape).astype(np.float32)})


def _config(tmp_path: Path) -> dict:
    return {
        "data_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "n_epochs": 1,
        "freeze_backbone": True,
        "normalization_mode": "per_sample",
        "use_subject_split": True,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "random_seed": 13,
        "num_workers": 0,
        "device": "cpu",
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "dropout": 0.0,
        "latent_dim": 16,
        "backbone_mode": "cnn",
        "zuna_model_name": "Zyphra/ZUNA",
        "num_classes": 3,
        "window_size_samples": 3200,
        "stride_size_samples": 3200,
        "target_sfreq": 200,
        "emotion_label_groups": {
            "negative": [1, 2],
            "neutral": [3],
            "positive": [4, 5],
        },
        "use_weighted_sampler": False,
        "use_class_weights": True,
        "label_smoothing": 0.05,
        "use_augmentation": True,
        "channel_dropout_prob": 0.1,
        "time_mask_prob": 0.2,
        "time_mask_fraction": 0.1,
        "amplitude_jitter_std": 0.01,
        "early_stopping_patience": 2,
        "permutation_trials": 10,
    }


def _populate_dataset(raw_dir: Path) -> None:
    filenames = [
        "G_S0001_M1_E1_R1_N1_raw_ref.mat",
        "G_S0002_M1_E2_R1_N1_raw_ref.mat",
        "G_S0003_M1_E3_R1_N1_raw_ref.mat",
        "G_S0004_M1_E4_R1_N1_raw_ref.mat",
        "G_S0005_M1_E5_R1_N1_raw_ref.mat",
        "G_S0006_M1_E3_R1_N1_raw_ref.mat",
    ]
    for name in filenames:
        _write_clip(raw_dir / name)


def test_eeg_emotion_classifier_output_shape():
    from src.models.eeg_emotion_classifier import EEGEmotionClassifier

    model = EEGEmotionClassifier(ch_names=[f"EEG{i}" for i in range(6)], latent_dim=16, dropout=0.0)
    out = model(torch.randn(2, 6, 3200))
    assert tuple(out.shape) == (2, 3)


def test_tiny_eeg_emotions_train_and_eval_runs(tmp_path):
    from src.data.eeg_emotions_dataset import inspect_and_cache_eeg_emotions
    from src.evaluation.evaluate_eeg_emotions import evaluate_eeg_emotions
    from src.training.train_eeg_emotions import train_eeg_emotions

    config = _config(tmp_path)
    _populate_dataset(tmp_path / "raw")
    inspect_and_cache_eeg_emotions(config)
    train_result = train_eeg_emotions(config)
    eval_config = dict(config)
    eval_config["checkpoint_path"] = train_result["best_checkpoint_path"]
    metrics = evaluate_eeg_emotions(eval_config)

    assert Path(train_result["best_checkpoint_path"]).exists()
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

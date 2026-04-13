"""Tests for src/training/finetune_zuna.py."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


class DummyBackbone(nn.Module):
    """Small train-test backbone with a frozen latent projection."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(4 * 8, latent_dim)

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        return self.proj(flat)


def _write_subject_files(
    root: Path,
    subject_id: str,
    sleep_stages,
    epoch_indices,
    *,
    target_scale: float = 1.0,
):
    eeg_dir = root / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    target_dir = root / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = len(sleep_stages)
    eeg = np.arange(n_epochs * 4 * 8, dtype=np.float32).reshape(n_epochs, 4, 8) * 0.001
    targets = np.full((len(epoch_indices), 384), target_scale, dtype=np.float32)

    np.savez(
        eeg_dir / f"sub-{subject_id}_epochs.npz",
        data=eeg,
        sleep_stages=np.array(sleep_stages, dtype=np.int64),
        subject_id=subject_id,
        sfreq=256.0,
        ch_names=np.array(["C1", "C2", "C3", "C4"]),
        epoch_times_s=np.arange(n_epochs, dtype=np.float64) * 2.0,
    )
    np.savez(
        target_dir / f"sub-{subject_id}_target_embeddings.npz",
        epoch_indices=np.array(epoch_indices, dtype=np.int64),
        target_embeddings=targets,
        report_text=np.array(f"dream report {subject_id}"),
    )


def _write_split_file(path: Path, train_ids, val_ids):
    payload = {"train": list(train_ids), "val": list(val_ids), "test": []}
    path.write_text(json.dumps(payload))


def _base_config(root: Path):
    return {
        "eeg_epochs_dir": str(root / "eeg"),
        "target_embeddings_dir": str(root / "targets"),
        "split_file": str(root / "dream_splits.json"),
        "zuna_model_name": "Zyphra/ZUNA",
        "latent_dim": 16,
        "target_embed_dim": 384,
        "dropout": 0.0,
        "cosine_weight": 0.7,
        "mse_weight": 0.3,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "n_epochs": 2,
        "warmup_steps": 0,
        "grad_clip": 1.0,
        "device": "cpu",
        "checkpoint_dir": str(root / "checkpoints"),
        "log_every_n_steps": 1,
        "val_every_n_epochs": 1,
    }


def test_create_dataset_for_split_filters_non_rem_epochs(tmp_path):
    from src.training.finetune_zuna import create_dataset_for_split

    _write_subject_files(tmp_path, "01", sleep_stages=[4, 1, 4], epoch_indices=[0, 1, 2])
    _write_split_file(tmp_path / "dream_splits.json", train_ids=["01"], val_ids=["01"])

    dataset = create_dataset_for_split(
        split_name="train",
        split_file=tmp_path / "dream_splits.json",
        eeg_epochs_dir=tmp_path / "eeg",
        target_embeddings_dir=tmp_path / "targets",
    )

    assert len(dataset) == 2
    eeg_item, target_item = dataset[0]
    assert tuple(eeg_item.shape) == (4, 8)
    assert tuple(target_item.shape) == (384,)


def test_train_model_saves_checkpoint_and_keeps_backbone_frozen(tmp_path):
    from src.models.zuna_decoder import ZUNAForSpeechDecoding
    from src.training.finetune_zuna import train_model

    _write_subject_files(tmp_path, "01", sleep_stages=[4, 4, 4, 4], epoch_indices=[0, 1, 2, 3], target_scale=0.5)
    _write_subject_files(tmp_path, "02", sleep_stages=[4, 4], epoch_indices=[0, 1], target_scale=0.25)
    _write_split_file(tmp_path / "dream_splits.json", train_ids=["01"], val_ids=["02"])

    model = ZUNAForSpeechDecoding(
        target_embed_dim=384,
        dropout=0.0,
        latent_dim=16,
        backbone=DummyBackbone(latent_dim=16),
    )
    result = train_model(_base_config(tmp_path), model=model)

    checkpoint_path = Path(result["best_checkpoint_path"])
    assert checkpoint_path.exists()
    assert result["best_val_cosine_similarity"] > float("-inf")
    assert result["best_epoch"] in {1, 2}
    assert all(param.grad is None for param in model.backbone.parameters())


def test_train_model_logs_optional_retrieval_metrics(tmp_path):
    from src.models.zuna_decoder import ZUNAForSpeechDecoding
    from src.training.finetune_zuna import train_model

    _write_subject_files(tmp_path, "01", sleep_stages=[4, 4, 4], epoch_indices=[0, 1, 2], target_scale=0.5)
    _write_subject_files(tmp_path, "02", sleep_stages=[4, 4], epoch_indices=[0, 1], target_scale=0.25)
    _write_split_file(tmp_path / "dream_splits.json", train_ids=["01"], val_ids=["02"])

    config = _base_config(tmp_path)
    config["retrieval_bank_dir"] = str(tmp_path / "targets")
    model = ZUNAForSpeechDecoding(
        target_embed_dim=384,
        dropout=0.0,
        latent_dim=16,
        backbone=DummyBackbone(latent_dim=16),
    )

    result = train_model(config, model=model)

    assert "val_top1" in result["history"]
    assert "val_top5" in result["history"]
    assert "val_top10" in result["history"]
    assert "val_mrr" in result["history"]
    assert len(result["history"]["val_top1"]) >= 1


class _NaNGradientLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred):
        ctx.shape = pred.shape
        return pred.sum() * 0.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(ctx.shape, float("nan"), device=grad_output.device)


def test_train_model_raises_on_non_finite_gradients(tmp_path, monkeypatch):
    from src.models.zuna_decoder import ZUNAForSpeechDecoding
    import src.training.finetune_zuna as finetune_module

    _write_subject_files(tmp_path, "01", sleep_stages=[4, 4], epoch_indices=[0, 1], target_scale=0.5)
    _write_subject_files(tmp_path, "02", sleep_stages=[4, 4], epoch_indices=[0, 1], target_scale=0.25)
    _write_split_file(tmp_path / "dream_splits.json", train_ids=["01"], val_ids=["02"])

    def fake_loss(pred_emb, target_emb, cosine_weight=0.7, mse_weight=0.3):
        return _NaNGradientLoss.apply(pred_emb)

    monkeypatch.setattr(finetune_module, "contrastive_mse_loss", fake_loss)

    model = ZUNAForSpeechDecoding(
        target_embed_dim=384,
        dropout=0.0,
        latent_dim=16,
        backbone=DummyBackbone(latent_dim=16),
    )

    with pytest.raises(RuntimeError, match="Non-finite gradient"):
        finetune_module.train_model(_base_config(tmp_path), model=model)

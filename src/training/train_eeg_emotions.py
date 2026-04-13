"""Training loop for EEG emotion classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.eeg_emotions_dataset import (
    EEGEmotionDataset,
    build_or_load_splits,
    build_weighted_sampler,
    compute_class_weights,
    inspect_and_cache_eeg_emotions,
    load_cached_index,
)
from src.models.eeg_emotion_classifier import EEGEmotionClassifier
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )


def _build_dataloaders(config: Dict[str, object]) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    processed_dir = Path(str(config["processed_dir"]))
    if bool(config.get("force_rebuild_cache", False)) or not (processed_dir / "index.json").exists():
        inspect_and_cache_eeg_emotions(config)
    index_payload = load_cached_index(processed_dir)
    split_payload = build_or_load_splits(config)

    train_dataset = EEGEmotionDataset(
        split_payload["train"],
        normalization_mode=str(config["normalization_mode"]),
        augment=bool(config.get("use_augmentation", False)),
        channel_dropout_prob=float(config.get("channel_dropout_prob", 0.0)),
        time_mask_prob=float(config.get("time_mask_prob", 0.0)),
        time_mask_fraction=float(config.get("time_mask_fraction", 0.1)),
        amplitude_jitter_std=float(config.get("amplitude_jitter_std", 0.0)),
        random_seed=int(config.get("random_seed", 13)),
    )
    val_dataset = EEGEmotionDataset(split_payload["val"], normalization_mode=str(config["normalization_mode"]))
    test_dataset = EEGEmotionDataset(split_payload["test"], normalization_mode=str(config["normalization_mode"]))

    sampler = None
    shuffle = True
    if bool(config.get("use_weighted_sampler", True)):
        sampler = build_weighted_sampler(train_dataset.labels, num_classes=int(config.get("num_classes", 3)))
        shuffle = False

    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=int(config.get("num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(config.get("num_workers", 0)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(config.get("num_workers", 0)))
    return train_loader, val_loader, test_loader, list(index_payload.get("label_names", [f"E{code}" for code in index_payload["selected_emotion_codes"]]))


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: Optional[AdamW] = None,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []
    all_logits = []
    all_labels = []

    for step_index, (eeg_batch, label_batch, _subject_ids) in enumerate(dataloader, start=1):
        eeg_batch = eeg_batch.to(device)
        label_batch = label_batch.to(device)
        if step_index == 1 or step_index % 50 == 0:
            logger.info(
                "train_eeg_emotions step=%d eeg_batch_shape=%s label_batch_shape=%s",
                step_index,
                tuple(eeg_batch.shape),
                tuple(label_batch.shape),
            )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(eeg_batch)
        loss = _classification_loss(
            logits,
            label_batch,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
        all_logits.append(logits.detach().cpu())
        all_labels.append(label_batch.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "loss": float(np.mean(losses)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def train_eeg_emotions(config: Dict[str, object]) -> Dict[str, object]:
    device = _resolve_device(str(config.get("device", "cpu")))
    train_loader, val_loader, test_loader, label_names = _build_dataloaders(config)
    class_weights = None
    if bool(config.get("use_class_weights", False)):
        class_weights = compute_class_weights(train_loader.dataset.labels, num_classes=int(config.get("num_classes", 3))).to(device)
        logger.info("train_eeg_emotions class_weights=%s", class_weights.detach().cpu().tolist())
    label_smoothing = float(config.get("label_smoothing", 0.0))

    ch_names = [f"EEG{i+1}" for i in range(train_loader.dataset[0][0].shape[0])]
    model = EEGEmotionClassifier(
        ch_names=ch_names,
        num_classes=int(config.get("num_classes", 3)),
        dropout=float(config.get("dropout", 0.3)),
        latent_dim=int(config["latent_dim"]) if config.get("latent_dim") is not None else None,
        backbone_mode=str(config.get("backbone_mode", "cnn")),
        zuna_model_name=str(config.get("zuna_model_name", "Zyphra/ZUNA")),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    ).to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
    }
    best_metric = float("-inf")
    best_epoch = None
    patience = int(config.get("early_stopping_patience", 5))
    epochs_without_improvement = 0
    checkpoint_dir = Path(str(config["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"

    for epoch_index in range(1, int(config["n_epochs"]) + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            class_weights=class_weights,
            label_smoothing=0.0,
        )
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_balanced_accuracy=%.6f val_macro_f1=%.6f",
            epoch_index,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["balanced_accuracy"],
            val_metrics["macro_f1"],
        )
        if val_metrics["macro_f1"] > best_metric:
            best_metric = val_metrics["macro_f1"]
            best_epoch = epoch_index
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": dict(config),
                    "label_names": label_names,
                    "best_epoch": best_epoch,
                    "val_macro_f1": best_metric,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch=%d", epoch_index)
                break

    split_payload = build_or_load_splits(config)
    summary = {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "best_checkpoint_path": str(best_checkpoint_path),
        "label_names": label_names,
        "split_mode": split_payload["mode"],
        "n_train": len(split_payload["train"]),
        "n_val": len(split_payload["val"]),
        "n_test": len(split_payload["test"]),
    }
    (checkpoint_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

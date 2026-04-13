"""Evaluation utilities for EEG emotion classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data.eeg_emotions_dataset import EEGEmotionDataset, build_or_load_splits
from src.models.eeg_emotion_classifier import EEGEmotionClassifier
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for eeg_batch, label_batch, _subject_ids in dataloader:
            eeg_batch = eeg_batch.to(device)
            logits = model(eeg_batch)
            all_logits.append(logits.cpu())
            all_labels.append(label_batch.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return logits.argmax(axis=1), labels


def _permutation_p_value(metric_fn, y_true: np.ndarray, y_pred: np.ndarray, n_perm: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    actual = float(metric_fn(y_true, y_pred))
    null_values = np.asarray([metric_fn(rng.permutation(y_true), y_pred) for _ in range(n_perm)], dtype=float)
    return {
        "actual": actual,
        "null_mean": float(null_values.mean()),
        "null_p95": float(np.percentile(null_values, 95)),
        "empirical_p_value": float((1 + np.sum(null_values >= actual)) / (1 + n_perm)),
    }


def evaluate_eeg_emotions(config: Dict[str, object]) -> Dict[str, object]:
    checkpoint_path = Path(str(config["checkpoint_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    split_payload = build_or_load_splits(config)
    test_dataset = EEGEmotionDataset(split_payload["test"], normalization_mode=str(config["normalization_mode"]))
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)))
    device = torch.device(str(config.get("device", "cpu")))
    ch_names = [f"EEG{i+1}" for i in range(test_dataset[0][0].shape[0])]
    model = EEGEmotionClassifier(
        ch_names=ch_names,
        num_classes=int(config.get("num_classes", 3)),
        dropout=float(config.get("dropout", 0.3)),
        latent_dim=int(config["latent_dim"]) if config.get("latent_dim") is not None else None,
        backbone_mode=str(config.get("backbone_mode", "cnn")),
        zuna_model_name=str(config.get("zuna_model_name", "Zyphra/ZUNA")),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    preds, labels = _predict(model, test_loader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(int(config.get("num_classes", 3)))))

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "split_mode": split_payload["mode"],
        "label_names": checkpoint.get("label_names", ["class0", "class1", "class2"]),
    }
    metrics["permutation_balanced_accuracy"] = _permutation_p_value(
        balanced_accuracy_score,
        labels,
        preds,
        n_perm=int(config.get("permutation_trials", 1000)),
        seed=int(config.get("random_seed", 13)),
    )
    metrics["permutation_macro_f1"] = _permutation_p_value(
        lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2],
        labels,
        preds,
        n_perm=int(config.get("permutation_trials", 1000)),
        seed=int(config.get("random_seed", 13)) + 1,
    )

    output_dir = Path(str(config["checkpoint_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(metrics["label_names"])), labels=metrics["label_names"], rotation=45, ha="right")
    ax.set_yticks(range(len(metrics["label_names"])), labels=metrics["label_names"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    logger.info("evaluate_eeg_emotions metrics=%s", metrics)
    return metrics

"""Fine-tune the Phase 2 speech-decoding head on REM-aligned DREAM epochs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.models.zuna_decoder import ZUNAForSpeechDecoding
from src.utils.logging import get_logger

logger = get_logger(__name__)


class REMAlignedEEGDataset(Dataset):
    """In-memory dataset of REM EEG windows aligned to semantic targets."""

    def __init__(self, eeg: np.ndarray, targets: np.ndarray, subject_ids: Sequence[str]):
        if eeg.ndim != 3:
            raise ValueError(f"Expected eeg shape (n, c, t), got {eeg.shape}")
        if targets.ndim != 2:
            raise ValueError(f"Expected target shape (n, d), got {targets.shape}")
        if eeg.shape[0] != targets.shape[0]:
            raise ValueError(f"Sample count mismatch: eeg={eeg.shape[0]} targets={targets.shape[0]}")
        if eeg.shape[0] != len(subject_ids):
            raise ValueError(f"Subject count mismatch: n={eeg.shape[0]} subject_ids={len(subject_ids)}")

        self.eeg = torch.from_numpy(eeg.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.subject_ids = list(subject_ids)
        logger.info(
            "REMAlignedEEGDataset eeg_shape=%s target_shape=%s n_subjects=%d",
            tuple(self.eeg.shape),
            tuple(self.targets.shape),
            len(set(self.subject_ids)),
        )

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.eeg[index], self.targets[index]


def load_split_subject_ids(split_file: Path, split_name: str) -> List[str]:
    """Load subject IDs for a named split from the split JSON file."""
    with Path(split_file).open() as handle:
        payload = json.load(handle)
    if split_name not in payload:
        raise KeyError(f"Split file {split_file} is missing key {split_name!r}")
    subject_ids = list(payload[split_name])
    logger.info("load_split_subject_ids split=%s n_subjects=%d", split_name, len(subject_ids))
    return subject_ids


def load_aligned_rem_examples(
    subject_ids: Sequence[str],
    eeg_epochs_dir: Path,
    target_embeddings_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load REM-only EEG examples and semantic targets for the given subjects."""
    all_eeg = []
    all_targets = []
    all_subject_refs: List[str] = []

    eeg_epochs_dir = Path(eeg_epochs_dir)
    target_embeddings_dir = Path(target_embeddings_dir)

    for subject_id in subject_ids:
        eeg_path = eeg_epochs_dir / f"sub-{subject_id}_epochs.npz"
        target_path = target_embeddings_dir / f"sub-{subject_id}_target_embeddings.npz"

        if not eeg_path.exists() or not target_path.exists():
            logger.warning(
                "Skipping subject %s because required files are missing: eeg=%s target=%s",
                subject_id,
                eeg_path.exists(),
                target_path.exists(),
            )
            continue

        eeg_npz = np.load(str(eeg_path), allow_pickle=True)
        target_npz = np.load(str(target_path), allow_pickle=True)

        eeg = eeg_npz["data"].astype(np.float32)
        sleep_stages = eeg_npz["sleep_stages"].astype(np.int64)
        epoch_indices = target_npz["epoch_indices"].astype(np.int64)
        target_embeddings = target_npz["target_embeddings"].astype(np.float32)

        if epoch_indices.shape[0] != target_embeddings.shape[0]:
            raise ValueError(
                f"Subject {subject_id} has mismatched alignment arrays: "
                f"epoch_indices={epoch_indices.shape[0]} target_embeddings={target_embeddings.shape[0]}"
            )

        valid_mask = (epoch_indices >= 0) & (epoch_indices < eeg.shape[0])
        if not np.all(valid_mask):
            logger.warning(
                "Subject %s has %d out-of-range epoch indices; dropping them.",
                subject_id,
                int((~valid_mask).sum()),
            )
            epoch_indices = epoch_indices[valid_mask]
            target_embeddings = target_embeddings[valid_mask]

        rem_mask = sleep_stages[epoch_indices] == 4
        if not np.any(rem_mask):
            logger.warning("Subject %s has no REM-aligned epochs after filtering; skipping.", subject_id)
            continue

        selected_indices = epoch_indices[rem_mask]
        selected_eeg = eeg[selected_indices]
        selected_targets = target_embeddings[rem_mask]
        logger.info(
            "load_aligned_rem_examples subject=%s eeg_shape=%s target_shape=%s",
            subject_id,
            tuple(selected_eeg.shape),
            tuple(selected_targets.shape),
        )

        all_eeg.append(selected_eeg)
        all_targets.append(selected_targets)
        all_subject_refs.extend([subject_id] * selected_eeg.shape[0])

    if not all_eeg:
        raise ValueError("No REM-aligned examples were found for the requested split.")

    eeg_array = np.concatenate(all_eeg, axis=0).astype(np.float32)
    target_array = np.concatenate(all_targets, axis=0).astype(np.float32)
    logger.info(
        "load_aligned_rem_examples total_eeg_shape=%s total_target_shape=%s",
        tuple(eeg_array.shape),
        tuple(target_array.shape),
    )
    return eeg_array, target_array, all_subject_refs


def create_dataset_for_split(
    split_name: str,
    split_file: Path,
    eeg_epochs_dir: Path,
    target_embeddings_dir: Path,
) -> REMAlignedEEGDataset:
    """Create a dataset for a named split."""
    subject_ids = load_split_subject_ids(split_file, split_name)
    if not subject_ids:
        raise ValueError(f"Split {split_name!r} is empty in {split_file}")
    eeg, targets, subject_refs = load_aligned_rem_examples(
        subject_ids=subject_ids,
        eeg_epochs_dir=eeg_epochs_dir,
        target_embeddings_dir=target_embeddings_dir,
    )
    return REMAlignedEEGDataset(eeg=eeg, targets=targets, subject_ids=subject_refs)


def create_dataloaders(config: Dict[str, object]) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders from config."""
    train_dataset = create_dataset_for_split(
        split_name="train",
        split_file=Path(config["split_file"]),
        eeg_epochs_dir=Path(config["eeg_epochs_dir"]),
        target_embeddings_dir=Path(config["target_embeddings_dir"]),
    )
    val_dataset = create_dataset_for_split(
        split_name="val",
        split_file=Path(config["split_file"]),
        eeg_epochs_dir=Path(config["eeg_epochs_dir"]),
        target_embeddings_dir=Path(config["target_embeddings_dir"]),
    )
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def contrastive_mse_loss(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    cosine_weight: float = 0.7,
    mse_weight: float = 0.3,
) -> torch.Tensor:
    """Combined cosine-distance and MSE loss."""
    pred_norm = F.normalize(pred_emb, dim=-1)
    target_norm = F.normalize(target_emb, dim=-1)
    cosine_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()
    mse_loss = F.mse_loss(pred_emb, target_emb)
    return cosine_weight * cosine_loss + mse_weight * mse_loss


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Return mean cosine similarity on the validation split."""
    similarities = []
    model.eval()
    with torch.no_grad():
        for eeg_batch, target_batch in dataloader:
            eeg_batch = eeg_batch.to(device)
            target_batch = target_batch.to(device)
            logger.info(
                "evaluate_model eeg_batch_shape=%s target_batch_shape=%s",
                tuple(eeg_batch.shape),
                tuple(target_batch.shape),
            )
            pred_batch = model(eeg_batch)
            similarities.append(F.cosine_similarity(pred_batch, target_batch, dim=-1).cpu())

    if not similarities:
        raise ValueError("Validation dataloader produced no batches.")

    merged = torch.cat(similarities)
    return float(merged.mean().item())


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def _build_warmup_scheduler(optimizer: AdamW, warmup_steps: int) -> Optional[LambdaLR]:
    if warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / float(warmup_steps))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _assert_backbone_frozen(model: nn.Module) -> None:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    for name, param in backbone.named_parameters():
        if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad)):
            raise RuntimeError(f"Frozen backbone parameter received gradient: {name}")


def _ensure_finite_gradients(model: nn.Module, epoch_index: int, step_index: int) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(
                f"Non-finite gradient detected at epoch={epoch_index} step={step_index} parameter={name}"
            )


def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch_index: int,
    val_cosine_similarity: float,
    config: Dict[str, object],
) -> Path:
    """Save the current best checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    payload = {
        "epoch": epoch_index,
        "val_cosine_similarity": val_cosine_similarity,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(config),
    }
    torch.save(payload, checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return checkpoint_path


def train_model(config: Dict[str, object], model: Optional[nn.Module] = None) -> Dict[str, object]:
    """Train the speech-decoding head and return training history."""
    device = _resolve_device(str(config["device"]))
    train_loader, val_loader = create_dataloaders(config)

    if model is None:
        model = ZUNAForSpeechDecoding(
            zuna_model_name=str(config["zuna_model_name"]),
            target_embed_dim=int(config["target_embed_dim"]),
            dropout=float(config["dropout"]),
            latent_dim=int(config["latent_dim"]) if config.get("latent_dim") is not None else None,
        )

    model = model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = _build_warmup_scheduler(optimizer, warmup_steps=int(config["warmup_steps"]))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_cosine_similarity": [],
    }
    best_val = float("-inf")
    best_checkpoint_path: Optional[Path] = None
    global_step = 0

    for epoch_index in range(1, int(config["n_epochs"]) + 1):
        model.train()
        epoch_losses = []

        for step_index, (eeg_batch, target_batch) in enumerate(train_loader, start=1):
            eeg_batch = eeg_batch.to(device)
            target_batch = target_batch.to(device)
            logger.info(
                "train_model eeg_batch_shape=%s target_batch_shape=%s",
                tuple(eeg_batch.shape),
                tuple(target_batch.shape),
            )

            optimizer.zero_grad(set_to_none=True)
            pred_batch = model(eeg_batch)
            loss = contrastive_mse_loss(
                pred_batch,
                target_batch,
                cosine_weight=float(config["cosine_weight"]),
                mse_weight=float(config["mse_weight"]),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch={epoch_index} step={step_index}")

            loss.backward()
            _ensure_finite_gradients(model, epoch_index, step_index)
            _assert_backbone_frozen(model)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(config["grad_clip"]))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            epoch_losses.append(float(loss.item()))

            if global_step % int(config["log_every_n_steps"]) == 0:
                logger.info(
                    "epoch=%d step=%d global_step=%d loss=%.6f",
                    epoch_index,
                    step_index,
                    global_step,
                    float(loss.item()),
                )

        if not epoch_losses:
            raise ValueError("Training dataloader produced no batches.")

        history["train_loss"].append(float(np.mean(epoch_losses)))

        should_validate = (
            epoch_index % int(config["val_every_n_epochs"]) == 0
            or epoch_index == int(config["n_epochs"])
        )
        if should_validate:
            val_cosine_similarity = evaluate_model(model, val_loader, device)
            history["val_cosine_similarity"].append(val_cosine_similarity)
            logger.info(
                "epoch=%d train_loss=%.6f val_cosine_similarity=%.6f",
                epoch_index,
                history["train_loss"][-1],
                val_cosine_similarity,
            )
            if val_cosine_similarity > best_val:
                best_val = val_cosine_similarity
                best_checkpoint_path = save_checkpoint(
                    checkpoint_dir=Path(config["checkpoint_dir"]),
                    model=model,
                    optimizer=optimizer,
                    epoch_index=epoch_index,
                    val_cosine_similarity=val_cosine_similarity,
                    config=config,
                )

    return {
        "history": history,
        "best_val_cosine_similarity": best_val,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
    }

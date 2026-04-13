"""Dataset utilities for emotion-labeled dream EEG clips."""

from __future__ import annotations

import json
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import scipy.io
from scipy.signal import resample_poly
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.inspect_eeg_emotions import inspect_eeg_emotions_dataset, parse_filename_metadata
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EmotionSample:
    sample_id: str
    processed_path: str
    window_index: int
    label: int
    raw_emotion_code: int
    subject_id: str
    night: int
    report_index: int
    stage: str


def _zscore(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    std = np.where(std > eps, std, 1.0)
    return (data - mean) / std


def _window_clip(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if data.shape[1] < window_size:
        pad = window_size - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
    starts = list(range(0, max(data.shape[1] - window_size, 0) + 1, stride))
    if not starts:
        starts = [0]
    windows = [data[:, start : start + window_size] for start in starts]
    return np.stack(windows, axis=0).astype(np.float32)


def _infer_sfreq(data_lengths: Sequence[int]) -> Optional[int]:
    if not data_lengths:
        return None
    if sum(length % 200 == 0 for length in data_lengths) >= int(0.8 * len(data_lengths)):
        return 200
    return None


def _select_emotion_codes(index_rows: Sequence[dict], requested_codes: Optional[Sequence[int]] = None) -> list[int]:
    if requested_codes:
        return [int(code) for code in requested_codes]
    counts = Counter(int(row["raw_emotion_code"]) for row in index_rows)
    return [int(code) for code, _count in counts.most_common(3)]


def _resolve_label_groups(config: Dict[str, object], index_rows: Sequence[dict]) -> tuple[dict[int, int], list[str], list[int]]:
    configured_groups = config.get("emotion_label_groups")
    if configured_groups:
        label_map: dict[int, int] = {}
        label_names: list[str] = []
        included_codes: list[int] = []
        for label_index, (label_name, raw_codes) in enumerate(configured_groups.items()):
            label_names.append(str(label_name))
            for raw_code in raw_codes:
                code = int(raw_code)
                if code in label_map:
                    raise ValueError(f"Duplicate raw emotion code in grouped mapping: E{code}")
                label_map[code] = label_index
                included_codes.append(code)
        return label_map, label_names, included_codes

    selected_codes = _select_emotion_codes(index_rows, config.get("emotion_codes"))
    label_map = {int(code): idx for idx, code in enumerate(selected_codes)}
    label_names = [f"E{code}" for code in selected_codes]
    return label_map, label_names, selected_codes


def inspect_and_cache_eeg_emotions(config: Dict[str, object]) -> dict:
    data_dir = Path(str(config["data_dir"]))
    processed_dir = Path(str(config["processed_dir"]))
    cache_dir = processed_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    inspect_summary = inspect_eeg_emotions_dataset(data_dir, processed_dir)
    inferred_sfreq = inspect_summary.get("inferred_sfreq")
    target_sfreq = config.get("target_sfreq")
    target_sfreq = int(target_sfreq) if target_sfreq is not None else inferred_sfreq
    window_size = int(config.get("window_size_samples", 3200))
    stride = int(config.get("stride_size_samples", window_size))

    raw_paths = sorted(path for path in data_dir.glob("*.mat"))
    sample_rows: list[dict] = []
    lengths: list[int] = []
    for path in raw_paths:
        raw = np.asarray(scipy.io.loadmat(path)["Data"], dtype=np.float32)
        meta = parse_filename_metadata(path)
        lengths.append(int(raw.shape[1]))
        current_sfreq = inferred_sfreq
        if target_sfreq is not None and current_sfreq is not None and current_sfreq != target_sfreq:
            raw = resample_poly(raw, up=target_sfreq, down=current_sfreq, axis=1).astype(np.float32)

        windows = _window_clip(raw, window_size=window_size, stride=stride)
        processed_path = cache_dir / f"{path.stem}.npz"
        np.savez_compressed(
            processed_path,
            windows=windows.astype(np.float32),
            subject_id=np.array(meta["subject_id"]),
            raw_emotion_code=np.array(meta["emotion_code"], dtype=np.int64),
            night=np.array(meta["night"], dtype=np.int64),
            report_index=np.array(meta["report_index"], dtype=np.int64),
            stage=np.array(meta["stage"]),
            sfreq=np.array(target_sfreq if target_sfreq is not None else -1, dtype=np.int64),
        )
        for window_index in range(windows.shape[0]):
            sample_rows.append(
                {
                    "sample_id": f"{path.stem}:{window_index}",
                    "processed_path": str(processed_path),
                    "window_index": int(window_index),
                    "raw_emotion_code": int(meta["emotion_code"]),
                    "subject_id": str(meta["subject_id"]),
                    "night": int(meta["night"]),
                    "report_index": int(meta["report_index"]),
                    "stage": str(meta["stage"]),
                }
            )

    label_map, label_names, selected_codes = _resolve_label_groups(config, sample_rows)
    filtered_rows = [row for row in sample_rows if int(row["raw_emotion_code"]) in label_map]
    for row in filtered_rows:
        row["label"] = int(label_map[int(row["raw_emotion_code"])])

    index_payload = {
        "data_dir": str(data_dir),
        "processed_dir": str(processed_dir),
        "window_size_samples": window_size,
        "stride_size_samples": stride,
        "target_sfreq": target_sfreq,
        "selected_emotion_codes": selected_codes,
        "label_map": label_map,
        "label_names": label_names,
        "samples": filtered_rows,
    }
    (processed_dir / "index.json").write_text(json.dumps(index_payload, indent=2))
    logger.info(
        "inspect_and_cache_eeg_emotions n_raw=%d n_windows=%d labels=%s selected_codes=%s",
        len(raw_paths),
        len(filtered_rows),
        label_names,
        selected_codes,
    )
    return index_payload


def load_cached_index(processed_dir: Path) -> dict:
    index_path = Path(processed_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing cached EEG emotion index: {index_path}")
    return json.loads(index_path.read_text())


def _split_subject_wise(
    rows: Sequence[dict],
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[list[int], list[int], list[int]]:
    subject_ids = sorted({row["subject_id"] for row in rows})
    by_subject: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_subject.setdefault(row["subject_id"], []).append(idx)

    rng = np.random.default_rng(random_seed)
    all_labels = {int(row["label"]) for row in rows}
    for _attempt in range(128):
        shuffled = subject_ids.copy()
        rng.shuffle(shuffled)
        n_subjects = len(shuffled)
        n_test = max(1, int(round(n_subjects * test_fraction)))
        n_val = max(1, int(round(n_subjects * val_fraction)))
        test_subjects = set(shuffled[:n_test])
        val_subjects = set(shuffled[n_test : n_test + n_val])
        train_subjects = set(shuffled[n_test + n_val :])
        if not train_subjects:
            continue
        train_idx = [idx for subject in train_subjects for idx in by_subject[subject]]
        val_idx = [idx for subject in val_subjects for idx in by_subject[subject]]
        test_idx = [idx for subject in test_subjects for idx in by_subject[subject]]
        splits = [train_idx, val_idx, test_idx]
        if all(all_labels.issubset({int(rows[i]["label"]) for i in split}) for split in splits):
            return train_idx, val_idx, test_idx

    logger.warning("Could not find subject-wise split with all classes present in each split; using best-effort split.")
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)
    n_subjects = len(shuffled)
    n_test = max(1, int(round(n_subjects * test_fraction)))
    n_val = max(1, int(round(n_subjects * val_fraction)))
    test_subjects = set(shuffled[:n_test])
    val_subjects = set(shuffled[n_test : n_test + n_val])
    train_subjects = set(shuffled[n_test + n_val :])
    train_idx = [idx for subject in train_subjects for idx in by_subject[subject]]
    val_idx = [idx for subject in val_subjects for idx in by_subject[subject]]
    test_idx = [idx for subject in test_subjects for idx in by_subject[subject]]
    return train_idx, val_idx, test_idx


def _split_sample_wise(
    rows: Sequence[dict],
    val_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[list[int], list[int], list[int]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    indices = np.arange(len(rows))
    test_split = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_seed)
    train_val_idx, test_idx = next(test_split.split(indices, labels))
    remaining_labels = labels[train_val_idx]
    effective_val_fraction = val_fraction / max(1e-8, 1.0 - test_fraction)
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=effective_val_fraction, random_state=random_seed + 1)
    train_sub_idx, val_sub_idx = next(val_split.split(train_val_idx, remaining_labels))
    train_idx = train_val_idx[train_sub_idx].tolist()
    val_idx = train_val_idx[val_sub_idx].tolist()
    return train_idx, val_idx.tolist(), test_idx.tolist()


def build_or_load_splits(config: Dict[str, object]) -> dict:
    processed_dir = Path(str(config["processed_dir"]))
    split_path = processed_dir / "splits.json"
    if split_path.exists() and not bool(config.get("force_rebuild_cache", False)):
        return json.loads(split_path.read_text())

    index_payload = load_cached_index(processed_dir)
    rows = index_payload["samples"]
    use_subject_split = bool(config.get("use_subject_split", True))
    val_fraction = float(config.get("val_fraction", 0.2))
    test_fraction = float(config.get("test_fraction", 0.2))
    random_seed = int(config.get("random_seed", 13))

    if use_subject_split and all(row.get("subject_id") for row in rows):
        split_indices = _split_subject_wise(rows, val_fraction, test_fraction, random_seed)
        split_mode = "subject"
    else:
        logger.warning("Falling back to stratified sample-wise split; subject-wise evaluation is unavailable.")
        split_indices = _split_sample_wise(rows, val_fraction, test_fraction, random_seed)
        split_mode = "sample"

    train_idx, val_idx, test_idx = split_indices
    payload = {
        "mode": split_mode,
        "train": [rows[i] for i in train_idx],
        "val": [rows[i] for i in val_idx],
        "test": [rows[i] for i in test_idx],
    }
    split_path.write_text(json.dumps(payload, indent=2))
    logger.info(
        "build_or_load_splits mode=%s train=%d val=%d test=%d",
        split_mode,
        len(payload["train"]),
        len(payload["val"]),
        len(payload["test"]),
    )
    return payload


class EEGEmotionDataset(Dataset):
    """Lazy dataset over cached fixed-length emotion windows."""

    def __init__(
        self,
        rows: Sequence[dict],
        normalization_mode: str = "per_sample",
        *,
        augment: bool = False,
        channel_dropout_prob: float = 0.0,
        time_mask_prob: float = 0.0,
        time_mask_fraction: float = 0.1,
        amplitude_jitter_std: float = 0.0,
        random_seed: int = 13,
    ) -> None:
        self.rows = [EmotionSample(**row) for row in rows]
        self.normalization_mode = normalization_mode
        self.augment = augment
        self.channel_dropout_prob = float(channel_dropout_prob)
        self.time_mask_prob = float(time_mask_prob)
        self.time_mask_fraction = float(time_mask_fraction)
        self.amplitude_jitter_std = float(amplitude_jitter_std)
        self._rng = np.random.default_rng(random_seed)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._stats_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        logger.info(
            "EEGEmotionDataset n_samples=%d normalization=%s augment=%s",
            len(self.rows),
            self.normalization_mode,
            self.augment,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_windows(self, path: str) -> np.ndarray:
        if path in self._cache:
            windows = self._cache.pop(path)
            self._cache[path] = windows
            return windows
        payload = np.load(path, allow_pickle=True)
        windows = payload["windows"].astype(np.float32)
        self._cache[path] = windows
        if len(self._cache) > 8:
            self._cache.popitem(last=False)
        return windows

    def _load_recording_stats(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if path in self._stats_cache:
            stats = self._stats_cache.pop(path)
            self._stats_cache[path] = stats
            return stats
        windows = self._load_windows(path)
        mean = windows.mean(axis=(0, 2), keepdims=False).astype(np.float32)
        std = windows.std(axis=(0, 2), keepdims=False).astype(np.float32)
        std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        self._stats_cache[path] = (mean, std)
        if len(self._stats_cache) > 8:
            self._stats_cache.popitem(last=False)
        return mean, std

    def _augment_window(self, x: np.ndarray) -> np.ndarray:
        augmented = x.copy()
        if self.channel_dropout_prob > 0.0:
            keep_mask = self._rng.random(augmented.shape[0]) >= self.channel_dropout_prob
            if not np.any(keep_mask):
                keep_mask[self._rng.integers(0, augmented.shape[0])] = True
            augmented = augmented * keep_mask[:, None]
        if self.time_mask_prob > 0.0 and self._rng.random() < self.time_mask_prob:
            mask_len = max(1, int(round(augmented.shape[1] * self.time_mask_fraction)))
            mask_len = min(mask_len, augmented.shape[1])
            start = int(self._rng.integers(0, augmented.shape[1] - mask_len + 1))
            augmented[:, start : start + mask_len] = 0.0
        if self.amplitude_jitter_std > 0.0:
            scale = 1.0 + self._rng.normal(0.0, self.amplitude_jitter_std)
            noise = self._rng.normal(0.0, self.amplitude_jitter_std, size=augmented.shape).astype(np.float32)
            augmented = augmented * np.float32(scale) + noise
        return augmented.astype(np.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[index]
        windows = self._load_windows(row.processed_path)
        x = windows[row.window_index].astype(np.float32)
        if self.normalization_mode == "per_sample":
            x = _zscore(x)
        elif self.normalization_mode == "per_recording":
            mean, std = self._load_recording_stats(row.processed_path)
            x = ((x - mean[:, None]) / std[:, None]).astype(np.float32)
        if self.augment:
            x = self._augment_window(x)
        return torch.from_numpy(x), int(row.label), row.subject_id

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([row.label for row in self.rows], dtype=np.int64)

    @property
    def subject_ids(self) -> list[str]:
        return [row.subject_id for row in self.rows]


def compute_class_weights(labels: Sequence[int], num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.where(counts > 0, counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: Sequence[int], num_classes: int = 3) -> WeightedRandomSampler:
    class_weights = compute_class_weights(labels, num_classes=num_classes).numpy()
    sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float32)
    return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)

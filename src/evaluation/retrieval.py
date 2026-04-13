"""Phrase-bank retrieval metrics for semantic decoding evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _as_2d_float32(name: str, values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return array / norms


def compute_cosine_similarity_matrix(
    predicted_embeddings: np.ndarray | Sequence[Sequence[float]],
    bank_embeddings: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Return cosine similarities between each prediction and each bank entry."""
    predicted = _as_2d_float32("predicted_embeddings", predicted_embeddings)
    bank = _as_2d_float32("bank_embeddings", bank_embeddings)
    if predicted.shape[1] != bank.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"predicted={predicted.shape[1]} bank={bank.shape[1]}"
        )
    return _normalize_rows(predicted) @ _normalize_rows(bank).T


def resolve_target_indices(
    *,
    target_indices: Sequence[int] | np.ndarray | None = None,
    target_phrases: Sequence[str] | np.ndarray | None = None,
    bank_phrases: Sequence[str] | np.ndarray | None = None,
) -> np.ndarray:
    """Resolve one ground-truth bank index per sample."""
    if target_indices is not None:
        indices = np.asarray(target_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"target_indices must have shape (n,), got {indices.shape}")
        if indices.size == 0:
            raise ValueError("target_indices cannot be empty.")
        return indices

    if target_phrases is None or bank_phrases is None:
        raise ValueError(
            "Retrieval evaluation requires either target_indices or both "
            "target_phrases and bank_phrases."
        )

    bank_list = [str(phrase) for phrase in bank_phrases]
    lookup = {phrase: index for index, phrase in enumerate(bank_list)}
    resolved = []
    for phrase in target_phrases:
        key = str(phrase)
        if key not in lookup:
            raise KeyError(f"Target phrase {key!r} is missing from the phrase bank.")
        resolved.append(lookup[key])
    return np.asarray(resolved, dtype=np.int64)


def compute_retrieval_metrics(
    similarity_matrix: np.ndarray,
    target_indices: Sequence[int] | np.ndarray,
    *,
    top_ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute top-k accuracy and MRR from a similarity matrix."""
    similarities = _as_2d_float32("similarity_matrix", similarity_matrix)
    targets = np.asarray(target_indices, dtype=np.int64)
    if targets.ndim != 1:
        raise ValueError(f"target_indices must have shape (n,), got {targets.shape}")
    if similarities.shape[0] != targets.shape[0]:
        raise ValueError(
            "Sample count mismatch: "
            f"similarities={similarities.shape[0]} targets={targets.shape[0]}"
        )
    if np.any(targets < 0) or np.any(targets >= similarities.shape[1]):
        raise ValueError("target_indices contains values outside the phrase-bank range.")

    rankings = np.argsort(-similarities, axis=1)
    match_positions = np.argmax(rankings == targets[:, None], axis=1)

    metrics: dict[str, float] = {}
    bank_size = similarities.shape[1]
    for raw_k in top_ks:
        requested_k = int(raw_k)
        k = max(1, min(requested_k, bank_size))
        metrics[f"top{requested_k}"] = float(np.mean(match_positions < k))
    metrics["mrr"] = float(np.mean(1.0 / (match_positions + 1)))
    metrics["bank_size"] = float(bank_size)
    metrics["target_count"] = float(targets.shape[0])
    return metrics


def evaluate_retrieval(
    *,
    predicted_embeddings: np.ndarray | Sequence[Sequence[float]],
    bank_embeddings: np.ndarray | Sequence[Sequence[float]],
    target_indices: Sequence[int] | np.ndarray | None = None,
    target_phrases: Sequence[str] | np.ndarray | None = None,
    bank_phrases: Sequence[str] | np.ndarray | None = None,
    top_ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Convenience wrapper that computes ranking metrics from embeddings."""
    similarities = compute_cosine_similarity_matrix(predicted_embeddings, bank_embeddings)
    resolved_targets = resolve_target_indices(
        target_indices=target_indices,
        target_phrases=target_phrases,
        bank_phrases=bank_phrases,
    )
    return compute_retrieval_metrics(similarities, resolved_targets, top_ks=top_ks)


def load_phrase_bank_from_target_dir(target_embeddings_dir: Path) -> tuple[list[str], np.ndarray]:
    """Build a phrase bank by averaging saved target embeddings per report text."""
    target_embeddings_dir = Path(target_embeddings_dir)
    phrase_to_embeddings: dict[str, list[np.ndarray]] = {}

    for path in sorted(target_embeddings_dir.glob("sub-*_target_embeddings.npz")):
        payload = np.load(str(path), allow_pickle=True)
        embeddings = payload["target_embeddings"].astype(np.float32)

        if "report_text" in payload.files:
            phrase = str(payload["report_text"]).strip()
            if phrase:
                phrase_to_embeddings.setdefault(phrase, []).append(embeddings.mean(axis=0))
            continue

        if "report_texts" in payload.files:
            report_texts = np.asarray(payload["report_texts"], dtype=object)
            for phrase in np.unique(report_texts):
                phrase = str(phrase).strip()
                if not phrase:
                    continue
                phrase_mask = report_texts == phrase
                phrase_to_embeddings.setdefault(phrase, []).append(embeddings[phrase_mask].mean(axis=0))

    if not phrase_to_embeddings:
        raise ValueError(f"No target embedding phrase bank could be built from {target_embeddings_dir}")

    phrases = sorted(phrase_to_embeddings)
    bank_embeddings = np.stack(
        [np.stack(phrase_to_embeddings[phrase], axis=0).mean(axis=0) for phrase in phrases],
        axis=0,
    ).astype(np.float32)
    return phrases, bank_embeddings

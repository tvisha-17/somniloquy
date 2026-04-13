"""Tests for retrieval-based evaluation metrics."""

import numpy as np
import pytest


def test_evaluate_retrieval_computes_topk_and_mrr():
    from src.evaluation.retrieval import evaluate_retrieval

    bank_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    predicted_embeddings = np.array(
        [
            [0.99, 0.01],
            [0.60, 0.80],
        ],
        dtype=np.float32,
    )

    metrics = evaluate_retrieval(
        predicted_embeddings=predicted_embeddings,
        bank_embeddings=bank_embeddings,
        target_indices=np.array([0, 2], dtype=np.int64),
        top_ks=(1, 2, 10),
    )

    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top2"] == pytest.approx(1.0)
    assert metrics["top10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_evaluate_retrieval_resolves_targets_from_phrases():
    from src.evaluation.retrieval import evaluate_retrieval

    metrics = evaluate_retrieval(
        predicted_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        bank_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        target_phrases=["flying"],
        bank_phrases=["flying", "running"],
    )

    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)


def test_evaluate_retrieval_requires_ground_truth_mapping():
    from src.evaluation.retrieval import evaluate_retrieval

    with pytest.raises(ValueError, match="requires either target_indices"):
        evaluate_retrieval(
            predicted_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            bank_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        )


def test_evaluate_retrieval_rejects_missing_target_phrase():
    from src.evaluation.retrieval import resolve_target_indices

    with pytest.raises(KeyError, match="missing from the phrase bank"):
        resolve_target_indices(target_phrases=["water"], bank_phrases=["flying", "running"])

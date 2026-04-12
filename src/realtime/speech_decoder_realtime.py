"""Realtime phrase retrieval, abstention, and smoothing."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _as_tensor(array: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(array):
        return array.to(device=device, dtype=torch.float32)
    return torch.as_tensor(array, device=device, dtype=torch.float32)


@dataclass
class PhraseBank:
    """Candidate phrase bank for cosine-similarity retrieval."""

    phrases: list[str]
    embeddings: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.phrases) == 0:
            raise ValueError("Phrase bank cannot be empty.")
        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings shape (n_phrases, dim), got {tuple(self.embeddings.shape)}")
        if self.embeddings.shape[0] != len(self.phrases):
            raise ValueError("Phrase count must match embedding rows.")
        self.embeddings = F.normalize(self.embeddings.to(dtype=torch.float32), dim=-1)
        logger.info("phrase_bank embeddings_shape=%s n_phrases=%d", tuple(self.embeddings.shape), len(self.phrases))

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @classmethod
    def from_target_embedding_dir(cls, target_embeddings_dir: Path) -> "PhraseBank":
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
                continue

            logger.warning("Skipping %s because it does not contain report_text or report_texts.", path.name)

        if not phrase_to_embeddings:
            raise ValueError(f"No target embedding files found under {target_embeddings_dir}")

        phrases = sorted(phrase_to_embeddings)
        stacked = np.stack(
            [np.stack(phrase_to_embeddings[phrase], axis=0).mean(axis=0) for phrase in phrases],
            axis=0,
        ).astype(np.float32)
        return cls(phrases=phrases, embeddings=torch.from_numpy(stacked))


class WindowStatisticsEncoder(nn.Module):
    """Deterministic fallback encoder for dry runs and tests."""

    def __init__(self, target_embed_dim: int = 384, seed: int = 13) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(8, target_embed_dim, generator=generator) / np.sqrt(8.0)
        self.register_buffer("projection", projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, channels, time), got {tuple(x.shape)}")
        logger.info("window_stats_encoder input_shape=%s", tuple(x.shape))
        first_diff = x[:, :, 1:] - x[:, :, :-1]
        features = torch.stack(
            [
                x.mean(dim=(1, 2)),
                x.std(dim=(1, 2)),
                x.abs().mean(dim=(1, 2)),
                x.amax(dim=(1, 2)),
                x.amin(dim=(1, 2)),
                first_diff.abs().mean(dim=(1, 2)),
                x.mean(dim=-1).std(dim=1),
                x.std(dim=-1).mean(dim=1),
            ],
            dim=-1,
        )
        logger.info("window_stats_encoder feature_shape=%s", tuple(features.shape))
        return F.normalize(features @ self.projection, dim=-1)


class RealTimeSpeechDecoder:
    """Realtime semantic retrieval with abstention and smoothing."""

    def __init__(
        self,
        model: nn.Module,
        phrase_bank: PhraseBank,
        *,
        confidence_threshold: float = 0.3,
        smoothing_window: int = 5,
        required_majority: int = 3,
        top_k: int = 3,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.phrase_bank = phrase_bank
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.required_majority = required_majority
        self.top_k = top_k
        self.device = torch.device(device)
        self.history: deque[Optional[str]] = deque(maxlen=smoothing_window)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.phrase_bank.embeddings = self.phrase_bank.embeddings.to(self.device)

    def reset(self) -> None:
        self.history.clear()

    def predict_embedding(self, window: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return a normalized embedding for one window."""
        batch = _as_tensor(window, self.device)
        if batch.ndim == 2:
            batch = batch.unsqueeze(0)
        if batch.ndim != 3:
            raise ValueError(f"Expected window shape (channels, time) or (batch, channels, time), got {tuple(batch.shape)}")
        logger.info("realtime_decoder input_shape=%s", tuple(batch.shape))
        with torch.no_grad():
            prediction = self.model(batch)
        if prediction.ndim == 1:
            prediction = prediction.unsqueeze(0)
        prediction = F.normalize(prediction, dim=-1)
        logger.info("realtime_decoder embedding_shape=%s", tuple(prediction.shape))
        return prediction[0]

    def retrieve(self, predicted_embedding: torch.Tensor) -> dict:
        """Retrieve the top phrases and compute confidence."""
        similarities = torch.matmul(self.phrase_bank.embeddings, predicted_embedding)
        top_k = min(self.top_k, similarities.shape[0])
        scores, indices = torch.topk(similarities, k=top_k)
        phrases = [self.phrase_bank.phrases[int(index)] for index in indices.cpu().tolist()]
        score_values = scores.cpu().tolist()
        top1 = float(score_values[0])
        top2 = float(score_values[1]) if len(score_values) > 1 else 0.0
        confidence = float((top1 - top2) / (top1 + top2 + 1e-8))
        return {
            "phrases": phrases,
            "scores": score_values,
            "confidence": confidence,
        }

    def _is_smoothed(self, phrase: str) -> bool:
        counts = Counter(item for item in self.history if item is not None)
        return counts[phrase] >= self.required_majority

    def process_window(
        self,
        window: np.ndarray | torch.Tensor,
        *,
        timestamp: float,
        rem_probability: float,
    ) -> dict:
        """Decode a window into a JSON-serializable payload."""
        embedding = self.predict_embedding(window)
        retrieval = self.retrieve(embedding)
        top_phrase = retrieval["phrases"][0]
        confidence = retrieval["confidence"]
        confident = confidence >= self.confidence_threshold
        self.history.append(top_phrase if confident else None)
        smoothed = confident and self._is_smoothed(top_phrase)

        payload = {
            "timestamp": float(timestamp),
            "predicted_text": top_phrase if smoothed else "low confidence",
            "confidence": float(confidence),
            "alternatives": retrieval["phrases"][1:self.top_k],
            "abstained": not smoothed,
            "raw_top_phrase": top_phrase,
            "raw_scores": retrieval["scores"],
            "rem_probability": float(rem_probability),
        }
        logger.info(
            "realtime_decoder top_phrase=%s confidence=%.4f abstained=%s history=%s",
            top_phrase,
            confidence,
            payload["abstained"],
            list(self.history),
        )
        return payload

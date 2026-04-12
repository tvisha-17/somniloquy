"""Realtime REM detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


@dataclass
class SpectralREMScorer:
    """Heuristic REM scorer based on normalized spectral band power."""

    sfreq: float = 256.0
    delta_band: tuple[float, float] = (0.5, 4.0)
    theta_band: tuple[float, float] = (4.0, 8.0)
    alpha_band: tuple[float, float] = (8.0, 12.0)
    beta_band: tuple[float, float] = (12.0, 30.0)

    def _band_power(self, spectrum: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> float:
        mask = (freqs >= band[0]) & (freqs < band[1])
        if not np.any(mask):
            return 0.0
        return float(spectrum[..., mask].mean())

    def predict_proba(self, window: np.ndarray, stage_hint: Optional[int] = None) -> float:
        """Return a REM probability for a single EEG window."""
        array = np.asarray(window, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected window shape (channels, time), got {array.shape}")
        logger.info("rem_scorer window_shape=%s", tuple(array.shape))

        if stage_hint is not None:
            probability = 0.95 if int(stage_hint) == 4 else 0.05
            logger.info("rem_scorer stage_hint=%s probability=%.3f", stage_hint, probability)
            return probability

        if array.shape[-1] < 8:
            logger.warning("Window too short for spectral REM scoring: shape=%s", tuple(array.shape))
            return 0.01

        centered = array - array.mean(axis=-1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(centered, axis=-1)) ** 2
        averaged_spectrum = spectrum.mean(axis=0)
        freqs = np.fft.rfftfreq(array.shape[-1], d=1.0 / self.sfreq)

        delta = self._band_power(averaged_spectrum, freqs, self.delta_band)
        theta = self._band_power(averaged_spectrum, freqs, self.theta_band)
        alpha = self._band_power(averaged_spectrum, freqs, self.alpha_band)
        beta = self._band_power(averaged_spectrum, freqs, self.beta_band)
        total = delta + theta + alpha + beta + 1e-8

        delta_ratio = delta / total
        theta_ratio = theta / total
        alpha_ratio = alpha / total
        beta_ratio = beta / total

        score = 5.0 * theta_ratio - 3.5 * delta_ratio - 1.0 * beta_ratio + 0.5 * alpha_ratio
        probability = float(np.clip(_sigmoid(score), 0.0, 1.0))
        logger.info(
            "rem_scorer ratios delta=%.4f theta=%.4f alpha=%.4f beta=%.4f probability=%.4f",
            delta_ratio,
            theta_ratio,
            alpha_ratio,
            beta_ratio,
            probability,
        )
        return probability


class REMDetector:
    """Tracks consecutive REM probabilities and emits a trigger."""

    def __init__(
        self,
        scorer: Optional[SpectralREMScorer] = None,
        threshold: float = 0.7,
        required_consecutive: int = 3,
    ) -> None:
        self.scorer = scorer or SpectralREMScorer()
        self.threshold = threshold
        self.required_consecutive = required_consecutive
        self._consecutive = 0

    def reset(self) -> None:
        self._consecutive = 0

    def process_window(self, window: np.ndarray, stage_hint: Optional[int] = None) -> dict:
        """Score a window and update the trigger state."""
        probability = float(self.scorer.predict_proba(window, stage_hint=stage_hint))
        if probability >= self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        triggered = self._consecutive >= self.required_consecutive
        result = {
            "rem_probability": probability,
            "triggered": triggered,
            "consecutive_count": self._consecutive,
        }
        logger.info(
            "rem_detector probability=%.4f threshold=%.2f consecutive=%d triggered=%s",
            probability,
            self.threshold,
            self._consecutive,
            triggered,
        )
        return result

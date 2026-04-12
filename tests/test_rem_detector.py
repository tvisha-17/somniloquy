"""Tests for src/realtime/rem_detector.py."""

import asyncio

import numpy as np
import pytest


def _make_sine_window(freq_hz: float, sfreq: float = 256.0, duration_s: float = 2.0, n_channels: int = 4):
    times = np.arange(int(sfreq * duration_s), dtype=np.float32) / sfreq
    wave = np.sin(2.0 * np.pi * freq_hz * times).astype(np.float32)
    return np.stack([wave for _ in range(n_channels)], axis=0)


def test_spectral_rem_scorer_prefers_theta_over_delta():
    from src.realtime.rem_detector import SpectralREMScorer

    scorer = SpectralREMScorer(sfreq=256.0)
    theta_window = _make_sine_window(6.0)
    delta_window = _make_sine_window(1.0)

    theta_probability = scorer.predict_proba(theta_window)
    delta_probability = scorer.predict_proba(delta_window)

    assert theta_probability > delta_probability


def test_rem_detector_triggers_after_three_consecutive_high_scores():
    from src.realtime.rem_detector import REMDetector

    detector = REMDetector(threshold=0.7, required_consecutive=3)
    window = _make_sine_window(6.0)

    state1 = detector.process_window(window, stage_hint=4)
    state2 = detector.process_window(window, stage_hint=4)
    state3 = detector.process_window(window, stage_hint=4)
    state4 = detector.process_window(window, stage_hint=1)

    assert state1["triggered"] is False
    assert state2["triggered"] is False
    assert state3["triggered"] is True
    assert state4["triggered"] is False
    assert state4["consecutive_count"] == 0


def test_rem_detector_rejects_wrong_input_shape():
    from src.realtime.rem_detector import SpectralREMScorer

    scorer = SpectralREMScorer()
    with pytest.raises(ValueError):
        scorer.predict_proba(np.zeros((512,), dtype=np.float32))

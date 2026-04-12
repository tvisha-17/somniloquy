"""Realtime inference and demo package for Somniloquy."""

from src.realtime.rem_detector import REMDetector, SpectralREMScorer
from src.realtime.speech_decoder_realtime import PhraseBank, RealTimeSpeechDecoder, WindowStatisticsEncoder

__all__ = [
    "PhraseBank",
    "REMDetector",
    "RealTimeSpeechDecoder",
    "SpectralREMScorer",
    "WindowStatisticsEncoder",
]

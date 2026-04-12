"""Tests for src/realtime/speech_decoder_realtime.py."""

import numpy as np
import pytest
import torch
from torch import nn


class QueuedEmbeddingModel(nn.Module):
    """Returns a predefined embedding on each forward call."""

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = [torch.as_tensor(embedding, dtype=torch.float32) for embedding in embeddings]

    def forward(self, x):
        embedding = self.embeddings.pop(0)
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        return embedding.repeat(x.shape[0], 1)


def test_phrase_bank_rejects_empty_input():
    from src.realtime.speech_decoder_realtime import PhraseBank

    with pytest.raises(ValueError):
        PhraseBank(phrases=[], embeddings=torch.zeros((0, 3)))


def test_realtime_decoder_retrieves_top_phrase_and_alternatives():
    from src.realtime.speech_decoder_realtime import PhraseBank, RealTimeSpeechDecoder

    phrase_bank = PhraseBank(
        phrases=["flying", "running", "water"],
        embeddings=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    model = QueuedEmbeddingModel([[1.0, 0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.0]])
    decoder = RealTimeSpeechDecoder(
        model=model,
        phrase_bank=phrase_bank,
        confidence_threshold=0.2,
        required_majority=3,
        smoothing_window=5,
        top_k=3,
    )

    payload1 = decoder.process_window(np.ones((4, 8), dtype=np.float32), timestamp=0.0, rem_probability=0.9)
    payload2 = decoder.process_window(np.ones((4, 8), dtype=np.float32), timestamp=2.0, rem_probability=0.9)
    payload3 = decoder.process_window(np.ones((4, 8), dtype=np.float32), timestamp=4.0, rem_probability=0.9)

    assert payload1["raw_top_phrase"] == "flying"
    assert payload1["alternatives"] == ["running", "water"]
    assert payload1["abstained"] is True
    assert payload2["abstained"] is True
    assert payload3["abstained"] is False
    assert payload3["predicted_text"] == "flying"


def test_realtime_decoder_abstains_when_confidence_is_low():
    from src.realtime.speech_decoder_realtime import PhraseBank, RealTimeSpeechDecoder

    phrase_bank = PhraseBank(
        phrases=["alpha", "beta"],
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.98, 0.02],
            ]
        ),
    )
    model = QueuedEmbeddingModel([[1.0, 0.0]])
    decoder = RealTimeSpeechDecoder(
        model=model,
        phrase_bank=phrase_bank,
        confidence_threshold=0.8,
        required_majority=1,
        smoothing_window=3,
        top_k=2,
    )

    payload = decoder.process_window(np.ones((4, 8), dtype=np.float32), timestamp=0.0, rem_probability=0.9)
    assert payload["abstained"] is True
    assert payload["predicted_text"] == "low confidence"


def test_phrase_bank_loads_from_target_embedding_dir(tmp_path):
    from src.realtime.speech_decoder_realtime import PhraseBank

    np.savez(
        tmp_path / "sub-01_target_embeddings.npz",
        epoch_indices=np.array([0, 1], dtype=np.int64),
        target_embeddings=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        report_text=np.array("flying"),
    )
    np.savez(
        tmp_path / "sub-02_target_embeddings.npz",
        epoch_indices=np.array([0], dtype=np.int64),
        target_embeddings=np.array([[0.0, 1.0]], dtype=np.float32),
        report_text=np.array("running"),
    )

    phrase_bank = PhraseBank.from_target_embedding_dir(tmp_path)
    assert phrase_bank.phrases == ["flying", "running"]
    assert tuple(phrase_bank.embeddings.shape) == (2, 2)

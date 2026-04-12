"""Tests for src/models/zuna_decoder.py."""

import torch
import pytest
from torch import nn


class DummyBackbone(nn.Module):
    """Simple backbone that returns a 2D latent tensor."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(64 * 512, latent_dim)

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        return self.proj(flat)


class SequenceBackbone(nn.Module):
    """Backbone that returns a dict with 3D sequence features."""

    def __init__(self, latent_dim: int = 24):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(64, latent_dim)

    def extract_features(self, x):
        pooled = x.mean(dim=-1)
        features = self.proj(pooled).unsqueeze(1).repeat(1, 3, 1)
        return {"features": features}


def test_zuna_decoder_output_shape_and_frozen_backbone():
    from src.models.zuna_decoder import ZUNAForSpeechDecoding

    model = ZUNAForSpeechDecoding(
        target_embed_dim=384,
        dropout=0.0,
        backbone=DummyBackbone(latent_dim=32),
    )
    dummy = torch.randn(4, 64, 512)
    output = model(dummy)

    assert output.shape == (4, 384)
    assert all(not param.requires_grad for param in model.backbone.parameters())


def test_zuna_decoder_pools_sequence_features_from_dict_output():
    from src.models.zuna_decoder import ZUNAForSpeechDecoding

    model = ZUNAForSpeechDecoding(
        target_embed_dim=384,
        dropout=0.0,
        backbone=SequenceBackbone(latent_dim=24),
    )
    dummy = torch.randn(2, 64, 512)
    output = model(dummy)

    assert output.shape == (2, 384)


def test_zuna_decoder_requires_3d_input():
    from src.models.zuna_decoder import ZUNAForSpeechDecoding

    model = ZUNAForSpeechDecoding(
        target_embed_dim=384,
        dropout=0.0,
        backbone=DummyBackbone(latent_dim=16),
    )

    with pytest.raises(AssertionError):
        model(torch.randn(4, 64))


def test_zuna_decoder_raises_importerror_without_real_zuna_dependency():
    from src.models.zuna_decoder import ZUNAForSpeechDecoding

    with pytest.raises(ImportError, match="zuna"):
        ZUNAForSpeechDecoding(target_embed_dim=384)

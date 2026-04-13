"""Emotion classifier that reuses the existing EEG encoder workflow."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.zuna_decoder import ZUNAForSpeechDecoding
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EEGEmotionClassifier(nn.Module):
    """3-class emotion classifier on top of the existing EEG feature extractor."""

    def __init__(
        self,
        ch_names: list[str],
        *,
        num_classes: int = 3,
        dropout: float = 0.3,
        latent_dim: int | None = None,
        backbone_mode: str = "cnn",
        zuna_model_name: str = "Zyphra/ZUNA",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.feature_extractor = ZUNAForSpeechDecoding(
            ch_names=ch_names,
            target_embed_dim=384,
            dropout=dropout,
            zuna_model_name=zuna_model_name,
            latent_dim=latent_dim,
            backbone_mode=backbone_mode,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor.extract_features(x)
        logger.info("eeg_emotion_classifier features_shape=%s", tuple(features.shape))
        logits = self.classifier(features)
        logger.info("eeg_emotion_classifier logits_shape=%s", tuple(logits.shape))
        return logits


"""ZUNA speech-decoding wrapper for Phase 2."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - exercised only when zuna is installed
    from zuna.model import ZUNADiffusion as _ZUNADiffusion
except ImportError:  # pragma: no cover - current environment has no zuna package
    _ZUNADiffusion = None


def _call_with_supported_kwargs(
    fn: Callable[..., Any],
    x: torch.Tensor,
    electrode_coords: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    *,
    return_features: bool = False,
) -> Any:
    """Call a backbone method while passing only supported keyword arguments."""
    kwargs = {}
    try:
        signature = inspect.signature(fn)
        if electrode_coords is not None and "electrode_coords" in signature.parameters:
            kwargs["electrode_coords"] = electrode_coords
        if mask is not None and "mask" in signature.parameters:
            kwargs["mask"] = mask
        if return_features and "return_features" in signature.parameters:
            kwargs["return_features"] = True
    except (TypeError, ValueError):
        # Some callables do not expose a Python signature. Fall back to x-only.
        kwargs = {}
    return fn(x, **kwargs)


def _resolve_tensor_from_output(output: Any) -> torch.Tensor:
    """Extract the first plausible feature tensor from a backbone output."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("features", "latent", "latents", "hidden_states", "embeddings", "x"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            if torch.is_tensor(value):
                return value

    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return _resolve_tensor_from_output(item)
            except RuntimeError:
                continue

    raise RuntimeError(f"Could not resolve a tensor from backbone output of type {type(output)!r}")


def _pool_latent_tensor(latent: torch.Tensor, latent_dim: Optional[int]) -> torch.Tensor:
    """Reduce higher-rank latent tensors to shape (batch, latent_dim)."""
    if latent.ndim < 2:
        raise RuntimeError(f"Expected latent tensor rank >= 2, got shape {tuple(latent.shape)}")

    if latent.ndim == 2:
        return latent

    if latent_dim is not None and latent.shape[1] == latent_dim:
        pooled = latent.transpose(1, -1).reshape(latent.shape[0], -1, latent_dim).mean(dim=1)
        return pooled

    pooled = latent.reshape(latent.shape[0], -1, latent.shape[-1]).mean(dim=1)
    return pooled


def _infer_latent_dim(backbone: nn.Module) -> Optional[int]:
    """Infer latent dimensionality from common module/config attributes."""
    candidates = [backbone, getattr(backbone, "config", None)]
    attr_names = (
        "latent_dim",
        "hidden_size",
        "d_model",
        "embed_dim",
        "embedding_dim",
        "model_dim",
        "width",
    )
    for candidate in candidates:
        if candidate is None:
            continue
        for attr_name in attr_names:
            value = getattr(candidate, attr_name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _load_default_backbone(zuna_model_name: str) -> nn.Module:
    """Instantiate the real ZUNA backbone when the dependency is available."""
    if _ZUNADiffusion is None:
        raise ImportError(
            "The `zuna` package is not installed, so the real backbone cannot be loaded. "
            "Install `zuna` or pass an injected `backbone` for offline testing."
        )
    return _ZUNADiffusion.from_pretrained(zuna_model_name)


class ZUNAForSpeechDecoding(nn.Module):
    """Frozen EEG backbone plus a trainable semantic projection head."""

    def __init__(
        self,
        zuna_model_name: str = "Zyphra/ZUNA",
        target_embed_dim: int = 384,
        dropout: float = 0.3,
        latent_dim: Optional[int] = None,
        backbone: Optional[nn.Module] = None,
        backbone_factory: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()

        if backbone is not None and backbone_factory is not None:
            raise ValueError("Pass either `backbone` or `backbone_factory`, not both.")

        if backbone is None:
            backbone = backbone_factory() if backbone_factory is not None else _load_default_backbone(zuna_model_name)

        self.zuna = backbone
        self.backbone = backbone
        self.target_embed_dim = target_embed_dim
        self.latent_dim = latent_dim or _infer_latent_dim(backbone)
        if self.latent_dim is None:
            raise ValueError(
                "Could not infer backbone latent_dim. Pass `latent_dim` explicitly after inspecting the backbone."
            )

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, target_embed_dim),
            nn.LayerNorm(target_embed_dim),
        )

    def train(self, mode: bool = True) -> "ZUNAForSpeechDecoding":
        """Keep the frozen backbone in eval mode even when training the head."""
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_latent_features(
        self,
        x: torch.Tensor,
        electrode_coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract a 2D latent tensor from the backbone."""
        candidate_calls = []

        if hasattr(self.backbone, "extract_features"):
            candidate_calls.append((getattr(self.backbone, "extract_features"), False))
        if hasattr(self.backbone, "encode"):
            candidate_calls.append((getattr(self.backbone, "encode"), False))
        if hasattr(self.backbone, "forward_features"):
            candidate_calls.append((getattr(self.backbone, "forward_features"), False))
        if hasattr(self.backbone, "denoiser"):
            candidate_calls.append((getattr(self.backbone, "denoiser"), True))
        candidate_calls.append((self.backbone, False))

        last_error = None
        for fn, return_features in candidate_calls:
            try:
                output = _call_with_supported_kwargs(
                    fn,
                    x,
                    electrode_coords,
                    mask,
                    return_features=return_features,
                )
                latent = _resolve_tensor_from_output(output)
                pooled = _pool_latent_tensor(latent, self.latent_dim)
                if pooled.shape[-1] != self.latent_dim:
                    raise RuntimeError(
                        f"Expected latent dim {self.latent_dim}, got pooled shape {tuple(pooled.shape)}"
                    )
                return pooled
            except Exception as exc:  # pragma: no cover - exercised via fallback ordering
                last_error = exc

        raise RuntimeError("Failed to extract latent features from backbone.") from last_error

    def forward(
        self,
        x: torch.Tensor,
        electrode_coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project EEG windows into the semantic embedding space."""
        assert x.ndim == 3, f"Expected input rank 3 (batch, channels, time), got shape {tuple(x.shape)}"
        logger.info("zuna_decoder.forward input_shape=%s", tuple(x.shape))

        latent = self.extract_latent_features(x, electrode_coords=electrode_coords, mask=mask)
        logger.info("zuna_decoder.forward latent_shape=%s", tuple(latent.shape))

        output = self.head(latent)
        logger.info("zuna_decoder.forward output_shape=%s", tuple(output.shape))
        return output

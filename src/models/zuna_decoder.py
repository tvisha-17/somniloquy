"""ZUNA-compatible speech decoding wrapper.

This module supports three encoder paths:
- injected backbone for tests and controlled offline runs
- real frozen ZUNA encoder when the package and weights are available
- lightweight CNN fallback for fast demo-first checkpoints
"""

from __future__ import annotations

import json
import site
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)

_TF = 32
_XYZ_RANGE = 0.13
_XYZ_BINS = 100


def _discover_zuna_paths() -> tuple[Optional[Path], Optional[Path]]:
    for candidate in Path(sys.prefix).rglob("AY2latent_bci/transformer.py"):
        internal = candidate.parent
        return internal, internal.parents[2]

    for package_root in site.getsitepackages():
        internal = Path(package_root) / "zuna/inference/AY2l/lingua/apps/AY2latent_bci"
        if internal.exists():
            return internal, internal.parents[2]

    return None, None


_ZUNA_INTERNAL, _LINGUA_ROOT = _discover_zuna_paths()
_ZUNA_OK = False
try:
    for maybe_path in (_ZUNA_INTERNAL, _LINGUA_ROOT, _ZUNA_INTERNAL.parent if _ZUNA_INTERNAL else None):
        if maybe_path is not None:
            path_str = str(maybe_path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder  # type: ignore
    from huggingface_hub import hf_hub_download
    from lingua.args import dataclass_from_dict  # type: ignore
    from safetensors.torch import load_file as safe_load

    _ZUNA_OK = True
except Exception as exc:  # pragma: no cover
    logger.warning("Could not import ZUNA internals (%s).", exc)


def _get_channel_positions(ch_names: list[str]) -> torch.Tensor:
    try:
        import mne

        montage = mne.channels.make_standard_montage("standard_1020")
        pos_dict = montage.get_positions()["ch_pos"]
    except Exception:
        pos_dict = {}

    positions = []
    for ch in ch_names:
        lookup = ch.split("-")[0].upper()
        raw_pos = pos_dict.get(ch)
        if raw_pos is None:
            raw_pos = pos_dict.get(lookup)
        xyz = np.array(raw_pos, dtype=np.float32) if raw_pos is not None else np.zeros(3, dtype=np.float32)
        disc = np.floor((xyz + _XYZ_RANGE) / (2 * _XYZ_RANGE) * _XYZ_BINS).astype(int)
        positions.append(np.clip(disc, 0, _XYZ_BINS - 1))

    if not positions:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.tensor(np.stack(positions), dtype=torch.long)


def _build_tok_idx(chan_pos_discrete: torch.Tensor, n_time_tokens: int) -> torch.Tensor:
    n_ch = chan_pos_discrete.shape[0]
    repeated_positions = chan_pos_discrete.repeat_interleave(n_time_tokens, dim=0)
    time_index = torch.arange(n_time_tokens).repeat(n_ch).unsqueeze(-1)
    return torch.cat([repeated_positions, time_index], dim=-1).unsqueeze(0)


def _call_backbone(backbone: nn.Module, x: torch.Tensor, electrode_coords=None, mask=None):
    if hasattr(backbone, "extract_features"):
        return backbone.extract_features(x)
    if hasattr(backbone, "encode"):
        return backbone.encode(x, electrode_coords=electrode_coords, mask=mask)
    if hasattr(backbone, "forward_features"):
        return backbone.forward_features(x)
    return backbone(x)


def _resolve_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("features", "latent", "hidden_states", "embeddings", "x"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        raise RuntimeError(f"Could not resolve latent tensor from dict keys={sorted(output.keys())}")

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value
            if isinstance(value, dict):
                try:
                    return _resolve_tensor(value)
                except RuntimeError:
                    continue
        raise RuntimeError("Could not resolve latent tensor from tuple/list output.")

    raise RuntimeError(f"Unsupported backbone output type: {type(output)!r}")


def _pool_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 2:
        return latent
    if latent.ndim == 3:
        return latent.mean(dim=1)
    raise RuntimeError(f"Expected 2D or 3D latent tensor, got shape {tuple(latent.shape)}")


def _infer_latent_dim(backbone: nn.Module, latent_dim: Optional[int]) -> int:
    if latent_dim is not None:
        return int(latent_dim)
    for attr in ("latent_dim", "out_dim", "output_dim", "embed_dim", "hidden_size"):
        value = getattr(backbone, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not infer latent_dim from injected backbone; pass latent_dim explicitly.")


class _CNNEncoder(nn.Module):
    def __init__(self, n_ch: int, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ZUNAForSpeechDecoding(nn.Module):
    """Speech-decoding model with real-ZUNA and fast-start fallback modes."""

    def __init__(
        self,
        ch_names: Optional[list[str]] = None,
        target_embed_dim: int = 384,
        dropout: float = 0.3,
        zuna_model_name: str = "Zyphra/ZUNA",
        latent_dim: Optional[int] = None,
        backbone: Optional[nn.Module] = None,
        backbone_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.ch_names = list(ch_names or [])
        self.expected_n_channels = len(self.ch_names) if self.ch_names else None
        self.target_embed_dim = int(target_embed_dim)
        self.encoder_mode = ""
        self.backbone: Optional[nn.Module] = None
        self.encoder: Optional[nn.Module] = None
        self.register_buffer("chan_pos_discrete", _get_channel_positions(self.ch_names))

        if backbone is not None:
            self.backbone = backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self.encoder_mode = "injected"
            self._encoder_dim = _infer_latent_dim(self.backbone, latent_dim)
        else:
            resolved_mode = backbone_mode.lower()
            if resolved_mode not in {"auto", "zuna", "cnn"}:
                raise ValueError(f"Unsupported backbone_mode={backbone_mode!r}")
            if resolved_mode in {"auto", "zuna"}:
                try:
                    self._init_zuna(zuna_model_name)
                    self.encoder_mode = "zuna"
                except Exception as exc:
                    if resolved_mode == "zuna":
                        raise
                    logger.warning("Falling back to CNN encoder because ZUNA init failed: %s", exc)
            if not self.encoder_mode:
                n_channels = self.expected_n_channels or 1
                self.encoder = _CNNEncoder(n_ch=n_channels, out_dim=int(latent_dim or 128))
                self.encoder_mode = "cnn"
                self._encoder_dim = int(latent_dim or 128)

        self.head = nn.Sequential(
            nn.Linear(self._encoder_dim, 512),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(512, self.target_embed_dim),
            nn.LayerNorm(self.target_embed_dim),
        )

    def _init_zuna(self, repo_id: str) -> None:
        if not _ZUNA_OK:
            raise ImportError("zuna internals are unavailable in this environment.")

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as handle:
            cfg_dict = json.load(handle)

        model_args: DecoderTransformerArgs = dataclass_from_dict(DecoderTransformerArgs, cfg_dict["model"])
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="model-00001-of-00001.safetensors",
            token=False,
        )
        state_dict_raw = safe_load(weights_path, device="cpu")
        state_dict = {key.removeprefix("model."): value for key, value in state_dict_raw.items()}

        full_model = EncoderDecoder(model_args)
        full_model.load_state_dict(state_dict, strict=True)
        self.backbone = full_model.encoder
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._encoder_dim = int(model_args.encoder_output_dim)

    def train(self, mode: bool = True) -> "ZUNAForSpeechDecoding":
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def _validate_input(self, x: torch.Tensor) -> None:
        assert x.ndim == 3, f"Expected (batch, channels, time), got {tuple(x.shape)}"
        if self.expected_n_channels is not None and x.shape[1] != self.expected_n_channels:
            raise ValueError(
                f"Expected {self.expected_n_channels} channels based on model configuration, got {x.shape[1]}"
            )

    def _encode_zuna(self, x: torch.Tensor) -> torch.Tensor:
        assert self.backbone is not None
        batch_size, n_ch, n_t = x.shape
        if n_t % _TF != 0:
            raise ValueError(f"Time dimension {n_t} must be divisible by {_TF} for ZUNA tokenization.")
        n_time_tokens = n_t // _TF
        seq_len = n_ch * n_time_tokens
        tokens = x.reshape(batch_size, n_ch, n_time_tokens, _TF).reshape(batch_size, seq_len, _TF)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)
        tok_idx = _build_tok_idx(self.chan_pos_discrete, n_time_tokens).to(x.device).expand(batch_size, -1, -1)
        with torch.no_grad():
            logits, _ = self.backbone(
                token_values=tokens,
                seq_lens=seq_lens,
                tok_idx=tok_idx,
                attn_impl="sdpa",
            )
        return logits.mean(dim=1)

    def _encode_injected(self, x: torch.Tensor, electrode_coords=None, mask=None) -> torch.Tensor:
        assert self.backbone is not None
        with torch.no_grad():
            output = _call_backbone(self.backbone, x, electrode_coords=electrode_coords, mask=mask)
        return _pool_latent(_resolve_tensor(output))

    def _encode_cnn(self, x: torch.Tensor) -> torch.Tensor:
        assert self.encoder is not None
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        electrode_coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_input(x)
        logger.info("zuna_decoder.forward input_shape=%s mode=%s", tuple(x.shape), self.encoder_mode)

        if self.encoder_mode == "zuna":
            latent = self._encode_zuna(x)
        elif self.encoder_mode == "injected":
            latent = self._encode_injected(x, electrode_coords=electrode_coords, mask=mask)
        else:
            latent = self._encode_cnn(x)

        if latent.ndim != 2:
            raise RuntimeError(f"Expected pooled latent shape (batch, dim), got {tuple(latent.shape)}")

        logger.info("zuna_decoder.forward latent_shape=%s", tuple(latent.shape))
        output = self.head(latent)
        logger.info("zuna_decoder.forward output_shape=%s", tuple(output.shape))
        return output

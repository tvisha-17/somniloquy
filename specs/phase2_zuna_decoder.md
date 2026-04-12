# Spec: Phase 2 ZUNA Decoder

## Module

`src/models/zuna_decoder.py`

## Goal

Expose a `ZUNAForSpeechDecoding` module that accepts preprocessed EEG windows and returns one semantic embedding per window while keeping the EEG backbone frozen.

## Inputs

- `x`: `torch.Tensor` with shape `(batch, n_channels, n_timepoints)`
- `electrode_coords`: optional tensor passed through to the backbone
- `mask`: optional tensor passed through to the backbone
- constructor config:
  - `zuna_model_name: str`
  - `target_embed_dim: int`
  - `dropout: float`
  - `backbone_mode: "auto" | "zuna" | "cnn"`
  - optional injected backbone for tests or offline runs
  - optional `ch_names: list[str]`

## Outputs

- `forward(...) -> torch.Tensor` with shape `(batch, target_embed_dim)`
- metadata on the module:
  - resolved encoder mode
  - resolved latent dimension
  - frozen backbone parameters when a frozen backbone is used

## Edge Cases

- Missing or unusable `zuna` package in `auto` mode: log a warning and fall back to the CNN encoder
- Missing or unusable `zuna` package in `zuna` mode: raise a clear `ImportError` or `RuntimeError`
- Backbone returns dict/tuple/tensor instead of a direct latent tensor: resolve using adapter rules or raise `RuntimeError`
- Input tensor is not 3D: raise `AssertionError`
- Backbone latent is not 2D after extraction: raise `RuntimeError`
- Channel count at inference differs from the model's configured channel count: raise `ValueError`

## Success Criteria

- All frozen-backbone parameters have `requires_grad=False`
- Dummy input of shape `(4, 64, 512)` returns `(4, 384)`
- The module is importable when `zuna` is absent by using either an injected backbone or `backbone_mode="cnn"`
- INFO logging records input, latent, and output shapes

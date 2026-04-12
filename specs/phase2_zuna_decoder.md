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
  - optional injected backbone or backbone factory for tests

## Outputs

- `forward(...) -> torch.Tensor` with shape `(batch, target_embed_dim)`
- metadata on the module:
  - resolved latent dimension
  - frozen backbone parameters

## Edge Cases

- Missing `zuna` package when no injected backbone is supplied: raise a clear `ImportError`
- Backbone returns dict/tuple/tensor instead of a direct latent tensor: resolve using adapter rules or raise `RuntimeError`
- Input tensor is not 3D: raise `AssertionError`
- Backbone latent is not 2D after extraction: raise `RuntimeError`

## Success Criteria

- All backbone parameters have `requires_grad=False`
- Dummy input of shape `(4, 64, 512)` returns `(4, 384)`
- The module is importable even when `zuna` is absent, as long as tests inject a dummy backbone
- INFO logging records input, latent, and output shapes

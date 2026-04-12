# Spec: Phase 2 Training Pipeline

## Module

`src/training/finetune_zuna.py`

## Goal

Train only the speech-decoding head on REM-aligned DREAM epochs using processed `.npz` artifacts and subject-level train/val/test splits.

## Inputs

- config dict / YAML:
  - paths to EEG epochs, target embeddings, and split JSON
  - model hyperparameters
  - optimizer/training hyperparameters
  - checkpoint directory
  - optional `backbone_mode` for selecting `zuna` vs. CNN
- aligned artifacts:
  - `sub-<id>_epochs.npz`
  - `sub-<id>_target_embeddings.npz`
- `ZUNAForSpeechDecoding` instance or factory

## Outputs

- training history dict with per-epoch metrics
- best checkpoint in `checkpoints/zuna_finetuned/`
- validation cosine similarity logs

## Edge Cases

- Subject listed in a split but missing one of the required `.npz` files: skip with warning
- Batch size of one: still compute the configured cosine + MSE loss without special contrastive pairing assumptions
- No REM-aligned examples after filtering: raise `ValueError`
- NaN loss or NaN gradient: raise `RuntimeError` with epoch/step context
- Validation split empty: raise `ValueError`
- Subjects in a split have different channel layouts: intersect channels across the loaded subjects, reorder each subject to the shared order, and raise `ValueError` only if the shared channel set is empty

## Success Criteria

- Training dataset contains only REM-aligned epochs referenced by `epoch_indices`
- Training and validation examples share a consistent channel order and count after harmonization
- Only decoder-head parameters receive gradients
- Validation cosine similarity is computed and the best checkpoint is saved
- Unit tests cover happy path, missing-data skip, and NaN-gradient failure

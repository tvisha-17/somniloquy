# Phase 5A: EEG Emotion Quick Accuracy Tuning

## Goal

Improve the first-pass EEG emotion baseline with minimal changes to the existing pipeline.

## Inputs

- Cached EEG emotion windows from `data/processed/eeg_emotions/cache/`
- Subject/sample splits from `data/processed/eeg_emotions/splits.json`
- Existing `EEGEmotionClassifier`

## Outputs

- Updated training path with configurable class balancing and regularization
- Updated dataset path with less destructive normalization and optional augmentation
- Config defaults for a stronger quick baseline

## Planned Changes

1. Add `per_recording` normalization so each window can be normalized using channel statistics from its parent clip instead of only within-window z-scoring.
2. Add light training-only augmentations:
   - channel dropout
   - short time masking
   - small additive/amplitude jitter
3. Add configurable label smoothing and allow class-weighted loss to be combined with the weighted sampler.

## Edge Cases

- Empty caches or malformed windows should still fail with the existing loader errors.
- Normalization must guard against near-zero channel variance.
- Augmentation must preserve `(channels, timepoints)` shape and dtype.
- `k`-style config values of zero should disable the corresponding augmentation cleanly.

## Success Criteria

- Existing emotion dataset/model tests still pass.
- The training path remains CLI-compatible.
- The new knobs are optional and default-safe.

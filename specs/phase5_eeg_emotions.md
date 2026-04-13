# Phase 5 Spec: EEG Emotion Classification Baseline

## Goal

Pivot the current EEG dream-decoding stack to a fast 3-class emotion classifier using the existing dream-emotion dataset and as much of the current encoder/training structure as possible.

## Inputs

- Raw emotion-labeled EEG clips stored as `.mat` files under a configurable `data_dir`
- Filename metadata containing:
  - subject identifier
  - session/night identifier
  - emotion code
  - report index
  - sleep stage tag
- Config values for:
  - selected emotion codes
  - window size / stride
  - split fractions
  - normalization mode
  - training hyperparameters

## Outputs

- Cached processed windows under `data/processed/eeg_emotions/`
- Subject-wise or sample-wise split manifest JSON
- Best classifier checkpoint selected by validation macro F1
- Evaluation JSON with:
  - accuracy
  - balanced accuracy
  - macro precision / recall / F1
  - confusion matrix
  - permutation-test p-values
- Confusion matrix PNG

## Processing

1. Inspect the dataset structure and sample several `.mat` files.
2. Parse filename metadata and infer candidate labels / subject IDs.
3. Cache each clip into fixed-length windows with inherited label metadata.
4. Build splits:
   - subject-wise when subject IDs exist
   - otherwise stratified sample-wise with a warning
5. Reuse the existing EEG encoder pattern and attach a 3-class head.
6. Train with cross-entropy and class weighting / weighted sampling.
7. Early-stop and checkpoint on validation macro F1.
8. Evaluate on the held-out test split and run permutation tests.

## Key Assumptions

- The quickest statistically defensible 3-class target is the three most common emotion codes present in the dataset unless an explicit mapping is provided.
- Fixed-size windows can be extracted from each clip without changing the raw data.
- The current CNN/ZUNA-style encoder can be reused as a frozen or partially frozen feature extractor.

## Edge Cases

- Missing dataset path: fail with a clear error and log the configured path.
- Clips shorter than one window: zero-pad to one full window.
- Subject-wise splits with missing class coverage: retry several random seeds, then warn if a perfect split is impossible.
- Missing subject IDs: fall back to stratified sample-wise splitting and log that it is a weaker protocol.

## Success Criteria

- One CLI command can inspect, cache, train, and evaluate the emotion classifier baseline.
- The baseline reports balanced accuracy, macro F1, confusion matrix, and permutation-test p-values.
- Existing dream-retrieval code paths remain intact.

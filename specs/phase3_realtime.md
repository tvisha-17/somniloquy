# Spec: Phase 3 Realtime Inference

## Modules

- `src/realtime/rem_detector.py`
- `src/realtime/speech_decoder_realtime.py`

## Goal

Convert streaming 2-second EEG windows into stable, confidence-scored phrase predictions with abstention.

## REM Detector Inputs

- EEG window: `np.ndarray` or `torch.Tensor`, shape `(n_channels, n_timepoints)`
- optional stage hint for simulated playback
- config:
  - `rem_threshold`
  - `required_consecutive`
  - heuristic band definitions / score weights

## REM Detector Outputs

- per-window REM probability in `[0, 1]`
- boolean trigger when probability exceeds threshold for `required_consecutive` windows

## REM Detector Edge Cases

- wrong input rank: raise `ValueError`
- too-short window for spectral scoring: return low probability with a warning-safe fallback
- stage hint provided: it may override the heuristic score for demo playback

## Speech Decoder Inputs

- predicted semantic embedding, shape `(embedding_dim,)`
- candidate phrase bank:
  - phrases
  - embeddings, shape `(n_phrases, embedding_dim)`
- config:
  - `top_k`
  - `confidence_threshold`
  - `smoothing_window`
  - `required_majority`

## Speech Decoder Outputs

- JSON-serializable dict with:
  - `timestamp`
  - `predicted_text`
  - `confidence`
  - `alternatives`
  - `abstained`
  - `rem_probability`

## Speech Decoder Edge Cases

- empty phrase bank: raise `ValueError`
- top-2 similarity unavailable because bank has one phrase: confidence degrades gracefully
- confidence below threshold: abstain
- phrase not repeated enough in the smoothing buffer: abstain even if retrieval confidence is high

## Success Criteria

- REM trigger fires only after the configured consecutive-threshold rule
- retrieval returns top-3 phrases sorted by cosine similarity
- confidence uses `(top1 - top2) / (top1 + top2 + 1e-8)`
- temporal smoothing requires repeated agreement before emission

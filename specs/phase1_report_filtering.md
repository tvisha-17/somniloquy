# Spec: Report Filtering And Cleaning

## Modules

- `src/data/align_dream_reports.py`
- `src/training/finetune_zuna.py`

## Goal

Improve target quality by dropping tiny or empty report texts before alignment, removing lightweight filler words from retained texts, and exposing the best validation epoch explicitly in training outputs/checkpoints.

## Inputs

- alignment config additions:
  - `min_report_chars: int`
  - `min_report_alpha_chars: int`
  - `drop_report_patterns: list[str]`
  - `filler_words: list[str]`
- raw report text from CSV rows
- training history from the normal training loop

## Outputs

- aligned target files that contain cleaned report text only for usable reports
- skipped low-information reports logged at INFO level
- training result dict containing:
  - `best_epoch`
  - `best_val_cosine_similarity`
  - `best_checkpoint_path`

## Edge Cases

- cleaning removes all useful content: skip the report
- report contains only punctuation, whitespace, or configured fillers: skip the report
- `filler_words` config is empty: leave content unchanged except whitespace normalization
- no validation improvement after the first checkpoint: `best_epoch` still reflects the first saved best model

## Success Criteria

- low-information reports do not produce target embeddings
- cleaned reports preserve semantic content while removing configured fillers
- best checkpoint metadata records the epoch that achieved the best validation cosine

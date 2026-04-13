# Phase 5B: DEED Label Remapping

## Goal

Remap DEED raw emotion codes into the intended 3-class affect setup without changing the rest of the EEG emotion pipeline.

## Inputs

- Raw DEED-style filenames containing `E0`-`E5` codes
- EEG emotion config with optional grouped label mapping

## Outputs

- Cached index rows labeled with grouped class IDs
- Human-readable label names stored in the cached index and checkpoints
- Training/evaluation that run unchanged on the grouped labels

## Mapping

Default grouped emotion run for DEED:

- `negative`: `{E1, E2}`
- `neutral`: `{E3}`
- `positive`: `{E4, E5}`

`E0` (`dreamless`) is excluded from this 3-class run.

## Edge Cases

- If a grouped mapping contains duplicate raw codes, later duplicates should be rejected.
- If no grouped mapping is provided, preserve the existing raw-code behavior.
- Existing cache/split files may be stale after a remap; the pipeline needs an explicit rebuild option.

## Success Criteria

- Existing pipeline still supports raw-code classification when requested.
- Grouped-label cache/eval runs produce stable label names and class IDs.

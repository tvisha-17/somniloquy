---
phase: 01-data-pipeline
plan: "02"
subsystem: data
tags: [preprocessing, eeg, mne, normalization, epoching, tdd]
dependency_graph:
  requires:
    - src.utils.logging (get_logger)
    - data/processed/dream/inspection_report.json
  provides:
    - src.data.preprocess_dream (map_sleep_stages, bandpass_notch_resample, epoch_data, reject_by_peak_to_peak, zscore_per_channel, preprocess_subject, save_subject_npz, _load_raw)
    - configs/preprocess_dream.yaml
    - scripts/preprocess_dream.py
  affects:
    - data/processed/dream/eeg/ (writes sub-*_epochs.npz when raw data present)
tech_stack:
  added: [mne, pytest]
  patterns: [TDD-RED-GREEN, MNE-pipeline, graceful-missing-data, per-channel-zscore]
key_files:
  created:
    - src/data/preprocess_dream.py
    - tests/test_preprocess_dream.py
    - scripts/preprocess_dream.py
    - configs/preprocess_dream.yaml
  modified: []
decisions:
  - "_load_raw is a top-level function (not a method) so tests can monkeypatch it cleanly"
  - "epoch_data derives sleep stages from raw.annotations at each epoch start time via _stage_at_time helper, ignoring the sleep_stages argument passed in (which is set to empty array for the subject pipeline)"
  - "CLI inserts project root onto sys.path so it can be run as python scripts/preprocess_dream.py from project root without PYTHONPATH"
  - "bandpass_notch_resample modifies raw in-place (MNE convention) and returns it for chaining"
metrics:
  duration_minutes: 12
  completed_date: "2026-04-11"
  tasks_completed: 2
  files_created: 4
requirements_satisfied: [PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, PREP-06]
---

# Phase 01 Plan 02: DREAM EEG Preprocessing Pipeline Summary

## One-liner

MNE-based DREAM EEG pipeline with bandpass/notch/resample, fixed-length epoching, 200 µV peak-to-peak rejection, per-channel z-score normalization, and float32 .npz output per subject.

## What Was Built

Two tasks implemented Agent 1A from AGENTS.md:

**Task 1 — Preprocessing module + TDD tests** (commits `13f37af`, `504b90d`)

- `src/data/preprocess_dream.py`: 7 required functions
  - `map_sleep_stages`: case-insensitive mapping W→0, N1→1, N2→2, N3→3, REM→4, else→-1
  - `bandpass_notch_resample`: filter → notch → resample using MNE in-place API
  - `epoch_data`: fixed-length windows via mne.make_fixed_length_epochs, stage alignment via _stage_at_time helper
  - `reject_by_peak_to_peak`: drops epochs where ptp.max(axis=-1) > threshold_v
  - `zscore_per_channel`: normalizes across (epochs, time) per channel, returns float32
  - `preprocess_subject`: full pipeline, skips subjects with < min_epochs, output validation asserts float32/no NaN/normalized
  - `save_subject_npz`: writes sub-{id}_epochs.npz via np.savez with all 6 required keys
- `tests/test_preprocess_dream.py`: 8 TDD tests covering all stages, all pass

**Task 2 — CLI + config** (commit `2978e6d`)

- `configs/preprocess_dream.yaml`: all keys per AGENTS.md Agent 1A schema
- `scripts/preprocess_dream.py`: argparse CLI, yaml.safe_load, file discovery (4 EEG extensions), per-subject loop, running totals logging, graceful exit 0 when raw_root missing

## Verification Results

```
python -m pytest tests/test_preprocess_dream.py -q  -> 8 passed
python scripts/preprocess_dream.py --config configs/preprocess_dream.yaml  -> exit 0 (raw_root missing, graceful)
grep "def map_sleep_stages" src/data/preprocess_dream.py  -> PASS
grep "def bandpass_notch_resample" src/data/preprocess_dream.py  -> PASS
grep "def epoch_data" src/data/preprocess_dream.py  -> PASS
grep "def reject_by_peak_to_peak" src/data/preprocess_dream.py  -> PASS
grep "def zscore_per_channel" src/data/preprocess_dream.py  -> PASS
grep "def preprocess_subject" src/data/preprocess_dream.py  -> PASS
grep "def save_subject_npz" src/data/preprocess_dream.py  -> PASS
grep "np.savez" src/data/preprocess_dream.py  -> PASS
grep "np.float32" src/data/preprocess_dream.py  -> PASS
grep "notch_filter" src/data/preprocess_dream.py  -> PASS
No print() in src/  -> PASS
Config l_freq=0.5, h_freq=40.0, target_sfreq=256, min_epochs=100, reject_threshold=200e-6  -> PASS
```

## Deviations from Plan

None — plan executed exactly as written. The TDD warnings about filter_length exceeding signal length are expected for the short 5-second synthetic raw used in test_preprocess_subject_skips_if_below_min_epochs (by design — that test verifies skipping short subjects).

## Known Stubs

None. All functions are fully implemented. Raw data is absent on this machine (confirmed by inspection_report.json), so no .npz files are produced — this is not a stub, it is the correct graceful-missing-data behavior.

## Threat Flags

None. This plan creates only local file-read and file-write operations. No network endpoints, auth paths, or trust boundaries introduced.

## Self-Check: PASSED

Files confirmed present:
- src/data/preprocess_dream.py: FOUND
- tests/test_preprocess_dream.py: FOUND
- scripts/preprocess_dream.py: FOUND
- configs/preprocess_dream.yaml: FOUND

Commits confirmed:
- 13f37af: test(01-02): add failing tests for preprocessing pipeline (RED)
- 504b90d: feat(01-02): implement DREAM EEG preprocessing pipeline (GREEN)
- 2978e6d: feat(01-02): add preprocessing CLI and config (Agent 1A)

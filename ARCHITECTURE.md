# Architecture: Somniloquy

## Purpose

Somniloquy converts REM-labeled DREAM EEG into semantic dream-report embeddings and, later, into real-time retrieved text. The system is organized as a staged pipeline so data preparation, model adaptation, training, and streaming inference remain independently testable.

## Current Architecture

### 1. Data Pipeline

- `src/data/inspect_dream.py`
  - Audits `data/raw/dream/`, probes EEG/report formats, and writes `data/processed/dream/DATASET_CARD.md`.
- `src/data/preprocess_dream.py`
  - Loads raw EEG, keeps EEG channels, filters, notches, resamples, epochs, rejects artifacts, normalizes per subject, and writes `data/processed/dream/eeg/sub-<id>_epochs.npz`.
- `src/data/align_reports.py`
  - Encodes dream reports into 384-dim semantic targets and aligns them to REM epochs.
- `src/data/make_splits.py`
  - Produces subject-level train/val/test splits in `data/splits/dream_splits.json`.

### 2. Model Adapter Layer

- `src/models/zuna_decoder.py`
  - Owns the speech-decoding model.
  - Loads a frozen EEG backbone and projects backbone latents into 384-dim semantic space.
  - Exposes a single `forward(x, electrode_coords=None, mask=None)` interface with shape checks and INFO-level shape logging.

### 3. Training Layer

- `src/training/finetune_zuna.py`
  - Loads REM-only aligned training examples from the processed `.npz` artifacts.
  - Trains only the decoding head with combined cosine + MSE loss.
  - Evaluates held-out subjects with cosine similarity.
  - Stops on NaN gradients and saves the best checkpoint.

### 4. Runtime Layer

- `src/realtime/rem_detector.py`
  - Computes REM probabilities from 2-second windows and applies consecutive-window triggering.
- `src/realtime/speech_decoder_realtime.py`
  - Retrieves top phrases from a candidate bank, computes abstention confidence, and applies temporal smoothing before emission.
- `src/realtime/demo_server.py`
  - Streams pre-recorded epochs as simulated real-time input, publishes JSON events over WebSocket, and serves a lightweight dashboard.

## Data Contracts

### Preprocessed EEG

`data/processed/dream/eeg/sub-<id>_epochs.npz`

- `data`: `float32`, shape `(n_epochs, n_channels, n_timepoints)`
- `sleep_stages`: `int`, shape `(n_epochs,)`
- `subject_id`: scalar string
- `sfreq`: scalar float
- `ch_names`: array of strings
- `epoch_times_s`: `float`, shape `(n_epochs,)`

### Semantic Targets

`data/processed/dream/sub-<id>_target_embeddings.npz`

- `epoch_indices`: `int64`, shape `(n_rem_epochs,)`
- `target_embeddings`: `float32`, shape `(n_rem_epochs, 384)`
- `report_text`: scalar string

### Training Split

`data/splits/dream_splits.json`

- `train`, `val`, `test`: disjoint subject ID lists

## Phase 2 Design Decision

The project target is the real `zuna` package, but the package is not installed in the current environment, so direct API inspection is blocked. To keep Phase 2 moving without inventing hard-coded package internals, the model layer will use a narrow backbone adapter:

- Default path: lazily import `zuna` and instantiate the real backbone when available.
- Test path: accept an injected backbone object/factory so unit tests stay offline and deterministic.
- Feature extraction path: prefer explicit backbone methods (`extract_features`, `encode`, `forward_features`) and only then inspect common dict/tuple/tensor outputs.

This keeps the architecture aligned with `AGENTS.md` while making the unverified real-ZUNA integration explicit.

## Phase 3 Design Decisions

- The runtime REM detector will use a pluggable probability scorer:
  - default path: a lightweight heuristic spectral scorer that works offline
  - demo path: optional stage hints from preprocessed labeled epochs for simulated playback
- The phrase bank will be built from saved target-embedding files when available, so retrieval does not depend on a separate text encoder at demo time.
- The demo server will use FastAPI + WebSocket because those packages are already available in the environment and keep the dashboard single-process.

## Logging Policy

- Every preprocessing and model/training stage logs tensor or array shapes at INFO level.
- No `print()` in runtime modules.

## Testing Policy

- Every module under `src/` must have a matching `tests/test_*.py`.
- Phase 2 tests must cover:
  - frozen backbone behavior
  - output shape of `(batch, 384)`
  - REM-only sample selection
  - NaN gradient failure path
  - checkpoint writing on validation improvement

## Change Log

- 2026-04-11: Created the architecture document because `AGENTS.md` requires it and the file was missing.
- 2026-04-11: Recorded the temporary backbone-adapter decision because the real `zuna` package is unavailable in this environment.
- 2026-04-11: Expanded the runtime layer to cover REM detection, phrase retrieval/abstention, and the FastAPI demo server for Phase 3.

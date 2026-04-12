# Spec: Phase 3 Demo Server

## Module / Entry Point

- `src/realtime/demo_server.py`
- `scripts/demo_realtime.py`

## Goal

Run a local demo that replays preprocessed DREAM epochs as a real-time stream, performs REM detection and phrase retrieval, and shows results in a browser dashboard.

## Inputs

- preprocessed epoch `.npz` file containing:
  - `data`
  - `sleep_stages`
  - `epoch_times_s`
- optional checkpoint path
- candidate-bank source directory with `sub-*_target_embeddings.npz`
- config:
  - host / port
  - playback interval
  - detector thresholds
  - decoder thresholds

## Outputs

- FastAPI app serving:
  - `/`: HTML dashboard
  - `/ws`: WebSocket JSON event stream
- broadcast events containing hypnogram status, detector score, confidence, and top phrases

## Edge Cases

- no checkpoint available: allow injected/fallback model for testing, but raise clearly when neither real model nor fallback is configured
- no candidate embeddings found: raise `ValueError`
- no connected WebSocket clients: playback still runs without error
- playback source exhausted: emit a final completion event

## Success Criteria

- playback emits one event per epoch in timestamp order
- dashboard can connect over WebSocket and render incoming events
- decoder latency stays bounded by per-window synchronous processing

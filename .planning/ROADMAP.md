# Roadmap: Somniloquy

## Overview

Three phases deliver the full pipeline: raw EEG data gets inspected and processed into labeled epochs with semantic targets (Phase 1), the ZUNA model is loaded, a decoding head is trained, and a checkpoint is saved (Phase 2), and finally a real-time inference engine and live demo dashboard close the loop (Phase 3).

## Phases

- [ ] **Phase 1: Data Pipeline** - Inspect, preprocess, and align the DREAM dataset into training-ready artifacts
- [ ] **Phase 2: Model & Training** - Build the speech decoding head, fine-tune on REM epochs, save checkpoint
- [ ] **Phase 3: Real-Time Inference & Demo** - REM detector, streaming decoder, WebSocket output, live dashboard

## Phase Details

### Phase 1: Data Pipeline
**Goal**: Training-ready data artifacts exist — preprocessed EEG epochs and semantic target embeddings are on disk
**Depends on**: Nothing (first phase)
**Requirements**: INSP-01, INSP-02, INSP-03, INSP-04, PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, PREP-06, ALGN-01, ALGN-02, ALGN-03, ALGN-04
**Success Criteria** (what must be TRUE):
  1. `data/processed/dream/DATASET_CARD.md` exists and documents format, signal properties, subject counts, and known issues
  2. `data/processed/dream/eeg/sub-<id>_epochs.npz` files exist in float32 with shape `(n_epochs, n_channels, n_timepoints)`, validated for NaN and normalization
  3. `data/processed/dream/sub-<id>_target_embeddings.npz` files exist with epoch indices, 384-dim embeddings, and report text
  4. `data/splits/dream_splits.json` exists with subject-level train/val/test splits (no subject in multiple sets)
**Plans**: 3 plans
Plans:
- [x] 01-01-PLAN.md — Inspect DREAM dataset and write DATASET_CARD.md + inspection_report.json
- [x] 01-02-PLAN.md — Preprocess EEG (filter, epoch, reject, normalize) to sub-<id>_epochs.npz
- [x] 01-03-PLAN.md — Encode dream reports to 384-dim targets and write subject-level splits

### Phase 2: Model & Training
**Goal**: A fine-tuned checkpoint exists that maps REM EEG epochs to semantic embeddings
**Depends on**: Phase 1
**Requirements**: MODL-01, MODL-02, MODL-03, TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05
**Success Criteria** (what must be TRUE):
  1. ZUNA backbone loads without error and all backbone parameters have `requires_grad=False`
  2. `ZUNAForSpeechDecoding` passes shape assertion with `(4, 64, 512)` dummy input, outputting `(4, 384)`
  3. Training runs with cosine+MSE loss on REM-only epochs, ZUNA gradients remain zero throughout
  4. A checkpoint is saved to `checkpoints/zuna_finetuned/` and validation cosine similarity is logged
**Plans**: TBD

### Phase 3: Real-Time Inference & Demo
**Goal**: A live demo streams pre-recorded DREAM EEG through the full pipeline and displays decoded dream speech on a web dashboard
**Depends on**: Phase 2
**Requirements**: RT-01, RT-02, RT-03, RT-04, RT-05, RT-06, DEMO-01, DEMO-02, DEMO-03
**Success Criteria** (what must be TRUE):
  1. REM detector triggers inference only when P(REM) > 0.7 for 3 consecutive 2s windows
  2. Decoder outputs top-3 nearest phrases with a confidence score and abstains when confidence is below threshold
  3. Predictions arrive as JSON over WebSocket within 500 ms per 2s window
  4. Web dashboard displays a live hypnogram, confidence scores, top-3 phrases, and an optional dream word cloud
**Plans**: TBD
**UI hint**: yes

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 0/3 | Not started | - |
| 2. Model & Training | 0/TBD | Not started | - |
| 3. Real-Time Inference & Demo | 0/TBD | Not started | - |

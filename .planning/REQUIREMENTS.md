# Requirements: Somniloquy

**Defined:** 2026-04-11
**Core Value:** A working end-to-end pipeline: REM EEG in → semantically meaningful text out → live dashboard demo

## v1 Requirements

### Data Inspection

- [ ] **INSP-01**: Dataset audit script produces a directory and file listing of `data/raw/dream/` with extension counts
- [ ] **INSP-02**: EEG signal properties are logged (sfreq, n_channels, channel types, duration, annotations) for at least one subject file
- [ ] **INSP-03**: Dream report structure is documented (alignment strategy, vocabulary, report length)
- [ ] **INSP-04**: `data/processed/dream/DATASET_CARD.md` is written with: format, signal properties, subject/session counts, label structure, known issues

### EEG Preprocessing

- [ ] **PREP-01**: Raw DREAM EEG is filtered (0.5–40 Hz bandpass, 50 Hz notch), resampled to 256 Hz, and EEG channels isolated
- [ ] **PREP-02**: Sleep stage annotations are extracted and mapped to numeric codes (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM, -1=unknown)
- [ ] **PREP-03**: Data is epoched into 2-second windows with sleep stage labels; epochs exceeding 200 µV peak-to-peak are rejected
- [ ] **PREP-04**: Each channel is z-score normalized per subject across all epochs
- [ ] **PREP-05**: Output saved as `data/processed/dream/eeg/sub-<id>_epochs.npz` with shape `(n_epochs, n_channels, n_timepoints)` in float32, validated for NaN and normalization
- [ ] **PREP-06**: Subjects with fewer than 100 epochs are skipped and logged

### Semantic Alignment

- [ ] **ALGN-01**: Dream reports are loaded per subject and encoded into 384-dim sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- [ ] **ALGN-02**: Awakening timestamps are identified from EEG annotations; REM epochs within 30s before awakening are selected
- [ ] **ALGN-03**: Target embeddings are saved as `data/processed/dream/sub-<id>_target_embeddings.npz` with `epoch_indices`, `target_embeddings`, and `report_text`
- [ ] **ALGN-04**: Train/val/test splits are produced as `data/splits/dream_splits.json` (subject-level split, no subject in multiple sets)

### Model

- [ ] **MODL-01**: ZUNA backbone (`Zyphra/ZUNA`) loads without error; backbone parameters are fully frozen
- [ ] **MODL-02**: Speech decoding head (Linear → ReLU → Dropout → Linear → LayerNorm) outputs `(batch, 384)` embeddings from EEG latents
- [ ] **MODL-03**: `ZUNAForSpeechDecoding` passes shape assertion test with a `(4, 64, 512)` dummy input

### Training

- [ ] **TRAIN-01**: DataLoader selects only REM-stage epochs (sleep_stage == 4) from preprocessed data
- [ ] **TRAIN-02**: Combined cosine+MSE loss (0.7/0.3 weighting) trains only the decoding head; ZUNA backbone gradients are zero
- [ ] **TRAIN-03**: Validation cosine similarity is computed on held-out subjects every 5 epochs; best checkpoint is saved
- [ ] **TRAIN-04**: NaN gradient detection stops training and reports the step/batch where it occurred
- [ ] **TRAIN-05**: Training run produces a checkpoint in `checkpoints/zuna_finetuned/`

### Real-Time Inference

- [ ] **RT-01**: REM detector classifies 2s sliding windows and triggers inference when P(REM) > 0.7 for 3 consecutive windows
- [ ] **RT-02**: Real-time decoder loads fine-tuned model, computes embedding for each REM window, retrieves top-3 nearest phrases from candidate bank via cosine similarity
- [ ] **RT-03**: Confidence score is computed as `(top1_sim - top2_sim) / (top1_sim + top2_sim + 1e-8)`; system abstains when confidence < threshold
- [ ] **RT-04**: Temporal smoothing: output only when same phrase appears in ≥ 3 of 5 consecutive windows
- [ ] **RT-05**: Decoder outputs JSON over WebSocket: `{timestamp, predicted_text, confidence, alternatives}`
- [ ] **RT-06**: End-to-end latency per 2s window is < 500 ms

### Demo

- [ ] **DEMO-01**: Demo script reads pre-recorded DREAM EEG file and simulates real-time streaming
- [ ] **DEMO-02**: Web dashboard displays live hypnogram, confidence scores, and top-3 predicted phrases
- [ ] **DEMO-03**: Dashboard optionally shows a "dream word cloud" of frequently decoded phrases

## v2 Requirements

### Evaluation

- **EVAL-01**: Leave-one-subject-out validation reports mean cosine similarity and top-5 retrieval accuracy
- **EVAL-02**: Cross-session validation (train on night 1, test on night 2)
- **EVAL-03**: Robustness to missing channels: performance reported at 10%, 30%, 50% channel dropout
- **EVAL-04**: Calibration curve: performance vs. amount of per-subject REM data (1 min, 5 min, 10 min)
- **EVAL-05**: Reject option curve: coverage vs. accuracy at varying confidence thresholds
- **EVAL-06**: Qualitative comparison table: 10 examples of predicted phrases vs. actual dream reports (BERTScore)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full ZUNA fine-tuning (all layers) | Dataset too small; would overfit. Freeze backbone. |
| Exact word-level decoding | DREAM reports are sentence-level free text; semantic alignment is the honest approach |
| Real hardware EEG streaming | v1 uses pre-recorded data to simulate real-time; hardware integration is v2+ |
| Cross-subject evaluation agents (4A-4F) | Deferred to v2 — core pipeline + demo is the v1 goal |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INSP-01 | Phase 1 | Pending |
| INSP-02 | Phase 1 | Pending |
| INSP-03 | Phase 1 | Pending |
| INSP-04 | Phase 1 | Pending |
| PREP-01 | Phase 1 | Pending |
| PREP-02 | Phase 1 | Pending |
| PREP-03 | Phase 1 | Pending |
| PREP-04 | Phase 1 | Pending |
| PREP-05 | Phase 1 | Pending |
| PREP-06 | Phase 1 | Pending |
| ALGN-01 | Phase 1 | Pending |
| ALGN-02 | Phase 1 | Pending |
| ALGN-03 | Phase 1 | Pending |
| ALGN-04 | Phase 1 | Pending |
| MODL-01 | Phase 2 | Pending |
| MODL-02 | Phase 2 | Pending |
| MODL-03 | Phase 2 | Pending |
| TRAIN-01 | Phase 2 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| TRAIN-03 | Phase 2 | Pending |
| TRAIN-04 | Phase 2 | Pending |
| TRAIN-05 | Phase 2 | Pending |
| RT-01 | Phase 3 | Pending |
| RT-02 | Phase 3 | Pending |
| RT-03 | Phase 3 | Pending |
| RT-04 | Phase 3 | Pending |
| RT-05 | Phase 3 | Pending |
| RT-06 | Phase 3 | Pending |
| DEMO-01 | Phase 3 | Pending |
| DEMO-02 | Phase 3 | Pending |
| DEMO-03 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-11*
*Last updated: 2026-04-11 after initial definition*

# Somniloquy — ZUNA Fine-Tuning for Dream Speech Decoding

## What This Is

A real-time, non-invasive speech decoding system that operates during REM sleep, fine-tuning the ZUNA EEG foundation model on the DREAM dataset (505 subjects) to decode imagined speech from EEG signals. The system outputs text predictions with confidence scores and abstention, validated against post-awakening dream reports. Built for the Global Neurohack e184 Track.

## Core Value

A working end-to-end pipeline: REM EEG in → semantically meaningful text out → live dashboard demo that demonstrates dream speech decoding is feasible.

## Requirements

### Validated

(None yet — ship to validate)

### Active

**Data Pipeline**
- [ ] Dataset inspection protocol complete — DATASET_CARD.md written for DREAM dataset
- [ ] EEG preprocessing pipeline: filter, resample to 256 Hz, epoch to 2s windows, reject artifacts, z-score normalize per subject → `.npz` output
- [ ] Dream report semantic alignment: sentence-transformer embeddings assigned to REM epochs preceding awakening

**Model**
- [ ] ZUNA backbone loaded and frozen (`Zyphra/ZUNA`), speech decoding head attached (EEG latents → 384-dim embeddings)
- [ ] Fine-tuning loop: cosine+MSE loss, train only the head on REM epochs, validate on held-out subjects

**Real-Time Inference**
- [ ] REM sleep detector: lightweight classifier on 2s sliding windows, trigger when P(REM) > 0.7 for 3 consecutive windows
- [ ] Real-time speech decoder: predicted embedding → top-3 nearest phrases from candidate bank, confidence score, abstention when confidence < threshold, temporal smoothing
- [ ] WebSocket output: JSON stream of predictions to dashboard

**Demo**
- [ ] Live demo: simulated streaming of pre-recorded DREAM EEG → REM detection → decoder → web dashboard showing hypnogram, confidence, top-3 phrases

### Out of Scope

- Full ZUNA fine-tuning (all layers) — dataset too small, would overfit; freeze backbone
- Exact word-level decoding — DREAM reports are sentence-level; system decodes semantic content, not precise words
- Evaluation agents (cross-subject validation, robustness to missing channels, calibration curves, reject option curves) — deferred to stretch/v2
- Real hardware EEG streaming — v1 uses pre-recorded data to simulate real-time

## Context

- **Dataset**: DREAM database — EEG + free-text dream reports from 505 participants. Exact file format unknown until inspection (likely `.edf`). Reports are sentence-level, not word-level — this drives the semantic alignment approach.
- **ZUNA**: EEG foundation model from Zyphra (`pip install zuna`). Pre-trained denoiser architecture; we bypass diffusion to extract latent features and attach a new head.
- **Hackathon constraints**: Must satisfy e184 Track requirements — speech decoding, real-world robustness handling (noise, missing channels, short calibration, cross-subject generalization), reject option.
- **Key honesty constraint**: The system decodes *semantic content*, not exact words. Documentation and demo must be clear about this.
- **Implementation status**: The repository now contains Phase 1 data-pipeline code, Phase 2 model/training code, and Phase 3 realtime/demo code with tests. Live validation is still blocked on two external prerequisites: real DREAM artifacts and the real `zuna` package.

## Constraints

- **Tech Stack**: Python 3.10+, PyTorch 2.x, MNE-Python, HuggingFace Transformers, sentence-transformers, scikit-learn — fixed per AGENTS.md
- **Data**: `data/raw/` is read-only; all outputs go to `data/processed/` — enforced by AGENTS.md discipline rules
- **Model**: ZUNA backbone must remain frozen — fine-tune only the decoding head
- **Real-time latency**: < 500 ms per 2s window for the decoder
- **Confidence**: System must implement abstention (reject option) — output only when confidence > threshold

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Semantic alignment via sentence-transformers (not word-level CTC) | DREAM reports are free text, not word-aligned transcriptions | — Pending |
| Freeze ZUNA backbone, train only the head | Small dataset (~505 subjects × REM epochs) would overfit full fine-tuning | — Pending |
| Cosine + MSE combined loss | Cosine preserves direction (semantic similarity), MSE regularizes magnitude | — Pending |
| Candidate phrase bank for retrieval | Converts open-ended embedding space into interpretable text output | — Pending |
| Confidence via top1/top2 cosine margin | Simple, interpretable abstention criterion | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-11 after Phase 3 implementation*

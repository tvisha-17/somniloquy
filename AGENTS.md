## AGENTS.md — SomnoSpeech: ZUNA Fine‑Tuning for Dream Speech Decoding

This document is the authoritative instruction set for all agents working on this project. Read it fully before taking any action. Do not skip sections. Do not assume context that is not stated here.

---

### Project Overview

We are building a **real‑time, non‑invasive speech decoding system** that operates **during REM sleep**, using a fine‑tuned version of the **ZUNA EEG foundation model**. The system will decode imagined speech from EEG signals and output text, ranked candidates, or semantic embeddings. The target dataset is the **DREAM database** (EEG + dream reports from 505 participants). The output will be validated against post‑awakening reports and real‑time confidence metrics.

This project fulfills the **Global Neurohack e184 Track** requirements:
- **Speech decoding** (required)
- **Real‑world constraints**: noise, missing channels, short calibration, cross‑subject/session generalization, reject option
- **Clear tradeoffs**: we will operate only during REM sleep, output only when confidence is high, and handle OOV words via semantic retrieval.

---

### Stack & Repository Structure

- **Python 3.10+**, **PyTorch 2.x**, **MNE‑Python** (EEG preprocessing), **HuggingFace Transformers** (for BERT embeddings if needed), **scikit‑learn**, **NumPy**, **matplotlib**.
- **ZUNA**: installed via `pip install zuna`. We will use its pre‑trained weights and modify the denoiser output head for speech decoding.

Repository structure:

```
/
├── AGENTS.md               # this file
├── ARCHITECTURE.md         # system architecture document (auto‑updated)
├── data/
│   ├── raw/                # DREAM dataset (read‑only)
│   ├── processed/          # preprocessed EEG epochs, subject splits, vocabulary
│   └── splits/             # train/val/test splits as JSON index files
├── src/
│   ├── data/               # dataset inspection, loading, preprocessing, sleep staging
│   ├── models/             # ZUNA wrapper, speech decoding head, semantic projector
│   ├── training/           # fine‑tuning loop, loss functions, optimizer
│   ├── evaluation/         # retrieval metrics, abstention, cross‑subject validation
│   ├── realtime/           # REM detection, sliding window inference, confidence scoring
│   └── utils/              # logging, checkpointing, config
├── configs/                # YAML config files per experiment
├── scripts/                # CLI entry points (preprocess, train, evaluate, demo)
├── tests/                  # unit and integration tests
├── notebooks/              # exploratory analysis only, never production code
└── specs/                  # per‑component spec files
```

---

### Agent Discipline Rules (Non‑Negotiable)

These apply to every agent at every step regardless of task scope.

1. **State before acting.** Before any implementation step, state in one sentence what you are about to do and why. Do not take silent actions.
2. **Spec before implementing.** For any non‑trivial function, class, or module, write a brief spec (inputs, outputs, edge cases, success criterion) before writing code. Save specs to `specs/` as markdown files.
3. **Check ARCHITECTURE.md before every change.** If an implementation decision is not consistent with the architecture document, surface the conflict and resolve it explicitly before proceeding. If the architecture needs to change, update ARCHITECTURE.md and note what changed and why.
4. **Test every module.** Every module in `src/` has a corresponding test in `tests/`. Tests cover the happy path, at least one edge case, and at least one expected failure. Tests must pass before a module is considered done.
5. **Never modify raw data.** Files in `data/raw/` are read‑only. All transformations write to `data/processed/` with a clear naming convention.
6. **Log shapes at every stage.** EEG data shapes are a common source of silent bugs. At every preprocessing and model step, log the shape of the tensor being processed. Use Python logging (not print) at INFO level.
7. **Surface blockers immediately.** If a step cannot be completed because of a missing dependency, unclear spec, or data issue, stop and report the blocker with enough detail to resolve it. Do not attempt workarounds without surfacing them.

---

### Dataset Inspection Protocol (DREAM Dataset)

The DREAM dataset contains EEG recordings during sleep + post‑awakening dream reports. Its exact structure is not known a priori. Every agent must inspect the dataset before writing any preprocessing code.

**Step 1: Directory and File Audit**

```python
import os, pathlib
from collections import Counter

root = pathlib.Path("data/raw/dream")
for p in sorted(root.rglob("*")):
    depth = len(p.relative_to(root).parts)
    if depth <= 3:
        indent = "  " * (depth - 1)
        print(f"{indent}{p.name}/" if p.is_dir() else f"{indent}{p.name} [{p.stat().st_size//1024} KB]")
ext_counts = Counter(p.suffix for p in root.rglob("*") if p.is_file())
print("File types:", dict(ext_counts))
```

**Step 2: Format Identification**

- Look for `.edf`, `.set`, `.fif`, `.vhdr` (EEG).
- Look for `.txt`, `.csv`, `.tsv`, `.json` for dream reports and event markers.
- Determine if data is already split into subjects, sessions, sleep stages.

**Step 3: EEG Signal Inspection**

```python
import mne
raw = mne.io.read_raw_edf("path/to/file.edf", preload=False)
print("sfreq:", raw.info['sfreq'])
print("n_channels:", len(raw.ch_names))
print("channel types:", set(raw.get_channel_types()))
print("duration (s):", raw.times[-1])
print("annotations:", raw.annotations)   # sleep stage annotations?
```

**Step 4: Dream Report / Label Inspection**

- Are dream reports stored per‑awakening? Are they aligned to EEG time stamps?
- Are there word‑level or sentence‑level transcriptions?
- If only free text, we will use **semantic alignment** (contrastive learning with sentence embeddings) rather than exact word decoding.
- Document the vocabulary size and typical report length.

**Step 5: Subject and Session Structure**

- Number of subjects, sessions per subject, recordings per night.
- Are sleep stages annotated? (Essential for REM detection.)

**Step 6: Write Dataset Card**

Save to `data/processed/dream/DATASET_CARD.md` with sections:
- Format, loading library
- Signal properties (sfreq, channels, duration)
- Subject/session counts
- Label structure (free text / word‑level / sentence‑level)
- Known issues (bad channels, missing annotations, unbalanced classes)

**Do not proceed to preprocessing until the dataset card is complete and reviewed.**

---

### Preprocessing Agents

#### Agent 1A: EEG Preprocessor for Sleep (DREAM dataset)

**Responsibility:** Convert raw DREAM EEG into epoched, cleaned, sleep‑stage‑labeled segments, ready for ZUNA fine‑tuning.

**Inputs:**
- Raw DREAM dataset in `data/raw/dream/`
- Dataset card
- Config: `configs/preprocess_dream.yaml`

**Config schema:**

```yaml
dataset_name: dream
subjects: all | list[str]
l_freq: 0.5          # high‑pass (remove slow drift)
h_freq: 40.0         # low‑pass (remove muscle noise)
notch_freqs: [50.0]  # or 60 Hz depending on region
epoch_duration: 2.0  # seconds per epoch for ZUNA input
overlap: 0.5         # 50% overlap for sliding windows
sleep_stage_mapping: {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "?"}
reject_threshold: 200e-6  # peak‑to‑peak rejection in Volts
target_sfreq: 256    # ZUNA expects 256 Hz
output_dir: data/processed/dream/eeg/
```

**Processing steps (in order):**

1. Load raw file using appropriate MNE loader.
2. Pick EEG channels only: `raw.pick_types(eeg=True)`.
3. Apply bandpass filter: `raw.filter(l_freq, h_freq, fir_design='firwin')`.
4. Apply notch filter.
5. Resample to `target_sfreq`.
6. Extract sleep stage annotations from `raw.annotations`. Map to numeric codes.
7. Epoch the data into non‑overlapping (or overlapping) windows of `epoch_duration` seconds. Keep sleep stage label for each epoch.
8. Reject epochs exceeding `reject_threshold` (peak‑to‑peak).
9. Z‑score normalize each channel across all epochs **per subject**.
10. Save output as `data/processed/dream/eeg/sub-<id>_epochs.npz` containing:
    - `data`: float32 `(n_epochs, n_channels, n_timepoints)`
    - `sleep_stages`: int array `(n_epochs,)` (0=Wake,1=N1,2=N2,3=N3,4=REM)
    - `subject_id`: str
    - `sfreq`: float
    - `ch_names`: list
    - `epoch_times_s`: float array of start times

**Edge cases:**
- If sleep stage annotations are missing for some epochs, label as `-1` and log a warning.
- If number of epochs per subject < 100, skip that subject.

**Output validation:**
```python
assert data.dtype == np.float32
assert data.shape == (n_epochs, n_channels, n_timepoints)
assert not np.any(np.isnan(data))
assert abs(data.mean()) < 0.1 and abs(data.std() - 1.0) < 0.2
```

#### Agent 1B: Dream Report Aligner (Semantic Target Generation)

**Responsibility:** Convert free‑text dream reports into semantic embeddings that will serve as training targets for the speech decoder.

**Inputs:**
- Processed EEG epochs (from Agent 1A)
- Raw dream report text files
- Config: `configs/align_reports.yaml`

**Config schema:**
```yaml
report_file_pattern: "sub-*_dream.txt"
embedding_model: "all-MiniLM-L6-v2"   # Sentence‑BERT, lightweight
embedding_dim: 384
time_alignment_window: 30.0           # seconds: assume report refers to 30s before awakening
```

**Processing steps:**
1. For each subject, load the dream report text.
2. Use a sentence‑transformer to encode the entire report into a fixed‑dimension embedding.
3. Identify the awakening time from EEG annotations (look for annotation marking “awakening” or end of recording).
4. Select all REM epochs that fall within `time_alignment_window` seconds before awakening.
5. Assign the same report embedding to each of those REM epochs as the target.
6. Save a mapping file: `data/processed/dream/sub-<id>_target_embeddings.npz` containing:
    - `epoch_indices`: indices of REM epochs used for training
    - `target_embeddings`: float32 `(n_rem_epochs, embedding_dim)`
    - `report_text`: original string (for evaluation)

**Edge cases:**
- If no REM epochs are found in the window, skip this awakening.
- If a subject has multiple awakenings per night, each gets its own embedding target.

---

### ZUNA Fine‑Tuning Agents

#### Agent 2A: ZUNA Wrapper with Speech Decoding Head

**Responsibility:** Load the pre‑trained ZUNA model, freeze its backbone, and attach a new trainable head for speech decoding (output = semantic embedding or class logits).

**File:** `src/models/zuna_decoder.py`

```python
import torch
import torch.nn as nn
from zuna.model import ZUNADiffusion

class ZUNAForSpeechDecoding(nn.Module):
    """
    ZUNA backbone frozen, with a new head that maps EEG latents to semantic embeddings.
    Input: (batch, n_channels, n_timepoints)  (already preprocessed)
    Output: (batch, target_embed_dim)
    """
    def __init__(self, zuna_model_name="Zyphra/ZUNA", target_embed_dim=384, dropout=0.3):
        super().__init__()
        self.zuna = ZUNADiffusion.from_pretrained(zuna_model_name)
        # Freeze all ZUNA parameters
        for param in self.zuna.parameters():
            param.requires_grad = False
        
        # Determine the latent dimension of ZUNA's denoiser output
        # (We need to inspect the model. Let's assume it's 1024)
        self.latent_dim = 1024  # Will be read from config after inspection
        
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, target_embed_dim),
            nn.LayerNorm(target_embed_dim)
        )
    
    def forward(self, x, electrode_coords=None, mask=None):
        # x: (batch, channels, time)
        # Get latent features from ZUNA's denoiser (bypass diffusion)
        # This requires modifying ZUNA's forward or using internal method.
        # For now, we assume we can call: latent = self.zuna.encode(x, electrode_coords, mask)
        # If not available, we will use the denoiser's intermediate representation.
        latent = self.zuna.denoiser(x, electrode_coords, mask, return_features=True)
        return self.head(latent)
```

**Implementation notes:**
- We must inspect the actual ZUNA source code (`zuna/model.py`) to find a method that returns latent features without noise prediction.
- If no such method exists, we will write a small wrapper that runs the denoiser up to the transformer output but stops before the final projection.
- Add shape assertions in forward: `assert x.ndim == 3` etc.

**Tests:**
```python
def test_zuna_decoder_output_shape():
    model = ZUNAForSpeechDecoding(target_embed_dim=384)
    dummy = torch.randn(4, 64, 512)  # 64 channels, 512 time points
    out = model(dummy)
    assert out.shape == (4, 384)
```

#### Agent 2B: Training Loop – Fine‑tuning ZUNA Head

**Responsibility:** Train only the new head using a contrastive or MSE loss between predicted embeddings and dream report embeddings.

**File:** `src/training/finetune_zuna.py`

**Config (`configs/finetune_zuna.yaml`):**
```yaml
# Data
eeg_epochs_dir: data/processed/dream/eeg/
target_embeddings_dir: data/processed/dream/
split_file: data/splits/dream_splits.json   # generated by Agent 1D

# Model
zuna_model_name: Zyphra/ZUNA
target_embed_dim: 384
dropout: 0.3

# Loss
loss_type: cosine_mse   # combination of cosine similarity + MSE
cosine_weight: 0.7
mse_weight: 0.3

# Training
batch_size: 32
lr: 1e-4
weight_decay: 1e-5
n_epochs: 30
warmup_steps: 500
grad_clip: 1.0
device: cuda

# Logging
checkpoint_dir: checkpoints/zuna_finetuned/
log_every_n_steps: 10
val_every_n_epochs: 5
```

**Loss function:**
```python
def contrastive_mse_loss(pred_emb, target_emb):
    # Normalize embeddings
    pred_norm = F.normalize(pred_emb, dim=-1)
    target_norm = F.normalize(target_emb, dim=-1)
    cos_sim = (pred_norm * target_norm).sum(dim=-1).mean()
    cos_loss = 1 - cos_sim
    mse_loss = F.mse_loss(pred_emb, target_emb)
    return cos_weight * cos_loss + mse_weight * mse_loss
```

**Training loop requirements:**
- Load preprocessed EEG epochs and corresponding target embeddings.
- Create a DataLoader that only includes REM epochs (sleep_stage == REM).
- Forward pass through `ZUNAForSpeechDecoding`, compute loss, backprop only through the head.
- Validation: compute cosine similarity between predicted and target embeddings on held‑out subjects.
- Save best model based on validation cosine similarity.

**Edge cases:**
- If a batch has only one sample, skip contrastive loss (log warning).
- If any gradient becomes NaN, stop training and report.

---

### Real‑Time Dream Speech Decoding (Demo Agent)

#### Agent 3A: REM Sleep Detector

**Responsibility:** Implement a lightweight sleep stage classifier that runs on streaming EEG to detect REM sleep in real time.

**File:** `src/realtime/rem_detector.py`

**Spec:**
- Input: sliding window of raw EEG (2 seconds, 256 Hz → 512 samples, 64 channels).
- Output: probability of REM sleep (0–1).
- Model: a small CNN or a pre‑trained sleep stage classifier (e.g., from `sleepy` library or a custom one trained on DREAM data).
- Threshold: output REM when probability > 0.7 for at least 3 consecutive windows.

**Processing:**
- Use a queue to buffer EEG data.
- For each new window, run the classifier.
- If REM detected, trigger the speech decoding pipeline.

**Test:** Simulate a recording with known REM periods and verify detection latency < 10 seconds.

#### Agent 3B: Real‑Time Speech Decoder with Reject Option

**Responsibility:** Load the fine‑tuned ZUNA model, apply it to incoming REM EEG windows, and output text (or ranked candidates) with confidence.

**File:** `src/realtime/speech_decoder_realtime.py`

**Processing loop:**
1. Buffer EEG windows of length 2 seconds (same as training).
2. For each window, pass through `ZUNAForSpeechDecoding` to get predicted embedding.
3. Compare predicted embedding to a **pre‑computed bank of candidate phrase embeddings** (e.g., 1000 common dream phrases).
4. Retrieve top‑3 nearest phrases via cosine similarity.
5. Compute confidence = (top1_sim - top2_sim) / (top1_sim + top2_sim + 1e-8).
6. If confidence > threshold (e.g., 0.3), output the top phrase. Else output "low confidence" and abstain.
7. Maintain a rolling buffer of last 10 predictions; only output when the same phrase appears in at least 3 out of 5 consecutive windows (temporal smoothing).

**Output format:** JSON over WebSocket to a dashboard:
```json
{"timestamp": 123.45, "predicted_text": "I am flying over a city", "confidence": 0.82, "alternatives": ["I am running", "I see water"]}
```

**Test:** Use pre‑recorded DREAM data with simulated streaming to measure latency, accuracy, and abstention rate.

---

### Evaluation Agents (Meeting Challenge Criteria)

#### Agent 4A: Cross‑Subject Generalization

- Run leave‑one‑subject‑out validation.
- Report: mean cosine similarity between predicted and target embeddings on held‑out subjects, standard deviation.
- Also report top‑5 retrieval accuracy from the candidate phrase bank.

#### Agent 4B: Cross‑Session Generalization

- Use first night for training, second night for testing (if multiple sessions exist).
- Report same metrics.

#### Agent 4C: Robustness to Missing Channels / Noise

- Artificially drop 10%, 30%, 50% of EEG channels at test time (set to zero).
- Report performance degradation.
- ZUNA is designed to handle missing channels via its masking mechanism – document that.

#### Agent 4D: Calibration Time Experiment

- Train on increasing amounts of REM data (1 minute, 5 minutes, 10 minutes) per subject.
- Show how performance improves with more calibration data.

#### Agent 4E: Reject Option Curve

- Vary confidence threshold and plot coverage (fraction of windows where system outputs something) vs. accuracy (among outputs).
- Find threshold where accuracy > 80%.

#### Agent 4F: Qualitative Dream Report Comparison

- For each test subject, have the system run on their REM EEG and record top predictions.
- Compare with actual dream reports (after the fact) using BERTScore or human evaluation.
- Produce a table of 10 examples.

---

### Integration & Demo

**Final deliverable:** A live demo that:
1. Reads a pre‑recorded DREAM EEG file (simulating real‑time streaming).
2. Detects REM sleep.
3. Runs the speech decoder and outputs text predictions to a web dashboard.
4. Shows a hypnogram, confidence scores, and top‑3 phrases.
5. Optionally, displays a “dream cloud” of frequently decoded words.

**Checklist before demo:**
- [ ] All dataset cards written.
- [ ] Preprocessing pipeline produces valid epochs.
- [ ] ZUNA fine‑tuning runs without errors and validation cosine similarity > 0.5.
- [ ] REM detector achieves > 0.8 F1 on held‑out data.
- [ ] Real‑time decoder maintains < 500 ms latency per window.
- [ ] All evaluation metrics computed and logged.

---

### What Not To Do

- Do not fine‑tune the entire ZUNA model (it’s too large for our dataset and would overfit). Freeze backbone.
- Do not claim exact word‑level decoding if we only have sentence‑level dream reports. We will be honest: our system decodes semantic content, not precise words.
- Do not ignore the reject option – the challenge explicitly values it.
- Do not pool subjects for cross‑subject evaluation; report per‑subject metrics.
- Do not use validation subjects for any training step.

---

**This AGENTS.md is the single source of truth. All agents must follow it. No deviations without updating this document.**

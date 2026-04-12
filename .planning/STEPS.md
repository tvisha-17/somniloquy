Current state
Phase	Component	Status
1A	EEG preprocessing	✅ Code complete, tested
1B	Dream report alignment	✅ Code complete, tested
1C	Train/val/test splits	✅ Code complete
2A	ZUNA model wrapper	✅ Code complete
2B	Fine-tuning loop	✅ Code complete
3A	REM detector	✅ Code complete
3B	Real-time decoder	✅ Code complete
—	Data actually generated	❌ Pipeline never run
—	ZUNA installed	❌ ModuleNotFoundError
—	Scripts use new modules	❌ Still pointing at old files
Ordered next steps
1. Update the two CLI scripts to use the new modules (5 minutes)
scripts/preprocess_dream.py and scripts/align_reports.py still import from the old src/data/preprocess_dream and src/data/align_reports. They need a one-line import swap to use preprocess_dream_eeg and align_dream_reports respectively, plus the load_reports_csv call instead of per-subject .txt lookup.

2. Install Python dependencies

python3.11 -m pip install sentence-transformers pyyaml torch torchvision
For ZUNA specifically — this is the critical unknown:


python3.11 -m pip install zuna
If that fails (package not on PyPI or requires special access), you have two options:

A — check whether Zyphra published it under a different name or via a private repo
B — swap the ZUNA backbone for a public EEG model like BENDR or replace it with a lightweight CNN baseline — the rest of the pipeline stays identical
3. Run the data pipeline (in order)

# Inspect (optional but generates the dataset card)
python3.11 scripts/inspect_dream.py --config configs/inspect_dream.yaml

# Preprocess all 28 subjects → data/processed/dream/eeg/sub-NNN_epochs.npz
python3.11 scripts/preprocess_dream.py --config configs/preprocess_dream.yaml

# Align reports → data/processed/dream/sub-NNN_target_embeddings.npz
python3.11 scripts/align_reports.py --config configs/align_reports.yaml

# Generate train/val/test splits → data/splits/dream_splits.json
python3.11 scripts/make_splits.py
Expect ~308 EDF files to be processed. With 28 subjects and segments ranging from a few minutes (SO files) to longer (REM/NREM), you'll likely get a few thousand epochs per subject.

4. Train the model

python3.11 scripts/finetune_zuna.py --config configs/finetune_zuna.yaml
The training loop only trains the projection head (ZUNA backbone is frozen), so a GPU is strongly recommended but not required for a first run with small batch size. Target: validation cosine similarity > 0.5 before calling it ready for the demo.

5. Validate data quality before training
After step 3, open one .npz and sanity-check it:


import numpy as np
d = np.load("data/processed/dream/eeg/sub-010_epochs.npz", allow_pickle=True)
print(d["data"].shape)          # expect (n_epochs, n_channels, 512)
print(d["sleep_stages"][:20])   # should contain 4s (REM), 2s (NREM), etc.
print(d["segment_types"][:10])  # "REM", "NREM", "Morning", "SO"
Also check how many subjects have sub-NNN_target_embeddings.npz — only subjects with a matching row in Reports.csv will be trainable. Given the CSV you found has entries for most subjects, coverage should be high, but confirm.

6. Run the demo

python3.11 scripts/demo_realtime.py --config configs/demo_realtime.yaml
This replays a preprocessed subject file through the REM detector and decoder, and serves a FastAPI dashboard at http://127.0.0.1:8000.

The one thing that could block everything
ZUNA availability. If pip install zuna doesn't work, the model wrapper in src/models/zuna_decoder.py won't load. The clean fallback is to replace the backbone with a small 1-D CNN or a public pretrained EEG model. The rest of the pipeline — preprocessing, alignment, training loop, real-time decoder, demo — is entirely independent of which backbone is used. Do you want me to check whether ZUNA is accessible and wire in a fallback if not?
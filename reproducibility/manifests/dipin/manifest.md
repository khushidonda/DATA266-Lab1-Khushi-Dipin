# DATA266 Lab 1 — Reproducibility Manifest

## Member
Name:

## System / Hardware
- Operating system:
- CPU:
- GPU:
- RAM:

## Software Environment
- Python version:
- PyTorch version:
- CUDA version:
- Other important libraries:

Environment file:
`environment.txt`

---

## Task 1 — GPT-Style LLM

### Final Run
- Run ID:
- Config file:
- Raw log:
- Checkpoint:
- Metrics file:
- Output/evidence:
- Date/time:

---

## Task 2 — Sentiment Classification

Machine-readable mapping (run ID → raw log → checkpoint → test metrics) is written by the notebook to
`reproducibility/manifests/dipin/task2_manifest.json`; environment to `environment_task2.txt`.

- Config file: `task2_sentiment/dipin/configs/task2_config.json`
- Notebook (executed): `task2_sentiment/dipin/src/task2_dipin.ipynb`
- Metrics file: `task2_sentiment/dipin/metrics_report.csv`

### Baseline (1-layer LSTM)
- Run ID: see `task2_manifest.json` → runs.baseline
- Raw log: `reproducibility/raw_logs/dipin/task2_baseline_<timestamp>.jsonl`
- Checkpoint: `task2_sentiment/dipin/checkpoints/baseline.pt`

### Experimental Model 1 (2-layer BiLSTM)
- Run ID: see `task2_manifest.json` → runs.experimental_1
- Raw log: `reproducibility/raw_logs/dipin/task2_experimental_1_<timestamp>.jsonl`
- Checkpoint: `task2_sentiment/dipin/checkpoints/experimental_1.pt`

### Experimental Model 2 (2-layer BiLSTM + additive attention)
- Run ID: see `task2_manifest.json` → runs.experimental_2
- Raw log: `reproducibility/raw_logs/dipin/task2_experimental_2_<timestamp>.jsonl`
- Checkpoint: `task2_sentiment/dipin/checkpoints/experimental_2.pt`

---

## Task 3 — CycleGAN

### Final Run
- Run ID:
- Config file:
- Raw log:
- Checkpoint(s):
- Generated outputs:
- Evaluation file:
- Kaggle submission:
- Metrics/evidence:
- Date/time:

# Task 2 (Dipin): how to run

All paths are relative to the repo root. Settings come from `configs/task2_config.json`.

## Setup
```bash
pip install -r task2_sentiment/dipin/requirements.txt
```
The Yelp Polarity dataset (`fancyzhx/yelp_polarity`) and the NLTK stopwords/WordNet data download automatically on the first run.

## Smoke test (about 2 min on a laptop CPU/MPS)
```bash
python task2_sentiment/dipin/src/smoke_test.py
```
This runs the whole pipeline on 4,000 reviews, 1 epoch per model. Output goes to `outputs/smoke/` and `checkpoints/*_smoke.pt` (both git-ignored).

## Full run (GPU Lab)
```bash
cd task2_sentiment/dipin/src
jupyter nbconvert --to notebook --execute --inplace task2_dipin.ipynb --ExecutePreprocessor.timeout=-1
```
This executes the notebook in place, so every output is saved inside `task2_dipin.ipynb`. It produces:

| Output | Location |
|---|---|
| Model weights | `checkpoints/{baseline,experimental_1,experimental_2}.pt` |
| Raw training logs (unedited JSONL) | `outputs/raw_logs/` |
| All metrics, one file | `metrics_report.csv` |
| Full run summary (history, hardware, metrics) | `outputs/run_summary.json` |
| Plots (curves, confusion, ROC/PR/calibration, EDA) | `outputs/plots/`, `outputs/eda/` |
| Test predictions | `outputs/predictions/` |
| 20 error-review candidates | `outputs/error_review/` |
| pip freeze | `outputs/environment.txt` |

Copy `checkpoints/` and `outputs/` off the lab machine as soon as the run finishes (commit and push them).

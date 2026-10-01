# Task 1 (Dipin): how to run

All paths are relative to the repo root. Settings come from `configs/task1_config.json`.

## Setup
```bash
pip install -r task1_llm/dipin/requirements.txt
```
TinyStories (`roneneldan/TinyStories`) downloads automatically on the first run.

## Smoke test (under 1 min on a laptop)
```bash
python task1_llm/dipin/src/smoke_test.py
```
300 stories, 2 epochs, then the full evaluation. Output goes to `outputs/smoke/` and `checkpoints/smoke/` (both git-ignored).

## Full run (GPU Lab)
```bash
cd task1_llm/dipin/src
jupyter nbconvert --to notebook --execute --inplace task1_dipin.ipynb --ExecutePreprocessor.timeout=-1
```
This preprocesses the data, trains for 10 epochs, evaluates, and saves every output inside `task1_dipin.ipynb`.

If the session is interrupted, run the same command again: training resumes from `checkpoints/latest.pt` after the last finished epoch and keeps appending to the same raw log. To train outside the notebook, use `python task1_llm/dipin/src/train.py` and then `python task1_llm/dipin/src/evaluate.py`.

| Output | Location |
|---|---|
| Final weights / best-validation weights | `checkpoints/model.pt`, `checkpoints/best.pt` |
| Raw training log (append-only JSONL) | `reproducibility/raw_logs/dipin/` |
| All metrics, one file | `metrics_report.csv` |
| Loss curves, stability plot | `outputs/loss_curves.png`, `outputs/training_stability.png` |
| Generated samples, failure-case candidates | `outputs/generated_samples.txt`, `outputs/failure_candidates.md` |
| Full evaluation record | `outputs/evaluation.json` |
| Tokenizer, split indices, sequence metadata | `data_processed/` |
| Run manifest and pip freeze | `reproducibility/manifests/dipin/task1_manifest.json`, `environment_task1.txt` |

Commit and push `checkpoints/`, `outputs/`, `data_processed/` and `reproducibility/*/dipin/` before leaving the lab machine.

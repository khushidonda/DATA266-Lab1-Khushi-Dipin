# Task 3 (Dipin): how to run

All commands run from the repo root. Settings come from `configs/task3_config.json`.

## Setup
```bash
python -m venv task3_gan/dipin/.venv
task3_gan/dipin/.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
task3_gan/dipin/.venv/Scripts/python -m pip install -r task3_gan/dipin/requirements.txt
```
Data: extract Kaggle's `gan-getting-started` into `task3_gan/data/dataset/`, so that it contains `monet_jpg/` (300) and `photo_jpg/` (7,038).
The fixed train/test split is committed in `data_processed/splits.json`: 500 test photos and 30 test Monet, seed 266.

## Smoke test (about 2 min)
The smoke test runs the whole pipeline on a small synthetic dataset. Set `TASK3_SMOKE_ROOT` to any folder containing `photo_jpg/` and `monet_jpg/`; all outputs then go to `outputs/smoke/` and `checkpoints/smoke/`, which are git-ignored.
```bash
TASK3_SMOKE_ROOT=/path/to/tiny_data python task3_gan/dipin/src/train.py --smoke
TASK3_SMOKE_ROOT=/path/to/tiny_data python task3_gan/dipin/src/evaluate.py
```

## Full run (RTX 4090: about 2.8 h training, plus about 10 min for evaluation)
```bash
python task3_gan/dipin/src/train.py            # add --resume to continue from checkpoints/latest.pt
python task3_gan/dipin/src/evaluate.py         # FID/KID/P-R/D-C, cycle L1, LPIPS, cosine -> outputs/eval/
python task3_gan/dipin/src/make_submission.py  # images.zip for Kaggle -> outputs/kaggle_submission/
python task3_gan/dipin/src/human_audit.py prepare   # blinded 30-sample audit pack -> outputs/human_audit/
# (two raters fill ratings_rater1.csv / ratings_rater2.csv independently)
python task3_gan/dipin/src/human_audit.py score
cd task3_gan/dipin/src && jupyter nbconvert --to notebook --execute --inplace task3_dipin.ipynb
```

| Output | Location |
|---|---|
| Final weights (one file per network) | `checkpoints/{G_AB,G_BA,D_A,D_B}.pt` |
| Raw training log (unedited JSONL, per 100 steps plus per epoch) | `outputs/logs/task3_dipin_full_*.jsonl` |
| Per-epoch sample grids on fixed test images | `outputs/samples/epoch_XXX.png` |
| Training summary (time, img/s, params, hardware) | `outputs/train_summary.json` |
| Evaluation metrics, per-image CSVs, failure candidates | `outputs/eval/` |
| Held-out translations and reconstructions | `outputs/translations/` |
| Plots | `outputs/plots/` |
| Kaggle submission manifest (zip and generator SHA-256) | `outputs/kaggle_submission/submission_info.json` |
| Human audit pack and results | `outputs/human_audit/` |
| All metrics in one file | `metrics_report.csv` |
| pip freeze | `outputs/environment.txt` |

A = photo, B = Monet. `G_AB` is photo→Monet (the Kaggle direction) and `G_BA` is Monet→photo.

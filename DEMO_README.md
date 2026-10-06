# Demo guide (all paths relative to the repository root)

Run from the repository root with the environment from `requirements.txt`.

## 0. Everything loads (10 s)
```bash
python smoke_test.py
```

## 1. Final checkpoints
| Task | Checkpoint |
|---|---|
| Task 1 (Khushi) | `task1_llm/khushi/checkpoints/task1_full_20260925_204011/epoch_9.pt` |
| Task 2 (Dipin) | `task2_sentiment/dipin/checkpoints/experimental_2.pt` (best), `baseline.pt`, `experimental_1.pt` |
| Task 3 (Khushi) | `task3_gan/khushi/checkpoints/task3_khushi_prod_20260930_154246/epoch_28.pt` (completed epoch 29; copy it from the submission package, it is not in Git) |
| Task 3 (Dipin) | `task3_gan/dipin/checkpoints/v2/{G_AB,G_BA,D_A,D_B}.pt` (final), `task3_gan/dipin/checkpoints/*.pt` (v1 baseline) |

## 2. Executed notebooks
`task1_llm/khushi/task1_khushi.ipynb` · `task2_sentiment/dipin/src/task2_dipin.ipynb` · `task3_gan/khushi/src/task3_khushi.ipynb` · `task3_gan/dipin/src/task3_dipin.ipynb`

## 3. Metrics tables
* Task 1: `task1_llm/khushi/metrics_report.csv`, `task1_llm/khushi/results.md`
* Task 2: `task2_sentiment/dipin/metrics_report.csv`, `task2_sentiment/dipin/results.md`
* Task 3: `task3_gan/khushi/full_metrics_report.csv`, `task3_gan/dipin/metrics_report.csv`
* Architecture / hyperparameter comparison and all comparison tables: `report/DATA266_Lab1_Report_Team_PAIR_01.pdf` (sections 2.2, 3.2 and 4.1)

## 4. Loss curves
* Task 1: `task1_llm/khushi/outputs/task1_full_20260925_204011/curves/loss_curves.png`
* Task 2: `task2_sentiment/dipin/outputs/plots/training_curves.png`
* Task 3 (Khushi): `task3_gan/khushi/outputs/plots/{generator_discriminator_losses,cycle_identity_losses,gradient_norms}.png`
* Task 3 (Dipin): `task3_gan/dipin/outputs/plots/loss_curves.png`, `stability.png`; v1/v2 comparison in the report (section 4.6)

## 5. Sample outputs
* Task 1: `task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/generations.txt`
* Task 2: error review `task2_sentiment/dipin/outputs/error_review/error_candidates_experimental_2.csv`
* Task 3 (Khushi): `task3_gan/khushi/outputs/samples/samples_epoch029.png`, `outputs/final_eval_epoch029/pred_A2B/` and `pred_B2A/` (300 images each), `outputs/human_audit/panels/`
* Task 3 (Dipin): `task3_gan/dipin/outputs/v2/samples/epoch_050.png`, `outputs/translations/`, `outputs/v2/class_kaggle/pred_A2B/`, `pred_B2A/`

## 6. Kaggle
* Submission file: `task3_gan/khushi/outputs/submission.csv` (`1,98.61422779159997,0.40998050570487976`)
* Leaderboard evidence: `task3_gan/khushi/outputs/kaggle/kaggle_leaderboard_rank29_score_-49.5121.png`

## 7. Quick generation with Khushi's Task 3 model
```bash
python task3_gan/khushi/src/evaluate_final.py --data-dir task3_gan/data --out /tmp/demo_eval   # full 300 + 300 evaluation (~15 min on CPU)
```
or, for a few images only, run the sample-generation cells of `task3_gan/khushi/src/task3_khushi.ipynb` (set `DATA266_DATA_DIR` to the folder containing `monet_jpg/` and `photo_jpg/`).

# Task 3 (Khushi) — checkpoint → result map

Only facts re-verified from the restored GPU backup (`BACKUP_MANIFEST.txt`, 2026-10-01) are listed here.
Every value here was read from the artifacts named next to it; nothing was estimated.

## Selected checkpoint
- File: `task3_gan/khushi/checkpoints/task3_khushi_prod_20260930_154246/epoch_28.pt` (0-indexed; = **completed epoch 29**)
- Size: 339,592,307 bytes
- SHA-256: `d3124ab24a6f2201a03673cde5a416d04c78dd44985a3c7d590c48efc7e89d3c`
- Matches: backup manifest SHA-256, the `checkpoint_saved` record in the raw log, and `phase0_sweep/epoch_029/metrics.json`.
- Contents: epoch 28, global_step 204,102 (= 29 x 7,038), 4 model state dicts (G_A2B, G_B2A, D_A, D_B), 3 optimizer states (G, D_A, D_B), 3 scheduler states (last_epoch 29), RNG states. All model keys load strictly with the current `task3_gan/khushi/src` code.
- Parameter counts: G_A2B 11,378,179; G_B2A 11,378,179; D_A 2,764,737; D_B 2,764,737; total 28,285,832.
- Not tracked by Git (340 MB > 100 MB limit; ignored by `.gitignore`).

## Official course evaluator result (N_EVAL = 300, JPEG q95 predictions)
| Quantity | Value | Source |
|---|---|---|
| FID A2B (Monet->Photo) | 95.00897934893692 | `outputs/phase0/sweep/epoch_029/metrics.json` |
| FID B2A (Photo->Monet) | 102.21947623426304 | same |
| MiFID A2B | 0.41308894753456116 | same |
| MiFID B2A | 0.40687206387519836 | same |
| Avg FID (submission) | 98.61422779159997 | `outputs/submission.csv` |
| Avg MiFID (submission) | 0.40998050570487976 | same |

- Executed notebook: `task3_gan/khushi/outputs/official_eval/epoch_029/Part3_Evaluation_Script_epoch029.ipynb` (course notebook, SHA-256 of original `702a1265...cef`; only the four folder-path lines were changed, see `official_eval/README.md`). Its printed cell outputs agree: `[Photo->Monet] FID=102.219 MiFID=0.4069`, `[Monet->Photo] FID=95.009 MiFID=0.4131`.
- `outputs/submission.csv` is byte-identical to `official_eval/epoch_029/submission.csv`.
- Epoch 29 is the best of the 16-checkpoint JPEG sweep (completed epochs 25-40) by average FID and by average MiFID: `outputs/phase0/phase0_summary.csv`. The sweep scored the same 300+300 reference images used for the final result, so this is selection on the evaluator metric itself.

## Training evidence
- Raw logs (unedited copies): `reproducibility/raw_logs/khushi/task3_khushi_prod_20260930_154246.jsonl` (epochs 1-5) and `..._resumed_20260930_164230.jsonl` (resumed from `epoch_4.pt`, epochs 6-40). `epoch_28.pt` is saved in the resumed segment.
- Per-epoch records include mean G/D losses, adversarial, cycle and identity components, gradient norms, epoch seconds, images/s and peak GPU memory.

## Evidence chain
`epoch_28.pt` (completed epoch 29, SHA-256 `d3124ab2...c89d3c`)
-> raw-log `checkpoint_saved` record with the same SHA-256 (`..._resumed_20260930_164230.jsonl`)
-> official metrics JSON (`outputs/phase0/sweep/epoch_029/metrics.json`, `checkpoint_sha256`, `n_eval` 300)
-> executed official notebook (`outputs/official_eval/epoch_029/Part3_Evaluation_Script_epoch029.ipynb`)
-> `outputs/submission.csv` (`1,98.61422779159997,0.40998050570487976`)
-> Kaggle leaderboard screenshot (`outputs/kaggle/kaggle_leaderboard_rank29_score_-49.5121.png`: rank 29, score -49.5121).

## Additional metrics on the same checkpoint
- KID, precision/recall, density/coverage, cycle L1, LPIPS, content cosine: `outputs/final_eval_epoch029/metrics_final.json` (the script asserts the checkpoint SHA-256 above before evaluating; protocol in `evaluation_protocol.md`).
- Human audit (30 samples, 2 raters): `outputs/human_audit/`.
- Training curves and efficiency: `outputs/plots/`, from the raw logs (epochs 1-29).
- Final tables: `task3_gan/khushi/full_metrics_report.csv` (also `metrics_report.csv`).
- Checkpoint selection sweep: `outputs/phase0/phase0_summary.csv` (selection used the same fixed in-domain reference set as the final score).

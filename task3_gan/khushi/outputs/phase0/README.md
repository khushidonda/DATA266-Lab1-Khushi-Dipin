# Checkpoint-selection sweep (completed epochs 25-40)

For each checkpoint (`epoch_24.pt` ... `epoch_39.pt`) the 300 + 300 predictions were generated with `src/generate.py` (JPEG quality 95, `eval()`, no random operations) and scored with the unmodified course evaluator (`N_EVAL = 300`). `phase0_summary.csv` lists the results (`epoch` = completed epoch = checkpoint index + 1); `sweep/epoch_NNN/` holds each checkpoint's `metrics.json` and `submission.csv`. A lossless PNG variant of completed epoch 29 (`png_epoch_029/`) scored worse (average FID 101.019), so JPEG q95 was kept. Completed epoch 29 (`epoch_28.pt`) has the lowest average FID and the lowest average MiFID.

The sweep scored the same fixed in-domain reference set as the final score, so this is model selection on the evaluation set. Machine-specific path strings in the `metrics.json` files were redacted to `<HOME>`.

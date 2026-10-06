# Path redaction note

Machine-specific user-directory prefixes (the Windows home directories of the lab machines) were replaced by `<HOME>` in a small number of derived or console files so that the repository contains no personal file paths:

- Task 3 (Khushi): the executed official-evaluator notebook copy and the per-checkpoint `metrics.json` files under `task3_gan/khushi/outputs/official_eval/` and `outputs/phase0/`.
- Task 2 / Task 3 (Dipin): `task2_sentiment/dipin/outputs/full_run_console.log`, `task3_gan/dipin/outputs/postprocess_console.log`, `task3_gan/dipin/outputs/v2/postprocess_console.log` and one output line of `task3_gan/dipin/outputs/class_kaggle/class_eval_run.ipynb`; the folder assignments in the source cells of `task3_gan/dipin/outputs/v2/class_kaggle/class_eval_run_v2.ipynb` were made repository-relative.

Only path strings changed. Code, parameters and numeric outputs are unchanged. The unedited JSONL raw training logs under `reproducibility/raw_logs/` were not modified. The unredacted originals of Khushi's Task 3 evidence are retained in the GPU backup (SHA-256 of the original executed official notebook: `2078f23b223134216b8094df793ad19f674fdc41066ac1c66a7fbf779c9b2092`).

Not modified: twelve early pre-run pipeline-check logs `reproducibility/raw_logs/khushi/task2_*_train_log_20260920_*.jsonl` (preliminary, `starting_configuration_not_final`, not used by any reported result) still contain a local macOS path in their first record; they are raw evidence and were left untouched.

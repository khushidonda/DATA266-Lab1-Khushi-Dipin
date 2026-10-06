# DATA266 Lab 1 — Reproducibility Manifest (Dipin)

## Member
Name: Dipin

## System / Hardware
- Task 2 runs (2026-09-25): Windows 11 (10.0.26200), Intel Core Ultra 9 285K, NVIDIA GeForce RTX 5090 (34.19 GB as reported by PyTorch)
- Task 3 runs (2026-09-29): Windows 11 (10.0.22631), AMD Ryzen CPU (`AMD64 Family 25 Model 97`), NVIDIA GeForce RTX 4090 (25.76 GB as reported by PyTorch)

## Software Environment
- Task 2: Python 3.12.10, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19 — `environment_task2.txt`
- Task 3: Python 3.13.2, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19 — `task3_gan/dipin/outputs/environment.txt`

---

## Task 1 — GPT-Style LLM

Task 1 was led by Khushi; its manifest is `reproducibility/manifests/khushi/manifest.md`. Dipin's Task 1 implementation code is in `task1_llm/dipin/`.

---

## Task 2 — Sentiment Classification

Machine-readable mapping (run ID → raw log → checkpoint → test metrics) is written by the notebook to `reproducibility/manifests/dipin/task2_manifest.json`; environment in `environment_task2.txt`.

- Config file: `task2_sentiment/dipin/configs/task2_config.json`
- Notebook (executed): `task2_sentiment/dipin/src/task2_dipin.ipynb`
- Metrics file: `task2_sentiment/dipin/metrics_report.csv`; run summary: `task2_sentiment/dipin/outputs/run_summary.json`

| Model | Run ID | Raw log | Checkpoint (SHA-256) |
|---|---|---|---|
| Baseline (1-layer LSTM) | `task2_baseline_20260925_163551` | `reproducibility/raw_logs/dipin/task2_baseline_20260925_163551.jsonl` | `task2_sentiment/dipin/checkpoints/baseline.pt` (`e6cba92006f269c9494d5c07951a117c3f5794b135c86ab96348d4fe452179cf`) |
| Experimental 1 (2-layer BiLSTM) | `task2_experimental_1_20260925_163845` | `reproducibility/raw_logs/dipin/task2_experimental_1_20260925_163845.jsonl` | `task2_sentiment/dipin/checkpoints/experimental_1.pt` (`d1fbe6543b594a2403b5a73a69c45060624bce09cb5816641578b7bbf3ec514d`) |
| Experimental 2 (BiLSTM + additive attention) | `task2_experimental_2_20260925_164755` | `reproducibility/raw_logs/dipin/task2_experimental_2_20260925_164755.jsonl` | `task2_sentiment/dipin/checkpoints/experimental_2.pt` (`c3f4da851296e39f3d3f6e83a091527d93f11a65a6e99e3cdcc1eb680ce2639e`) |

---

## Task 3 — CycleGAN

Two runs with the same data split (seed 266, `task3_gan/dipin/data_processed/splits.json`), 50 epochs each. **v2 is the final model**; v1 is the baseline it was compared against. Each version has its own checkpoint, raw log and metrics.

| | v1 (baseline) | v2 (final) |
|---|---|---|
| Config | `task3_gan/dipin/configs/task3_config.json` | `task3_gan/dipin/configs/task3_config_v2.json` |
| Run ID | `task3_dipin_full_20260929_144428` | `task3_dipin_full_20260929_174752` |
| Raw log | `task3_gan/dipin/outputs/logs/task3_dipin_full_20260929_144428.jsonl` | `task3_gan/dipin/outputs/v2/logs/task3_dipin_full_20260929_174752.jsonl` |
| Checkpoints (final epoch 50) | `task3_gan/dipin/checkpoints/{G_AB,G_BA,D_A,D_B}.pt` | `task3_gan/dipin/checkpoints/v2/{G_AB,G_BA,D_A,D_B}.pt` |
| `G_AB.pt` SHA-256 | `c4e2f8cb211b564f2e94c8c5b8bbb699af698babf0ea073b8b2e8cfb145df1e5` | `57df58dce7cce286e11ff3f4b91ad0ad82e26045c97cfc175febf1e3efa7e07c` |
| `G_BA.pt` SHA-256 | `f54270fa9114e690f94a196bfc8db83d4b2ecff3294207214e5e5349a9fca374` | `c00dcc22e32f8cec62ff913f294677cfdbd35035a47042729c36e3727e09dd7f` |
| Evaluation | `task3_gan/dipin/outputs/eval/` | `task3_gan/dipin/outputs/v2/eval/` |
| Class-evaluator predictions and score | `task3_gan/dipin/outputs/class_kaggle/` | `task3_gan/dipin/outputs/v2/class_kaggle/` (also top-level `task3_gan/dipin/submission.csv`) |
| Metrics file | `task3_gan/dipin/metrics_report.csv` (both versions, one `model_version` column) | same file |

The generator hashes equal the `generator_sha256` values in each `predictions_info.json`, which ties the class-evaluator predictions to these checkpoints. Notebooks: `task3_gan/dipin/src/task3_dipin.ipynb` (analysis), `task3_gan/dipin/Part3_Evaluation_Script.ipynb` (class evaluator, v2).

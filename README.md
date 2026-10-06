# DATA266 Lab 1 — LLM Pretraining, Sentiment Classification and CycleGAN Style Transfer

**Team PAIR_01** — Khushi Donda and Dipin
**Repository:** <https://github.com/khushidonda/DATA266-Lab1-Khushi-Dipin>

Khushi led Task 1, Dipin led Task 2, and both team members independently developed and evaluated models for Task 3.

## Quick start (one command)

```bash
git clone https://github.com/khushidonda/DATA266-Lab1-Khushi-Dipin.git
cd DATA266-Lab1-Khushi-Dipin
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python smoke_test.py
```

`python smoke_test.py` (about 10 seconds on a CPU, no dataset needed) loads every final checkpoint with its own model code, performs a strict state-dict load and checks that a forward pass is finite: the Task 1 GPT, the Task 2 BiLSTM + attention classifier, Dipin's final Task 3 generator, and Khushi's Task 3 CycleGAN (when `epoch_28.pt` is present, see below). Exit code 0 means all checks passed.

## Repository layout

```
README.md, DEMO_README.md, requirements.txt, smoke_test.py
task1_llm/      data/ khushi/   (src, configs, data_processed, checkpoints, outputs, results.md, failure_analysis.md, metrics_report.csv, executed notebook)
task2_sentiment/ data/ dipin/   (same structure)
task3_gan/      data/ khushi/ dipin/   (same structure; metrics in full_metrics_report.csv / metrics_report.csv)
reproducibility/   manifests/ (environment + checkpoint-to-result mapping)   raw_logs/ (unedited JSONL training logs)
report/            DATA266_Lab1_Report_Team_PAIR_01.pdf (+ Markdown source, figures, build_report.py)
```

## Dataset (not stored in Git)

The datasets are intentionally **not** committed to this repository.

* **Task 1 — TinyStories** (`roneneldan/TinyStories`, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`): downloaded automatically by `datasets`; the train/validation story split is committed (`task1_llm/khushi/data_processed/split_indices.json`).
* **Task 2 — Yelp Polarity** (`fancyzhx/yelp_polarity`): downloaded automatically on first use; NLTK stopwords/WordNet data are fetched automatically.
* **Task 3 — Monet and photo images** (Kaggle competition "I'm Something of a Painter Myself", `gan-getting-started`): 300 Monet and 7,038 photographs, JPEG, RGB, 256 x 256. Place them as

  ```
  task3_gan/data/monet_jpg/   (300 files)
  task3_gan/data/photo_jpg/   (7038 files)
  ```
  or point any script/notebook to another location with the environment variable `DATA266_DATA_DIR` (the folder that contains `monet_jpg/` and `photo_jpg/`). Dipin's scripts expect the same two folders under `task3_gan/data/dataset/` (see `task3_gan/dipin/RUN.md`).

## Final models and where they are

| Task | Final model | Checkpoint | In Git? |
|---|---|---|---|
| 1 | Khushi, character-level GPT, 574,830 parameters | `task1_llm/khushi/checkpoints/task1_full_20260925_204011/epoch_9.pt` | yes (7 MB) |
| 2 | Dipin, BiLSTM + attention (experimental 2), plus baseline and experimental 1 | `task2_sentiment/dipin/checkpoints/*.pt` | yes |
| 3 | Khushi, CycleGAN, file `epoch_28.pt` = completed epoch 29 | `task3_gan/khushi/checkpoints/task3_khushi_prod_20260930_154246/epoch_28.pt` (340 MB, SHA-256 `d3124ab24a6f2201a03673cde5a416d04c78dd44985a3c7d590c48efc7e89d3c`) | **no** (exceeds GitHub's 100 MB limit); included in the Canvas submission ZIP |
| 3 | Dipin, CycleGAN v2 (final) and v1 (baseline) | `task3_gan/dipin/checkpoints/v2/*.pt`, `task3_gan/dipin/checkpoints/*.pt` | yes |

To use Khushi's Task 3 checkpoint, copy `epoch_28.pt` from the submission package into the path above; `python smoke_test.py` then also verifies its SHA-256 and strict load.

## Results at a glance

* **Task 1** (Khushi): validation cross-entropy 0.8643 nats/char, perplexity 2.373, 1.247 bits/char, top-1 accuracy 72.75% — `task1_llm/khushi/results.md`, `metrics_report.csv`.
* **Task 2** (Dipin): test accuracy 0.9518 / 0.9531 / 0.9533 (baseline / 2-layer BiLSTM / BiLSTM + attention); McNemar p = 0.082 and 0.060 against the baseline — `task2_sentiment/dipin/results.md`, `metrics_report.csv`.
* **Task 3**: Khushi — course-evaluator average FID 98.614, MiFID 0.4100 on a fixed in-domain evaluation set (all 300 Monet + first 300 sorted photos; not a held-out set), KID, precision/recall, cycle L1, LPIPS, content cosine and a 2-rater human audit in `task3_gan/khushi/full_metrics_report.csv` and `results.md`. Dipin — v2 average FID 102.386 (v1 baseline 100.405) under the same course evaluator, with held-out metrics in `task3_gan/dipin/metrics_report.csv` and `results.md`.
* **Kaggle:** team PairProgramming_Team_1, rank 29 at the time of the screenshot, score -49.5121, from Khushi's checkpoint — `task3_gan/khushi/outputs/submission.csv`, `task3_gan/khushi/outputs/kaggle/`.

The combined report is `report/DATA266_Lab1_Report_Team_PAIR_01.pdf`.

## Reproducing the evaluation

```bash
# Task 3 (Khushi): regenerate predictions from the final checkpoint and recompute all extra metrics (CPU, ~15 min)
python task3_gan/khushi/src/evaluate_final.py --data-dir task3_gan/data
python task3_gan/khushi/src/plot_training_curves.py        # curves from the raw logs
python task3_gan/khushi/src/build_metrics_report.py        # rebuilds full_metrics_report.csv / metrics_report.csv
```

Executed notebooks (outputs saved in the files): `task1_llm/khushi/task1_khushi.ipynb`, `task2_sentiment/dipin/src/task2_dipin.ipynb`, `task3_gan/khushi/src/task3_khushi.ipynb`, `task3_gan/dipin/src/task3_dipin.ipynb`. Full training runs are described in each member's `RUN.md` / `results.md` and recorded, step by step, in `reproducibility/raw_logs/`; `reproducibility/manifests/` maps every checkpoint to its result with SHA-256 values and environment records.

## Reproducibility notes

* Runs are configuration-driven (`configs/*.json`); no personal file paths or credentials are used in code, configuration or notebooks.
* Raw training logs are the unedited JSONL files written during the runs.
* Khushi's Task 3 evaluation set is **in-domain**: the CycleGAN was trained on the full image domains, so those metrics measure in-domain translation quality and not held-out generalization; checkpoint selection used the same reference set.

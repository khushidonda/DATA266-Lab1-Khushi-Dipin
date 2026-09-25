# DATA266 Lab 1 — Reproducibility Manifest

## Member
Name: Khushi Donda

## System / Hardware
Task 1 GPU run: lab machine ADS-A51-836-10, 2026-09-25. Full details in `environment.txt`.
- Operating system: Microsoft Windows 11 Pro 10.0.26200 (64-bit); all Task 1 GPU work ran in Docker Desktop 4.91.0 (engine 29.8.0) on WSL2 2.5.9.0 (kernel 6.6.87.2)
- CPU: Intel Core Ultra 9 285K, 24 cores
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, compute capability 12.0; driver 610.60 (WDDM), CUDA UMD 13.3
- RAM: 63.5 GiB on the host; 31.1 GiB available to the Docker VM

## Software Environment
Task 1, inside the Docker image `khushi-task1-gpu:torch2.14-cu130`:
- Python version: 3.12.3
- PyTorch version: 2.14.0+cu130
- CUDA version: 13.0 runtime (`torch.version.cuda`); cuDNN 9.24.0
- Other important libraries: numpy 2.5.2, datasets 5.0.1, huggingface_hub 1.33.0, pyarrow 25.0.1, matplotlib 3.11.2

Environment file:
`environment.txt` (host facts, image digests and the full pip freeze). Image recipe: `task1_gpu.Dockerfile`.

---

## Task 1 — GPT-Style LLM

### Final Run
- Run ID: `task1_full_20260925_204011`
- Config file: `task1_llm/khushi/configs/task1_config.json`
- Raw log: `reproducibility/raw_logs/khushi/task1_full_20260925_204011.jsonl` (100,782 records ending in a normal `run_end`); console output: `reproducibility/raw_logs/khushi/task1_full_20260925_204011_console.log`
- Checkpoint: `task1_llm/khushi/checkpoints/task1_full_20260925_204011/epoch_9.pt` (primary: lowest validation CE; also the final epoch), SHA-256 `5bd9af9bde778b4916696e18ecb435df040f9750037e7fb9918fa59ef04d398b`
- Metrics file: `task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/metrics.json`
- Output/evidence: `task1_llm/khushi/outputs/task1_full_20260925_204011/` (`eval_epoch_9/` generations and failure candidates, `curves/` loss curves and CSVs, `run_summary.json`)
- Date/time: training 2026-09-25 20:40:10–21:25:25 UTC (13:40–14:25 PDT); evaluation 2026-09-25 21:30:36 UTC

### Task 1 provenance

#### Code
| Purpose | Commit / hash |
|---|---|
| Training code used by the run (`model.py`, `train.py`, `run_task1.py`) | `e2160c24d8c7bd05bc0bb99864d42da60326de8a`; the run recorded `git_worktree_dirty = false` |
| Evaluation and plotting code that produced `metrics.json`, the generations and the curves | `234cfc250ea13612b19ec20dd566fe718462182e`; `metrics.json` records `eval_script_sha256 = 30e3f5c85f755f4269876f67bd8b92f114052debf718d8c6d87f9a3ab9543f2d` |
| Presentation-only fixes used afterwards to re-render `training_dynamics.png` and rebuild `failure_candidates.md` (no metric changes) | not committed yet: `evaluate_task1.py` SHA-256 `2bc614ddb33f8721413117fd36ec313930e3ac66ace8319bd710fcc75968b556`, `plot_task1_curves.py` SHA-256 `ff3babc9e0dfd8b5a81ece3469836f49cbf9b740e6c7cfbda8ab1e15fddbe446` |

#### Environment
- Base image: `pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime@sha256:9c99fafa01edfaa3d16da8c209b38b5970bb6fd6e72725ef60efc901489f70c6`
- Built image: `khushi-task1-gpu:torch2.14-cu130`, id `sha256:f1772b5467574f78e175cbc4f67f0e74e81a87ba77d872cf97829cc1b1745ba7` (recipe `task1_gpu.Dockerfile`, packages in `environment.txt`)
- The lab image from `PyTorch.bat` (`gdevakumar/pytorch@sha256:98a489e952cce30f86816be8111f4b800298591ee19258f7d3b54fc8c4ae1fea`, torch 2.1.2+cu121) was not used: it has no sm_120 kernels, so CUDA kernels fail on the RTX 5090.

#### Inputs (SHA-256 recorded by the run in `run_start`; all re-checked against disk on 2026-09-25: match)
| File | SHA-256 |
|---|---|
| `task1_llm/khushi/src/model.py` | `30ceb20ec1fbf0217d9b1fdf95d27022f1289b70fbbfce3239591d1ab10e60a8` |
| `task1_llm/khushi/src/train.py` | `d306a2f4e2ee46f60883b492e3abe4a450e9fca1adc319890fa83cfcffb333fd` |
| `task1_llm/khushi/src/run_task1.py` | `7e46008bdfd5d9ff308a856ec7c7a16a3181f5fde72bfd956231b2d687348a08` |
| `task1_llm/khushi/configs/task1_config.json` | `94df376faebb354824549d8ccc0a2e857bec07a03084035e64ea1ec56b30fe2b` |
| `task1_llm/khushi/data_processed/tokenizer.json` | `c40a2c99d698c5eeabcbe10b1283115c42898f27545b4fe3a9ed56211a8c9d3d` |
| `task1_llm/khushi/data_processed/split_indices.json` | `877de2cfc9cd0c3eb679a3f457b23ccec19d42ba13fb851b8cab9f9ef931cf5d` |
| `task1_llm/khushi/data_processed/sequences_metadata.json` | `2bb86e4d8898b096f808b4e153f0d588ede761ebb1e89bf81054d2aaa73be4df` |
| `task1_llm/khushi/data_processed/train_inputs.npy` (gitignored) | `3f1bd4a6ff192a4ec8f07ef33925ea65210d95776bdd3044dff34ebc06e0c15c` |
| `task1_llm/khushi/data_processed/train_targets.npy` (gitignored) | `dae0bf57eacb81771808def721719f0a8bd40108086cbae54c93d3e3706e1f9c` |
| `task1_llm/khushi/data_processed/validation_inputs.npy` (gitignored) | `829fdf47d5f0344f6acf9d4855a0909ff162b0a3ead6969f7a84f6341629986a` |
| `task1_llm/khushi/data_processed/validation_targets.npy` (gitignored) | `08951838b29e3bf11b85b77e6791f75cc047a6f91eaf221984c15147a9d4755e` |

- Dataset: `roneneldan/TinyStories`, Hugging Face revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, train split of 2,119,719 stories. Tracked split: 100,000 train / 10,000 validation stories (seed 42, no overlap). Vocabulary: 110 characters built from the training split.
- Sequences: 644,854 train and 64,576 validation sequences of 128 characters (non-overlapping 129-character windows per story; target = input shifted by one). Regenerated on this machine with `sequences.py`; `sequences_metadata.json` came out byte-identical to the committed file.

#### Training setup (from `task1_config.json`, recorded in `run_start`)
Seed 42 for random, NumPy and PyTorch (CPU and CUDA); 10 epochs × 10,076 steps = 100,760 optimizer steps; batch size 64; AdamW, learning rate 3e-4, weight decay 0.01; 1,000 linear warmup steps, then cosine decay to 0 at step 100,760; 574,830 parameters; FP32 (matmul precision "highest").

#### Checkpoint → metrics (primary checkpoint `epoch_9.pt`)
| Metric | Value |
|---|---|
| Validation CE (eval mode, all 64,576 sequences) | 0.8642926901 nats/char |
| Training CE (eval mode, all 644,854 sequences) | 0.8638971224 nats/char |
| generalization_gap = validation_eval_ce − training_eval_ce | 0.0003955677 nats |
| Perplexity (validation / training) | 2.3733 / 2.3724 |
| Bits per character (validation / training) | 1.2469 / 1.2463 |
| Top-1 next-character accuracy (validation / training) | 0.72748 / 0.72756 |
| Character Distinct-1 / 2 / 3 (primary) | 0.0051 / 0.0436 / 0.1592 |
| Repeated character 4-gram rate (primary) | 0.1309 |
| Word Distinct-1 / 2 / 3 (secondary) | 0.2395 / 0.6463 / 0.8569 |
| Repeated word 4-gram rate (secondary) | 0.0020 |
| Unseen-word share (secondary; inflated by the fixed 200-character cutoff) | 0.0111 |
| Generation tokens/sec (batch size 1) | 566.4 |
| Training tokens/sec (includes per-step gradient-norm and raw-log overhead) | 307,817 |
| Peak GPU memory during training | 465,645,568 bytes allocated (444.1 MiB); 503,316,480 bytes reserved (480.0 MiB) |
| Total training time | 2,705.1 s |
| NaN / Inf count | 0 |
| Logged final-epoch training CE (dropout on; training-dynamics statistic) | 0.9404 |

Generation protocol: prompts "Once upon a time", "One day,", "Lily and Ben", "The little bird", "Tom was sad because"; 10 samples each (50 in total), 200 new characters, temperature 0.8, sampling seed 42.

#### All epoch checkpoints (`task1_llm/khushi/checkpoints/task1_full_20260925_204011/`, 7,049,223 bytes each)
| Epoch | Logged validation CE | SHA-256 |
|---|---|---|
| 0 | 1.0431043364 | `36e0b1aebfbcb71b3f5b1c61dadcaf406ecc4743a3ef52c3c7cedf22f52c8203` |
| 1 | 0.9534726402 | `ab1cd61e1f86fe3939e08726f0cf00a51ee656c4bba5142ddfc521a982901a6e` |
| 2 | 0.9183389272 | `174177083cbddebf9f3970a45cbd364f2e83f429e93be506f3052162b01f0369` |
| 3 | 0.8985053413 | `0ff63bc42beff35b7ac619b3380d6f7923c039ac8908fbf8c90d943651c5e44e` |
| 4 | 0.8854493341 | `4b6ee8b517e64351ff6ce3a9cc49a6c20c7d54ac14f62c0e6d6f5408fa3b2e1c` |
| 5 | 0.8767458223 | `65dd9b6abc8d032b50a7e2048dd134ddbf619e4fc93bdf923c3456f195b38a04` |
| 6 | 0.8708259131 | `58233bbc0eba8d62b3e943536bc179ae0297ae6f2fdc41378354d9b478777505` |
| 7 | 0.8661954785 | `8fa823752677a7cd00c57261764176e92f92c6139f6665e3d19a90d266dd0b2c` |
| 8 | 0.8646075290 | `be424ea55df4d608f322b3d7c3c03e6706485cac09d8329413bb9bd0305759f6` |
| 9 | 0.8642926901 | `5bd9af9bde778b4916696e18ecb435df040f9750037e7fb9918fa59ef04d398b` |

#### Logs and outputs (SHA-256)
| File | SHA-256 |
|---|---|
| `reproducibility/raw_logs/khushi/task1_full_20260925_204011.jsonl` | `aa65db75d5b96f08bbb43cd55760420fc09d357631661465324933fdd61109b4` |
| `reproducibility/raw_logs/khushi/task1_full_20260925_204011_console.log` | `d37066c3d48b03d9ccb64333cf532f20215c5e7d9d6c496f04379245f722328e` |
| `reproducibility/raw_logs/khushi/task1_smoke_20260925_203044.jsonl` (GPU smoke test, 100 steps) | `edd2ebf02229bee55ccb34dc2e66940b485779f060849db8f8eece0c57cb3d3a` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/run_summary.json` | `52ae539980bfb9f3688ea2025a06feae20cfaf8e8a203e6d6a6c50cce7f692c6` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/metrics.json` | `70b5dc46193f3eee57c6be87255934e1556f5c725c3c8f33285a6a0e154c35c1` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/generations.jsonl` | `034f09b8ca2140d6af549c5a7928cd4aa8dd294b49f4f7e7016b7d52b34e6446` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/generations.txt` | `925b27a5af5637ef93a13be3215fc1f7ba87a2757a59f5797a37532565e503f4` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/failure_candidates.md` (rebuilt with the presentation fixes) | `8218771ac6174e8befd6e428bcd06446bef934bb0099624ccc326028c3423002` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/curves/loss_curves.png` | `4730a6410a639e13e2ef34e697ab31849ab5b2e6279d3de869601c6be6ef4120` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/curves/loss_curves.csv` | `7afdfa2fbcfc7503cc01983bca29af8b994536bfb68617ff647ceb8cd6353edb` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/curves/training_dynamics.png` (re-rendered with the presentation fixes) | `cf23297fea051d60a62b0bcb14fabad6f96fc07f77c63c341addfd4e394f63be` |
| `task1_llm/khushi/outputs/task1_full_20260925_204011/curves/train_steps.csv` | `0aa9b1bbe66e17d03849e3f27d974f9d834de709781f685a283c93de2885fd29` |

#### Commands used (Windows host, repo root; `<repo>` = the clone path)
```powershell
docker build -t khushi-task1-gpu:torch2.14-cu130 -f reproducibility/manifests/khushi/task1_gpu.Dockerfile reproducibility/manifests/khushi
docker volume create tinystories-hf-cache
# Regenerate the gitignored arrays: only sequences.py (split and tokenizer are tracked files)
docker run --rm -e PYTHONUTF8=1 -v "<repo>:/workspace" -v tinystories-hf-cache:/root/.cache/huggingface -w /workspace/task1_llm/khushi/src khushi-task1-gpu:torch2.14-cu130 python -u sequences.py
# GPU smoke test (100 steps) and full run (detached); TASK1_* values are recorded in the raw log
docker run --rm --gpus all -e PYTHONUTF8=1 -e TASK1_DOCKER_IMAGE=... -e TASK1_GIT_COMMIT=... -e TASK1_GIT_DIRTY=... -v "<repo>:/workspace" -w /workspace/task1_llm/khushi/src khushi-task1-gpu:torch2.14-cu130 python -u smoke_test_gpu.py
docker run -d --name task1-full-khushi --gpus all -e PYTHONUTF8=1 -e TASK1_DOCKER_IMAGE=... -e TASK1_GIT_COMMIT=... -e TASK1_GIT_DIRTY=false -v "<repo>:/workspace" -w /workspace/task1_llm/khushi/src khushi-task1-gpu:torch2.14-cu130 python -u run_task1.py --mode full
# Evaluation and plots: repo mounted read-only, only outputs/ writable
docker run --rm --gpus all -e PYTHONUTF8=1 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TASK1_GIT_COMMIT=... -v "<repo>:/workspace:ro" -v "<repo>/task1_llm/khushi/outputs:/workspace/task1_llm/khushi/outputs" -v tinystories-hf-cache:/root/.cache/huggingface -w /workspace/task1_llm/khushi/src khushi-task1-gpu:torch2.14-cu130 python -u evaluate_task1.py --run-id task1_full_20260925_204011
docker run --rm -e PYTHONUTF8=1 -v "<repo>:/workspace:ro" -v "<repo>/task1_llm/khushi/outputs:/workspace/task1_llm/khushi/outputs" -w /workspace/task1_llm/khushi/src khushi-task1-gpu:torch2.14-cu130 python -u plot_task1_curves.py --run-id task1_full_20260925_204011
# Presentation-only fixes afterwards (same mounts): re-render one figure, rebuild the candidate list
python plot_task1_curves.py --run-id task1_full_20260925_204011 --dynamics-only
python evaluate_task1.py --run-id task1_full_20260925_204011 --rebuild-candidates
```

---

## Task 2 — Sentiment Classification

### Baseline
- Run ID:
- Config file:
- Raw log:
- Checkpoint:
- Metrics/evidence:

### Experimental Model 1
- Run ID:
- Config file:
- Raw log:
- Checkpoint:
- Metrics/evidence:

### Experimental Model 2
- Run ID:
- Config file:
- Raw log:
- Checkpoint:
- Metrics/evidence:

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

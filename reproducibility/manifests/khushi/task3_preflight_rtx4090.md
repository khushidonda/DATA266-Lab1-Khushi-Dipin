# Task 3 (Khushi) — RTX 4090 production preflight

This is a preflight only. No training run was started and no predictions were created.
Every number here was measured on the machine listed below during this session.

## When / what
- Date/time: 2026-09-30 22:31 UTC (15:31 PDT)
- Git HEAD audited: `8e11e9a` (in sync with `origin/main`, clean tree). Khushi's Task 3 source is unchanged since `dccea7f`.
- Config: `task3_gan/khushi/configs/task3_config.json`, SHA-256 `0fe611da9b561e2a02450f0bc88b83b13ee321297e34afe761f87644d8a65979`

## Environment
- Machine: lab machine ADS-R15-840-04, AMD Ryzen 9 7950X, Windows 11 Pro 10.0.26200, native (not Docker)
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9; driver 610.60
- Python 3.12.10 (venv); PyTorch 2.14.1+cu126; torchvision 0.29.1+cu126; CUDA runtime 12.6; cuDNN 9.10.02 (91002)
- numpy 2.5.2, pillow 12.3.0
- Evaluator dependencies installed for this preflight (they were missing; nothing already installed was upgraded, and the pip freeze diff shows additions only): scipy 1.18.1, pandas 3.0.6, tqdm 4.70.1 (plus their dependencies colorama, python-dateutil, six, tzdata)
- Precision/flags as production runs them: FP32 (no AMP), `cudnn.benchmark=False`, TF32 matmul off / cuDNN TF32 on (PyTorch defaults), DataLoader `num_workers=0`

## Locked design check: PASS
The code matches the locked design: ResNet-9 generator (reflection padding, 2 down / 9 residual / 2 up, InstanceNorm, ReLU, Tanh); 70x70 PatchGAN with InstanceNorm and a raw output (30x30 map at 256x256); LSGAN; L1 cycle λ=10; L1 identity λ=5; Adam(2e-4, 0.5, 0.999); batch 1; 200 epochs, constant LR for 1–100 then linear decay (epoch 200 multiplier 0.0099); resize 286, random crop 256, horizontal flip; two replay pools of 50 (seeds 42/43); traversal of the larger domain (7038 steps/epoch, 1,407,600 total); seed 42.
Parameter counts: G_A2B 11,378,179; G_B2A 11,378,179; D_A 2,764,737; D_B 2,764,737; total 28,285,832.

## Dataset: PASS
- monet_jpg: 300 files, all .jpg, RGB, 256x256, 0 corrupt (PIL verify + full decode)
- photo_jpg: 7038 files, all .jpg, RGB, 256x256, 0 corrupt
- The dataset stays outside Git.

## Official evaluator: PASS
- Course-provided `Part3_Evaluation_Script.ipynb`, kept outside the repo and not modified. SHA-256 `702a1265433bf2f15c7900c83442c626d10ac0918094d093dde8ef82069d4cef` (8,866 bytes).
- Folders: `monet_jpg` (real A), `photo_jpg` (real B), `pred_A2B` (Monet→Photo, generated B), `pred_B2A` (Photo→Monet, generated A).
- File selection: globs `.jpg/.jpeg/.png` (both cases), removes duplicates, sorts by full path, and takes the first `N_EVAL = 300`. `calculate_fid_mifid` then sorts again and truncates both sets to the smaller count.
- Features: torchvision `inception_v3(IMAGENET1K_V1, transform_input=False)` with `fc=Identity` (2048-d); preprocessing Resize(299) → CenterCrop(299) → ImageNet mean/std normalization.
- FID: standard Fréchet distance using `scipy.linalg.sqrtm`, with an eps offset if the result is non-finite and the real part taken if it is complex.
- "MiFID" as supplied: the mean over i of `scipy cosine(real_act[i], gen_act[i])`, paired by sorted index. It is used exactly as provided.
- Pairs compared: B2A = real Monet vs `pred_B2A`; A2B = real Photo vs `pred_A2B`. Submission = the mean of the two directions for both FID and MiFID. The notebook writes `submission.csv` with columns `ID,FID,MiFID` and one row, ID 1.
- PNG: accepted, since `.png` is in the glob list.
- Inception weights cached in the torch hub cache: `inception_v3_google-0cc3c7bd.pth`, 108,949,747 bytes, SHA-256 `0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef`. A CUDA forward pass returns (N, 2048) finite features.

## Direction mapping: PASS (checked on actual tensors)
- Training, checked with forward hooks on one real `train_one_step`. Call order: `G_A2B<-real_A(Monet)`, `G_B2A<-fake_B`, `G_B2A<-real_B(Photo)`, `G_A2B<-fake_A`, `G_B2A<-real_A` (identity), `G_A2B<-real_B` (identity). D_B scores exactly the tensor G_A2B produced and scores real photos; D_A scores exactly the tensor G_B2A produced and scores real Monet. So G_A2B learns Monet→Photo and G_B2A learns Photo→Monet.
- Inference: `generate_both_directions` was run on a saved checkpoint. Every `pred_A2B/<monet_stem>.jpg` exactly matches (mean abs diff 0.0) G_A2B(Monet) re-encoded with the same JPEG settings, and differs from G_B2A(Monet) by about 42 gray levels. Every `pred_B2A/<photo_stem>.jpg` exactly matches G_B2A(Photo). Output filenames equal the source stems.
- The evaluator's first 300 sorted `pred_B2A` files are exactly the outputs of the first 300 sorted photos. All stems are 10-character hex, so sorting is unambiguous.

## Input/output pipeline: PASS
- Disk JPEG → PIL `.convert("RGB")` → float32 /255 → ×2−1 → HWC→CHW → (1,3,256,256) in [−1,1]. Channel 0 is R, so RGB is not swapped to BGR. Resizing 256→256 changes nothing.
- Normalizing then denormalizing reproduces the source pixels bit for bit, so nothing is normalized twice.
- `generator.eval()` and `torch.no_grad()` are used; there is no random crop or flip; two runs give identical outputs. Train and eval mode outputs are also identical (max diff 0.0), because InstanceNorm has affine=False and no running stats, and there is no dropout.
- Output is (1,3,256,256), finite, and inside [−1,1]; saved as an RGB 256x256 image.
- Output format today: JPEG quality 95 with Pillow's default 4:2:0 chroma subsampling. Lossless PNG round-trips exactly. For comparison, re-encoding real dataset images at q95 gives a mean abs error of 0.62 gray levels, and the real dataset JPEGs are also 4:2:0.

## CUDA smoke test: 26/26 PASS
Covers generator/discriminator forward and backward; D frozen during the G update (0 D params with grad); D updates after unfreezing; finite gradients; replay pools; independent Monet resampling (19 distinct Monet images in 20 draws for the same photo index); every photo visited exactly once per epoch (7038/7038); NaN/Inf detection and the JSONL NaN event; checkpoint save/reload (eval output max diff 0.0); image save; LR boundaries; resume state and LR continuation.
Extra CUDA checks run for this preflight: Adam `exp_avg`/`exp_avg_sq`/`step` restored exactly for G, D_A and D_B; scheduler `last_epoch` restored; epoch/global_step restored; Python/NumPy/torch-CPU/torch-CUDA RNG restored so the next draws are bit-identical. After a CUDA resume, Adam `step` sits on cuda:0; the next 100 steps measured 0.0994 s median, so there is no slowdown.

## RTX 4090 benchmark (exact production path)
Setup: real `build_models/build_optimizers/build_schedulers/build_dataloader`, `ImagePool`, `train_one_step`, real dataset, batch 1, FP32, full G + D_A + D_B update. The same per-step JSONL record as `run_training` was written for every step. CUDA was synchronized around every timed region.

| | train_one_step only | full iteration (data + step + log) |
|---|---|---|
| warmup / measured steps | 20 / 300 | 20 / 300 |
| mean s/step | 0.1021 | 0.1088 |
| median s/step | 0.1002 | 0.1070 |
| p95 s/step | 0.1144 | 0.1215 |
| min / max s/step | 0.0935 / 0.1564 | 0.0995 / 0.1628 |
| steps/s (mean) | 9.79 | 9.19 |

- Throughput: 9.19 batch-1 steps/s end to end. Each step uses one Monet and one photo sample, so that is 18.4 real images/s.
- Peak GPU memory: 2.59 GiB allocated, 2.92 GiB reserved.
- GPU utilization (nvidia-smi, 250 ms samples, n=137): mean 60%, median 63%.
- Loss ranges over the 300 measured steps (steps 20–320 from init):
  - loss_G 5.45–19.61
  - G adv: A2B 0.137–1.216, B2A 0.137–0.864
  - cycle (raw L1): A 0.136–0.689, B 0.123–0.933
  - identity (raw L1): A 0.109–0.654, B 0.109–0.886
  - D_A 0.096–0.822; D_B 0.186–0.997
- Gradient norms: G 22.0–173.8; D_A 3.29–39.58; D_B 2.49–68.41.
- NaN/Inf: 0.
- Other costs: grad-norm helpers take 6.8 ms/step (a `.item()` call per parameter); one checkpoint save takes 0.66 s and 339.6 MB (production writes `epoch_N.pt` plus `latest.pt` each time); the raw log grows by about 858 bytes/step.

### Estimated wall-clock time (7038 steps/epoch, full-iteration timing; checkpoint time is negligible)
| epochs | mean (0.1088 s) | median (0.1070 s) |
|---|---|---|
| 1 | 12.8 min | 12.5 min |
| 5 | 1.06 h | 1.05 h |
| 10 | 2.13 h | 2.09 h |
| 25 | 5.32 h | 5.23 h |
| 50 | 10.64 h | 10.46 h |
| 100 | 21.27 h | 20.91 h |
| 200 | 42.55 h | 41.82 h |

## Checkpoint / resume / logging
- Checkpoints hold the 4 models, 3 optimizers, 3 schedulers, epoch, global_step, config and RNG states, and each save is SHA-256 logged. Resume continues the LR schedule and step count.
- Resume writes a new raw log with a fresh run ID. `RawLogger` refuses to reuse an existing log file, and records are append-only JSONL.
- Checkpoint files are named by 0-indexed epoch (`epoch_4.pt` = epoch 5). The CLI does not expose `checkpoint_every_epochs`, so production saves every epoch. That covers every 5 epochs; the full run is about 68 GB and the disk has 2.4 TB free.
- Resume is not bit-exact: replay-pool contents and the dataset's own per-process RNGs are not checkpointed. After a resume, the pools refill within 50 steps and the Monet sampling/crop stream restarts from the seed.
- Resuming from a checkpoint older than the latest into the same run ID would silently overwrite later `epoch_N.pt` files. Resume from `latest.pt` or pass a new `--run-id`.

## Findings (non-model, non-hyperparameter)
- BLOCKER: none.
- SHOULD FIX (decide before the first incremental evidence commit):
  1. The per-step raw log will be about 1.2 GB (858 B × 1,407,600) in the tracked `reproducibility/raw_logs/khushi/`. That is over GitHub's 100 MB per-file limit, so a plan is needed for how it gets committed.
  2. `*.pt` is not gitignored, and every production checkpoint (339.6 MB) is over GitHub's 100 MB limit. Stage files explicitly, never with `git add -A`.
  3. Procedure: resume only from `latest.pt`, or use a new `--run-id`, to avoid overwriting later checkpoints.
- OPTIONAL:
  1. Lossless PNG predictions instead of JPEG q95 with 4:2:0 chroma. The evaluator accepts PNG. Whether FID moves up or down is unknown, so compare both formats on the same checkpoint before switching. A pred folder must never hold both .jpg and .png, because the evaluator would count both.
  2. Bit-exact resume: checkpoint the pool contents and dataset RNGs.
  3. Throughput headroom: GPU utilization is about 60%; grad-norm `.item()` loops take about 6.8 ms/step and synchronous data loading about 6.7 ms/step.
  4. Pass `device=cuda` to `generate_both_directions`; it defaults to CPU.
- NO ISSUE: direction mapping, evaluator folders, normalization/denormalization, stochastic inference, train/eval mode, filename order, checkpoint loading, scheduler/resume continuity, NaN stability, source/prediction pairing, evaluator dependencies.

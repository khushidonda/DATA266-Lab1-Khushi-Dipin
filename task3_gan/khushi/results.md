# Task 3 (Khushi) — CycleGAN, Monet <-> Photo

Every number below is read from a committed artifact; the source is named next to each table. Checkpoint file `epoch_28.pt` corresponds to **completed epoch 29** under zero-based checkpoint naming (SHA-256 `d3124ab24a6f2201a03673cde5a416d04c78dd44985a3c7d590c48efc7e89d3c`).

## 1. Objective
Unpaired image-to-image translation between photographs (domain B, 7,038 images) and Monet paintings (domain A, 300 images), 256 x 256 RGB, with a CycleGAN (Zhu et al., 2017) implemented from scratch in PyTorch. No pretrained weights are used in the generators or discriminators; pretrained Inception-v3 and AlexNet are used only to *measure* images.

- `G_A2B`: Monet -> photo. `G_B2A`: photo -> Monet. `D_A` judges Monet images, `D_B` judges photos.

## 2. Architecture
| Network | Design | Parameters |
|---|---|---|
| `G_A2B`, `G_B2A` | ResNet-9 generator: reflection padding, 7x7 stem, 2 stride-2 downsampling convs, 9 residual blocks, 2 upsampling stages (`ConvTranspose2d`), InstanceNorm, ReLU, Tanh output | 11,378,179 each |
| `D_A`, `D_B` | 70x70 PatchGAN, InstanceNorm, raw (no sigmoid) output, 30 x 30 patch map at 256 x 256 | 2,764,737 each |
| Total | | **28,285,832** |

Source: `src/models.py`, `configs/task3_config.json`; parameter counts counted from the loaded checkpoint and equal to the `run_start` record of the raw log.

## 3. Training setup
| Setting | Value |
|---|---|
| Objective | LSGAN (MSE) adversarial loss + 10 x L1 cycle + 5 x L1 identity |
| Optimizer | Adam, lr 2e-4, betas (0.5, 0.999), separate optimizers for G, D_A, D_B |
| Batch size | 1 |
| Augmentation | resize 286 -> random crop 256 -> random horizontal flip, independent per domain |
| Replay pool | 50 generated images per domain |
| Epoch definition | 7,038 steps (one pass over the larger domain); Monet images resampled uniformly at random |
| Schedule | constant lr 2e-4 for epochs 1-100 then linear decay (planned 200 epochs); this run was stopped after epoch 40, so the decay phase was never reached |
| Seed | 42 |
| Hardware | NVIDIA GeForce RTX 4090 (24 GB), FP32, PyTorch + CUDA |
| Run | `task3_khushi_prod_20260930_154246` (two raw-log segments: epochs 1-5 and epochs 6-40 resumed from `epoch_4.pt`) |

Sources: `configs/task3_config.json`, `reproducibility/raw_logs/khushi/task3_khushi_prod_20260930_154246*.jsonl`, `reproducibility/manifests/khushi/task3_preflight_rtx4090.md`.

## 4. Selected checkpoint and why
Checkpoints from completed epochs 25-40 (every epoch, JPEG quality 95, 300 + 300 images) were scored with the course evaluator. Completed epoch 29 had the lowest average FID and the lowest average MiFID:

| Completed epoch | avg FID | | Completed epoch | avg FID |
|---|---|---|---|---|
| 25 | 101.141 | | 33 | 101.125 |
| 26 | 103.041 | | 34 | 100.261 |
| 27 | 102.183 | | 35 | 99.262 |
| 28 | 101.695 | | 36 | 99.766 |
| **29** | **98.614** | | 37 | 99.851 |
| 30 | 102.204 | | 38 | 99.257 |
| 31 | 99.762 | | 39 | 102.176 |
| 32 | 99.115 | | 40 | 100.930 |

Lossless PNG output of the same checkpoint scored worse (average FID about 101.019), so JPEG quality 95 was kept. Source: `outputs/phase0/phase0_summary.csv`.

**Limitation of the selection.** The sweep used the same fixed 300-image in-domain reference as the final score, so this is model selection on the evaluation set. The selected score is therefore optimistic and is not an estimate of generalization to unseen images. Neighbouring epochs differ by up to about 4 FID points, which suggests the epoch-to-epoch noise is of similar size as the gap to the runner-up (epoch 32, 99.115).

## 5. Evaluation protocol
All metrics use one **fixed in-domain evaluation set**: all 300 sorted Monet images and the first 300 sorted photos (the files the course evaluator selects with `N_EVAL = 300`). The generators were trained on these image domains, so the numbers measure in-domain translation quality, not held-out generalization. Full protocol: `outputs/final_eval_epoch029/evaluation_protocol.md`.

## 6. Results (checkpoint `epoch_28.pt`, completed epoch 29)

### 6.1 Official course evaluator (Inception-v3, N_EVAL = 300, JPEG q95)
| Metric | A2B (Monet -> photo) | B2A (photo -> Monet) | Average |
|---|---|---|---|
| FID | 95.009 | 102.219 | **98.614** |
| MiFID | 0.4131 | 0.4069 | **0.4100** |

Exact values: FID 95.00897934893692 / 102.21947623426304 / 98.61422779159997; MiFID 0.41308894753456116 / 0.40687206387519836 / 0.40998050570487976 (`outputs/phase0/sweep/epoch_029/metrics.json`, executed notebook `outputs/official_eval/epoch_029/Part3_Evaluation_Script_epoch029.ipynb`, `outputs/submission.csv`). A CPU recomputation of FID with the same code and regenerated predictions gave 94.989 and 102.207; the official values above are the ones reported.

### 6.2 Distribution and fidelity metrics (`outputs/final_eval_epoch029/metrics_final.json`)
| Metric | A2B | B2A |
|---|---|---|
| KID (unbiased MMD², polynomial kernel, 100 subsets of 100) | 0.01397 ± 0.00217 | 0.00906 ± 0.00179 |
| KID, full 300-vs-300 | 0.01410 | 0.00914 |
| Precision (k = 3) | 0.717 | 0.400 |
| Recall (k = 3) | 0.390 | 0.613 |
| Density (k = 5, supplementary) | 1.289 | 0.392 |
| Coverage (k = 5, supplementary) | 0.940 | 0.717 |
| Content-preservation cosine (Inception features, input vs translation) | 0.796 | 0.771 |

KID kernel: degree 3, gamma 1/2048, coef0 1, subset seed 42. Precision/recall: k-NN manifolds (Kynkaanniemi et al., 2019) on the 2048-d Inception features.

### 6.3 Cycle consistency and perceptual distance (raw tensors, [0, 1] pixel scale)
| Metric | A (Monet -> photo -> Monet) | B (photo -> Monet -> photo) | Overall |
|---|---|---|---|
| Cycle-reconstruction L1 | 0.0421 | 0.0513 | 0.0467 |
| LPIPS (AlexNet), input vs cycle reconstruction | 0.2419 | 0.1912 | 0.2165 |
| LPIPS, input vs direct translation (supplementary; style change is intended) | 0.3678 | 0.3903 | 0.3790 |

### 6.4 Human audit (`outputs/human_audit/`)
30 fixed samples (15 A2B + 15 B2A, drawn with seed 42 without replacement from the evaluation set, shuffled, blinded: input | translation only), two independent raters, 1-5 scale.

| Criterion | Rater 1 | Rater 2 | Combined (mean ± SD) | Quadratic-weighted kappa | Exact agreement | Mean abs. difference |
|---|---|---|---|---|---|---|
| Style quality (higher better) | 4.00 ± 0.91 | 2.97 ± 0.85 | 3.48 ± 1.02 | 0.130 | 33.3% | 1.10 |
| Content preservation (higher better) | 4.57 ± 0.57 | 4.07 ± 0.94 | 4.32 ± 0.81 | 0.181 | 50.0% | 0.70 |
| Artifact severity (1 = none, 5 = severe) | 1.60 ± 0.62 | 2.43 ± 1.04 | 2.02 ± 0.95 | -0.025 | 36.7% | 1.03 |

Both raters agree that content is preserved better than style is converted. Agreement between the raters is low (kappa 0.13, 0.18 and -0.03), so the combined means should be read as a blend of two different rating scales rather than a precise consensus. The main difference is on photo -> Monet artifacts: rater 1 averaged 1.47 and rater 2 averaged 3.20 over the 15 B2A samples, while for Monet -> photo the raters were close (1.73 and 1.67).

### 6.5 Training efficiency (`reproducibility/raw_logs/khushi/`, `outputs/plots/training_summary.json`)
| Quantity | Value |
|---|---|
| Training time through completed epoch 29 | 20,946 s (about 5.8 h), 724.5 s for epoch 29 |
| Throughput | 9.71 iterations/s in epoch 29 (9.75 mean over epochs 1-29); one iteration processes one photo and one Monet image |
| Peak GPU memory | 2,894,273,536 bytes (2.70 GiB, `torch.cuda.max_memory_allocated`) |
| Global step at the selected checkpoint | 204,102 (= 29 x 7,038) |

### 6.6 Training stability (`outputs/plots/`)
Epoch-29 means: generator total 3.402 (adversarial 1.035 + 10 x cycle 0.170 + 5 x identity 0.134); D_A 0.116; D_B 0.177; cycle A/B 0.074/0.096; identity A/B 0.061/0.073; gradient norm G 19.6, D_A 6.4, D_B 5.3. Recorded NaN/Inf abort events: 0 (the logger aborts and writes a `nan_detected` record on a non-finite value; none was written).

The generator loss and both cycle and identity losses decrease smoothly over epochs 1-29 and the gradient norms level off (`generator_discriminator_losses.png`, `cycle_identity_losses.png`, `gradient_norms.png`). D_A fell from 0.229 to about 0.082 at epoch 26 and rose to 0.116 by epoch 29, i.e. the Monet discriminator became somewhat less confident over the last three epochs; D_B stayed near 0.177 throughout. This is a moderate change and is not by itself evidence of instability, but it is a reason for checking neighbouring checkpoints (done in section 4).

### 6.7 Qualitative results
`outputs/samples/samples_epoch029.png` shows the first eight sorted Monet and photo inputs with their translations; `outputs/human_audit/panels/` shows the 30 blinded audit samples. Translated Monet images become photo-like while keeping scene layout; photos acquire painterly colour and brushwork, with visible texture artifacts on large smooth regions. Concrete cases are analysed in `failure_analysis.md`.

### 6.8 Kaggle
The submitted file `outputs/submission.csv` (`ID,FID,MiFID` / `1,98.61422779159997,0.40998050570487976`) gives the class metric -(FID + MiFID)/2 = -49.5121. The leaderboard screenshot `outputs/kaggle/kaggle_leaderboard_rank29_score_-49.5121.png` shows team PairProgramming_Team_1 at rank 29 with score -49.5121 (banner: previous best -49.8366). The rank is the position at the time of the screenshot.

## 7. Interpretation
- **Monet -> photo (A2B)** reaches the lower FID (95.0 vs 102.2) but a higher KID (0.0140 vs 0.0091), so the two distribution metrics do not rank the directions the same way. With only 300 reference images and 2048-dimensional features the FID covariance is rank-deficient, and the KID subset standard deviation (0.002) is not small next to the gap between directions, so neither ordering should be over-read.
- **Precision/recall are asymmetric.** A2B has relatively high precision and lower recall (0.717 / 0.390): most generated photos fall inside the real-photo manifold, but they cover less of the real-photo variety. B2A shows the opposite tendency (0.400 / 0.613): generated paintings spread over more of the Monet manifold but fewer of them are individually close to a real Monet image. These are descriptive readings of a 300 vs 300 comparison; they are consistent with, but do not prove, a mode-coverage difference between the directions.
- **Cycle consistency** is good in both directions (L1 0.042 / 0.051 on a [0, 1] scale), and content cosine is 0.77-0.80, so scene content is largely preserved. A low cycle loss alone does not prove a faithful translation, which is why LPIPS, content cosine and the human audit are reported next to it.
- **Human ratings** place content preservation (4.32) above style conversion (3.48), with low inter-rater agreement.

## 8. Limitations
- In-domain evaluation: the 300 + 300 images were part of the training domains, and the checkpoint was selected on the same reference set; no held-out generalization claim is made.
- 300 evaluation images per set: FID is biased at this sample size and KID/precision/recall have material sampling uncertainty.
- Training stopped at epoch 40 of a planned 200 and the linear-decay phase was not reached; the selected checkpoint is from the constant-learning-rate phase.
- The human audit has two raters, 30 samples and low agreement; the rating scales differ between raters.
- The Kaggle rank is a team-level snapshot.
- CPU re-evaluation differs from the official GPU-generated score by under 0.02 FID; the official evaluator values are the ones reported.

## 9. Reproducibility
- Executed notebook (loads the checkpoint and the stored artifacts, no retraining): `src/task3_khushi.ipynb`.
- Evaluation scripts: `src/evaluate_final.py` (predictions, KID, precision/recall, cycle L1, LPIPS, content cosine), `src/plot_training_curves.py`, `src/build_metrics_report.py`, `src/make_human_audit_pack.py`; protocol and environment in `outputs/final_eval_epoch029/evaluation_protocol.md` and `environment.txt`; SHA-256 of the final scripts and tables in `outputs/final_eval_epoch029/SHA256SUMS.txt`.
- Checkpoint-to-result chain: `reproducibility/manifests/khushi/task3_checkpoint_result_map.md`; raw training logs: `reproducibility/raw_logs/khushi/task3_khushi_prod_20260930_154246*.jsonl`.
- Final tables: `full_metrics_report.csv` (identical copy `metrics_report.csv`).

# Results: Task 3 (Dipin), CycleGAN photo ↔ Monet

Every number below comes from committed artifacts: the raw training log `outputs/logs/task3_dipin_full_20260929_144428.jsonl`, `outputs/eval/eval_metrics.json`, and the executed notebook `src/task3_dipin.ipynb`. All metrics are collected in `metrics_report.csv`. `RUN.md` has the reproduction steps.

## Model / Approach
Unpaired image-to-image translation between **A = photos** (7,038) and **B = Monet paintings** (300) from the Kaggle *I'm Something of a Painter Myself* data. It is a CycleGAN (Zhu et al., 2017) implemented from scratch in PyTorch, with no pretrained weights anywhere in the generators or discriminators.

- `G_AB`: photo → Monet. `G_BA`: Monet → photo. `D_A` judges photos and `D_B` judges Monet paintings.
- **Data split** (seed 266, `data_processed/splits.json`): 6,538 train / 500 test photos, and 270 train / 30 test Monet. The 530 test images are never shown to the model during training.
- **Unpaired sampling:** each step draws photos and paintings independently and uniformly, so there is no fixed A/B pairing.

## Architecture
| Network | Design | Parameters |
|---|---|---|
| G_AB, G_BA | ResNet generator: c7s1-64, d128, d256, 9 × residual block (256 channels), u128, u64, c7s1-3, tanh. Reflection padding, instance norm | 11,372,931 each |
| D_A, D_B | 70×70 PatchGAN: C64-C128-C256-C512 → 1-channel 30×30 patch map, LeakyReLU 0.2, instance norm (not on C64), no sigmoid (LSGAN) | 2,763,841 each |
| **Total** | | **28,273,544** |

Weights are initialised N(0, 0.02).

## Hyperparameters
| Setting | Value |
|---|---|
| Objective | LSGAN (MSE) adversarial loss + λ_cyc · L1 cycle + λ_id · L1 identity |
| λ_cyc / λ_id | 10 / 5 (identity = 0.5 · λ_cyc, as in the paper for photo↔painting) |
| Optimiser | Adam, lr 2e-4, β = (0.5, 0.999), separate optimisers for G and D |
| Schedule | 50 epochs: 25 at constant lr, then 25 decaying linearly to 0 |
| Epoch | 1,635 steps × batch 4 = one pass over the 6,538 training photos |
| Batch size | 4 (instance norm normalises each image separately, so this is equivalent to batch 1 per sample) |
| Replay buffer | 50 generated images per domain (Shrivastava et al., 2017) |
| Augmentation | resize 286 → random crop 256 → random horizontal flip, independent per domain |
| Precision | bf16 autocast for convolutions; losses, gradient norms and optimizer state in fp32 |
| Total | 81,750 steps, 327,000 training samples per domain |

**Deviations from the paper and why:**
- **Batch size 4 instead of 1 (throughput).** At batch 1, a step on the RTX 4090 is dominated by per-kernel launch overhead under the Windows WDDM driver. Measured throughput was 12.5 img/s at batch 1 vs 18.9 at batch 4 in fp32, and 34 img/s with bf16 at batch 4.
- **50 epochs instead of 200 (compute budget).** The paper's 200 epochs on monet2photo is about 214k samples at batch 1. Our 327k samples is a comparable amount of training.

## Training Setup
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB), AMD CPU, Windows 11.
- **Software:** Python 3.13.2, PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19. Full `pip freeze` in `outputs/environment.txt`.
- **Training time:** 2.61 h of pure training (9,382 s), about 187 s per epoch.
- **Throughput:** 34.85 training img/s (each "image" is one photo plus one Monet sample through all four networks). Inference runs at 351 img/s (G_AB) and 361 img/s (G_BA).
- **Peak GPU memory:** 4.88 GB (`torch.cuda.max_memory_allocated`).
- **Random seed:** 266 for Python, NumPy, torch and the sampler. The resume checkpoint stores all RNG states.

## Evaluation Results

### Quantitative metrics (held-out inputs)
**Protocol.**
- **A2B:** the 500 test photos are translated and compared against all 300 real Monet paintings, the same reference set Kaggle's MiFID uses.
- **B2A:** all 300 paintings are translated and compared against the 500 real test photos. Only 30 Monet are held out, which is too few for FID, so the distribution metrics use all 300. The per-image metrics use only the 30 held-out paintings.
- **Features:** FID, KID and precision/recall use Inception-v3 pool3 features (`torch-fidelity` FID weights). The pretrained networks (Inception, AlexNet-LPIPS) only *measure* images; they never produce or modify outputs.

| Metric | A2B photo→Monet | B2A Monet→photo |
|---|---|---|
| **FID** ↓ | **94.91** | **88.11** |
| FID of untranslated inputs (baseline) | 126.80 | 126.80 |
| **KID** ↓ (mean ± std, 100 subsets of 100) | **0.0143 ± 0.0021** | **0.0167 ± 0.0024** |
| KID of untranslated inputs | 0.0489 | 0.0493 |
| Precision ↑ (k=3) | 0.364 | 0.670 |
| Recall ↑ (k=3) | 0.550 | 0.338 |
| Density ↑ (k=5) | 0.415 | 0.845 |
| Coverage ↑ (k=5) | 0.807 | 0.720 |
| **Cycle-reconstruction L1** ↓ ([0,1] pixels) | **0.0462** (PSNR 24.92 dB) | **0.0603** (PSNR 22.98 dB) |
| **LPIPS** input vs translation | 0.3715 | 0.2635 |
| LPIPS input vs reconstruction ↓ | 0.1577 | 0.2325 |
| **Content cosine** (input vs translation) ↑ | **0.757** | **0.831** |
| Kaggle-style MiFID estimate | 94.91 (memorization distance 0.245 > 0.1, so no penalty) | 88.11 |

The translation cuts FID by about 25% and KID by about 70% relative to untranslated inputs in both directions, so both generators move images toward the target domain.

### Cycle-consistency verification (notebook §6)
| Check | Result |
|---|---|
| Hand-computed cycle loss vs `nn.L1Loss` on held-out batch | 0.29078 vs 0.29078 (identical) |
| Correct composition F(G(a)) vs wrong composition G(G(a)), L1 to input | 0.138 vs 0.263 |
| Cycle L1 on 100 test photos: trained vs randomly initialised generators | 0.047 vs 0.299 |
| Reconstruction closer to input than the translation is | 97.0% of photos, 70.0% of Monet |

The constraint is implemented correctly: each generator is composed with its *inverse*, the loss matches the library, and reconstructions are 6× closer than an untrained pair gives. Only 70% for Monet happens because G_BA often changes a painting very little (translation L1 only 0.088), so the translation is already about as close to the input as the reconstruction.

### Checkpoint selection (`src/checkpoint_sweep.py`, `outputs/eval/checkpoint_sweep.csv`)
All 10 generator snapshots (every 5 epochs) were scored on the held-out protocol above. The selection rule was fixed before looking at the results: lowest KID averaged over both directions, with FID as the tie-breaker. The class-competition metric was not used for selection.

| Epoch | FID A2B | FID B2A | KID A2B | KID B2A | Mean KID |
|---|---|---|---|---|---|
| 5 | 127.22 | 130.14 | 0.0426 | 0.0598 | 0.0512 |
| 10 | 124.48 | 107.36 | 0.0405 | 0.0349 | 0.0377 |
| 15 | 121.48 | 97.90 | 0.0427 | 0.0268 | 0.0348 |
| 20 | 124.80 | 95.18 | 0.0442 | 0.0212 | 0.0327 |
| 25 | 108.38 | 90.43 | 0.0254 | 0.0194 | 0.0224 |
| 30 | 103.12 | 90.98 | 0.0217 | 0.0196 | 0.0207 |
| 35 | 102.24 | 88.43 | 0.0204 | 0.0165 | 0.0184 |
| 40 | 96.71 | 89.56 | 0.0156 | 0.0172 | 0.0164 |
| 45 | 94.90 | 90.43 | 0.0142 | 0.0182 | 0.0162 |
| **50 (selected)** | **94.91** | **88.11** | **0.0143** | **0.0167** | **0.0155** |

The final checkpoint is the best, so the submitted model is unchanged. Photo→Monet quality barely moves during the constant-lr phase (FID about 121–127 up to epoch 20), then improves sharply once the learning rate starts decaying (108 at epoch 25, 95 by epoch 45). The Monet critic's overfitting in the second half therefore did not hurt output quality. Mean KID was still falling at epoch 50, which suggests a longer schedule would help further.

### Training losses and stability (final-epoch means; plots in `outputs/plots/`)
| Quantity | Value |
|---|---|
| Generator total loss | 2.943 (from 11.19 at the first log window) |
| Generator adversarial loss (sum of both directions) | 1.248 |
| Discriminator loss (D_A / D_B) | 0.161 / 0.063 |
| Cycle-consistency loss (unweighted, [-1,1] scale) | 0.126 (from 0.657) |
| Identity loss (unweighted) | 0.086 (from 0.639) |
| Gradient norm G / D | 18.7 / 7.6 (run maximum G 41.8 at step 100) |
| NaN / Inf steps | **0** over 81,750 steps |

### Class Kaggle competition
**Submitted: v2** (see "Improvement experiment: v2" below). The class-provided `Part3_Evaluation_Script.ipynb` was run with only its folder paths changed, plus a SciPy-compatible `sqrtm` call with the same result. It points at `outputs/v2/class_kaggle/pred_A2B` and `pred_B2A` and writes `submission.csv`.

The script names the domains the other way round, so its `pred_A2B` (Monet→photo) is our `G_BA` applied to all 300 paintings, and its `pred_B2A` (photo→Monet) is our `G_AB` applied to the 500 held-out photos. The predictions are raw generator outputs of v2's final (epoch-50) weights, `checkpoints/v2/G_AB.pt` and `G_BA.pt`, and their SHA-256 hashes are in `outputs/v2/class_kaggle/predictions_info.json`. The same script was also run on v1 (`outputs/class_kaggle/`).

| Direction (class naming) | v1 FID | v1 "MiFID" | **v2 FID (submitted)** | **v2 "MiFID" (submitted)** |
|---|---|---|---|---|
| Photo → Monet (B2A) | 102.43 | 0.3976 | 99.90 | 0.3913 |
| Monet → photo (A2B) | 98.38 | 0.4131 | 104.87 | 0.4118 |
| **submission.csv (mean)** | 100.41 | 0.4054 | **102.39** | **0.4016** |

The switch to v2 was the team's choice after comparing the two runs. By our pre-set held-out rule (mean KID) and by the class script's FID, v1 scores better. v2 has the better photo→Monet FID, the lower class-script "MiFID", and visibly stronger Monet style.

The class script uses torchvision's ImageNet Inception with 299-pixel resizing, so its FID differs from ours (94.9 / 88.1). Its "MiFID" is the mean cosine distance between index-paired features, not Kaggle's memorization-penalised FID.

- Calculated class metric from `submission.csv`: -(FID + MiFID)/2 = -51.394 (v2); v1: -50.405.
- No leaderboard record for this model is stored in the repository. The team's leaderboard entry (rank 29, score -49.5121 at the time of the screenshot) was produced by Khushi's Task 3 checkpoint; see `task3_gan/khushi/outputs/kaggle/`.

### Human audit
30 fixed held-out samples (20 photo→Monet, 10 Monet→photo) from the v1 translations were shuffled under anonymous IDs (`outputs/human_audit/`) and rated 1–5 for style, content and artifacts (artifacts: 5 = no visible artifacts) by two independent raters (`ratings_rater1.csv`, `ratings_rater2.csv`). Summary: `outputs/human_audit/audit_summary.json` and `audit_summary.csv`, produced by `src/human_audit_summary.py` (the id-to-image key is not stored in the repository, so per-direction means are not reported).

| Criterion (1-5, higher is better) | Rater 1 | Rater 2 | Combined (mean ± SD) | Quadratic-weighted kappa | Exact agreement | Mean abs. difference |
|---|---|---|---|---|---|---|
| Style | 3.80 ± 0.55 | 3.33 ± 0.88 | 3.57 ± 0.77 | 0.579 | 56.7% | 0.47 |
| Content preservation | 4.67 ± 0.55 | 4.20 ± 0.61 | 4.43 ± 0.62 | 0.462 | 53.3% | 0.47 |
| Artifacts (5 = none) | 3.73 ± 0.64 | 2.97 ± 1.03 | 3.35 ± 0.94 | 0.421 | 43.3% | 0.77 |

Both raters judged content preservation highest. Agreement is moderate (quadratic-weighted kappa 0.42 to 0.58); rater 2 scored style and artifacts lower than rater 1. Rater 2's notes name the same recurring problems as the failure analysis: streak, hatch and grid textures in skies, blotches, and outputs that stay close to the input painting.

## Observations
1. **Convergence.** Cycle and identity losses fall steadily, with the fastest drop in the first 10 epochs and a second, smoother decline once the learning rate starts decaying at epoch 26. Held-out cycle L1 on the four fixed test images per domain falls from 0.25 at epoch 1 to 0.13–0.18 from epoch 15 on, and is noisy after that; four images is a small sample.
2. **Discriminator collapse episodes (main stability finding).** D_B, the Monet critic, lost the ability to separate real from fake three times: epochs 10–11 (steps 16,100–17,700), 13–14 (21,100–22,800), and 26–29 (42,000–46,700). During these, its real-minus-fake output gap fell as low as 0.012 (D(real) ≈ D(fake) ≈ 0.5), and the generators' adversarial loss dropped to about 0.74. **Each episode starts immediately after a single 100-step window with a large discriminator gradient spike:** 21.6 at step 16,000, 26.7 at 21,000, and 32.6 at 41,900, where the typical level is 8–12. These are the three largest D gradient norms of the run after warm-up. So each collapse is triggered by one oversized D update, not by gradual drift. D_B recovered each time without intervention, and nothing diverged. D_A, the photo critic, never collapsed after the first 300 steps. Gradient clipping on D would be a cheap fix.
3. **Discriminator overfitting on 270 paintings.** Between collapses, D_B separates real from fake better and better: from epoch 10 to 50, D(real) rises from 0.67 to 0.86 and D(fake) falls from 0.33 to 0.14. Its loss ends at 0.063, well below the 0.25 equilibrium, and the generator's adversarial loss climbs from 0.90 to 1.25. D_A stays flat at about 0.68 / 0.32 all run. The asymmetry matches the data sizes (270 vs 6,538 training images): D_B is memorising its small real set. The decaying learning rate slows this but does not stop it.
4. **Visual quality.** Photo→Monet reliably produces Monet's palette (lilac and blue shadows, warm highlights) and a brushed surface texture while keeping the scene layout, and works best on textured natural scenes (forests, fields, water). It stylises only moderately: brushstroke-level repainting is limited, which is expected with λ_id = 5. Monet→photo produces plausible skies and contrast but leaves some brush texture, and returns the most realistic Monet works almost unchanged.
5. **Metric asymmetries.** A2B precision is low (0.364) but coverage is high (0.807): the translations spread over the Monet feature manifold, but many fall outside the tight k-NN balls of only 300 real paintings, so precision is pessimistic with such a small reference set. B2A shows the opposite pattern (precision 0.670, recall 0.338): generated photos look realistic but only cover the part of photo space reachable from 300 paintings.

## Improvement experiment: v2 (submitted to Kaggle)
The v1 analysis identified four problems, and v2 changes exactly those four things. Everything else is identical to v1: data split, seed, 50 epochs, learning-rate schedule and batch size. The config is `configs/task3_config_v2.json`, and the outputs are in `outputs/v2/` and `checkpoints/v2/`. The recipe was committed before any v2 results existed.

| v1 finding | v2 change |
|---|---|
| Weak stylisation | Identity weight λ_id 5 → 1 |
| D_B overfits 270 paintings | DiffAugment (colour, translation, cutout; Zhao et al., 2020) on every discriminator input |
| D_B collapses after single D-gradient spikes | Clip D's global gradient norm at 15 |
| Checkerboard artifacts | Nearest ×2 upsample + 3×3 conv instead of `ConvTranspose2d` (same 28,273,544 parameters) |

**Model choice.** Both runs were scored with the same held-out protocol and the pre-set rule (lowest mean KID over both directions). v2's own sweep selects its epoch 40.

| | Mean KID ↓ | FID A2B ↓ | KID A2B | FID B2A ↓ | KID B2A |
|---|---|---|---|---|---|
| **v1, epoch 50 (submitted)** | **0.0155** | 94.91 | **0.0143** | **88.11** | **0.0167** |
| v2, epoch 40 (v2's best) | 0.0196 | 93.45 | 0.0166 | 93.62 | 0.0227 |
| v2, epoch 50 (final) | 0.0217 | **92.03** | 0.0175 | 95.49 | 0.0260 |
| Class script (combined FID / MiFID): v1 | | | 100.41 / 0.4054 | | |
| Class script (combined FID / MiFID): v2 epoch 50 | | | 102.39 / 0.4016 | | |

By the pre-set rule, and by the class script's combined FID, v1 is the better model. **v2 (final epoch 50) was nevertheless chosen as the Kaggle submission,** for its stronger stylisation, better photo→Monet FID and lower class-script "MiFID". The numbers above are reported unchanged either way. Unless marked v2, the evaluation tables earlier in this file describe v1.

**Per-direction metrics (final epochs, v1 → v2):**
- Photo→Monet:
  - FID 94.9 → **92.0**
  - Precision 0.364 → 0.428
  - Recall 0.550 → 0.497
  - LPIPS between input and translation 0.372 → 0.417 (more change)
  - LPIPS between input and reconstruction 0.158 → 0.258 (less faithful cycle)
- Monet→photo:
  - FID 88.1 → 95.5
  - Precision 0.670 → 0.597
  - Recall 0.338 → 0.292
  - Content cosine 0.831 → 0.801
  - LPIPS between input and reconstruction 0.233 → 0.381

**What v2 shows**
1. **Lower λ_id gives stronger style but a less faithful cycle.** v2's photo→Monet outputs look clearly more painted (pointillist texture, pastel palettes), and it has the best photo→Monet FID of any checkpoint in either run. The price is reconstructions that are much less faithful in both directions. Monet→photo overshoots: in the fixed samples, a misty Seine becomes a saturated purple-pink sunset.
2. **DiffAugment and clipping fixed the instability they targeted.** v2 had no discriminator collapse: D_B's smallest real-minus-fake gap was 0.102 vs 0.012 in v1, and its final real/fake scores were 0.757/0.243 instead of 0.857/0.143, so it overfit less. The largest pre-clip D gradient after warm-up was 20.0 vs 32.6. Clipping fired on 82% of steps in epoch 1, 11% by epoch 10, and 0.1% in the final epoch, so it acted as a spike guard, not a permanent lr cut. There were 0 NaN steps.
3. **But DiffAugment on D_A was probably a mistake.** The photo critic has 6,538 real images and did not need augmentation. In v2 it was much less decisive for most of the run (mean real-minus-fake gap 0.235 vs 0.361), which weakened the pressure on Monet→photo realism. It had caught up with v1 by the final epoch (0.660/0.340 vs 0.664/0.336). Together with λ_id = 1 letting G_BA change paintings more (LPIPS 0.306 vs 0.264), this explains why v2 loses exactly in the Monet→photo direction.
4. **The resize-conv upsampler traded one artifact for another.** Checkerboard grids are gone, but v2 produced blob artifacts instead:
   - Epochs 5–10: dark dots at image edges and red spots.
   - From about epoch 15: mostly gone.
   - Final model: a small yellow dot fixed in the **top-left corner of every image**. It persists through the cycle, so both generators maintain it; this is consistent with the instance-norm "droplet" artifact described for StyleGAN.
5. **v2 costs more.** 3.44 h vs 2.61 h, 26.4 vs 34.9 img/s, and 6.60 vs 4.88 GB peak memory, all from DiffAugment and full-resolution upsampling convolutions.

**The next experiment the evidence points to (not run):** keep v2's clipping and use DiffAugment on D_B only, with λ_id around 2–3 and v1's transposed-convolution upsampler. That aims to keep v2's photo→Monet gain without its Monet→photo loss.

## Strengths and Limitations
**Strengths**
- Both directions improve FID and KID substantially over untranslated inputs.
- The cycle constraint is verified numerically three ways.
- Zero NaN/Inf steps. Full per-step stability telemetry (losses, D outputs, gradient norms) is in the raw log.
- Fast and reproducible: fixed seed and split, resumable training, and only 4.9 GB peak memory.

**Limitations and shortcomings of this analysis**
- **One run, one seed.** The collapse episodes and final metrics may vary between seeds. We had no budget for repeats or for tuning λ_id.
- **Small Monet reference set.** FID and precision/recall against only 300 paintings are biased and high-variance, and B2A distribution metrics necessarily include the 270 training paintings. KID (unbiased) is the more trustworthy comparison.
- **Content-cosine and LPIPS reward under-translation.** The best-scoring "content preservation" photos barely changed, so these metrics must be read together with FID/KID and the human audit, not alone.
- **A low cycle loss does not guarantee a faithful translation.** Some saturated sunsets are mapped to the wrong hue (green or yellow), yet reconstruct perfectly. The colour is hidden in a low-amplitude signal ("CycleGAN steganography", Chu et al., 2017). See `failure_analysis.md`.
- **No discriminator regularisation.** DiffAugment or adaptive augmentation on D_B, spectral normalisation, gradient clipping on D (to stop the single-step spikes that trigger collapse), or a smaller identity weight are the obvious next experiments to address D_B's overfitting and instability.

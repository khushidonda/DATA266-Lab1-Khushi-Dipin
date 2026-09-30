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
We ran the class-provided `Part3_Evaluation_Script.ipynb` as a copy (`outputs/class_kaggle/class_eval_run.ipynb`). Only the folder paths were changed, plus a SciPy-compatible `sqrtm` call with the same result. The script names the domains the other way round, so its `pred_A2B` (Monet→photo) is our `G_BA` applied to all 300 paintings, and its `pred_B2A` (photo→Monet) is our `G_AB` applied to the 500 held-out photos. The predictions are raw generator outputs; the generator SHA-256 hashes are in `predictions_info.json`.

| Direction (class naming) | FID | "MiFID" (class script) |
|---|---|---|
| Photo → Monet (B2A) | 102.43 | 0.3976 |
| Monet → photo (A2B) | 98.38 | 0.4131 |
| **submission.csv (mean)** | **100.41** | **0.4054** |

The class script uses torchvision's ImageNet Inception with 299-pixel resizing, so its FID differs from ours (94.9 / 88.1). Its "MiFID" is the mean cosine distance between index-paired features, not Kaggle's memorization-penalised FID.

- Leaderboard public score: *pending (to be recorded after submission)*
- Leaderboard private score: *pending*
- Team rank: *pending*

### Human audit
30 fixed held-out samples (20 photo→Monet, 10 Monet→photo) have been shuffled under anonymous IDs in `outputs/human_audit/`. Each sample is rated 1–5 for style, content and artifacts by two independent raters. Results and inter-rater agreement (Cohen's κ, unweighted and quadratic, plus % exact and within-1 agreement) are computed by `human_audit.py score`. *Pending ratings.*

## Observations
1. **Convergence.** Cycle and identity losses fall steadily, with the fastest drop in the first 10 epochs and a second, smoother decline once the learning rate starts decaying at epoch 26. Held-out cycle L1 on the four fixed test images per domain falls from 0.25 at epoch 1 to 0.13–0.18 from epoch 15 on, and is noisy after that; four images is a small sample.
2. **Discriminator collapse episodes (main stability finding).** D_B, the Monet critic, lost the ability to separate real from fake three times: epochs 10–11 (steps 16,100–17,700), 13–14 (21,100–22,800), and 26–29 (42,000–46,700). During these, its real-minus-fake output gap fell as low as 0.012 (D(real) ≈ D(fake) ≈ 0.5), and the generators' adversarial loss dropped to about 0.74. **Each episode starts immediately after a single 100-step window with a large discriminator gradient spike:** 21.6 at step 16,000, 26.7 at 21,000, and 32.6 at 41,900, where the typical level is 8–12. These are the three largest D gradient norms of the run after warm-up. So each collapse is triggered by one oversized D update, not by gradual drift. D_B recovered each time without intervention, and nothing diverged. D_A, the photo critic, never collapsed after the first 300 steps. Gradient clipping on D would be a cheap fix.
3. **Discriminator overfitting on 270 paintings.** Between collapses, D_B separates real from fake better and better: from epoch 10 to 50, D(real) rises from 0.67 to 0.86 and D(fake) falls from 0.33 to 0.14. Its loss ends at 0.063, well below the 0.25 equilibrium, and the generator's adversarial loss climbs from 0.90 to 1.25. D_A stays flat at about 0.68 / 0.32 all run. The asymmetry matches the data sizes (270 vs 6,538 training images): D_B is memorising its small real set. The decaying learning rate slows this but does not stop it.
4. **Visual quality.** Photo→Monet reliably produces Monet's palette (lilac and blue shadows, warm highlights) and a brushed surface texture while keeping the scene layout, and works best on textured natural scenes (forests, fields, water). It stylises only moderately: brushstroke-level repainting is limited, which is expected with λ_id = 5. Monet→photo produces plausible skies and contrast but leaves some brush texture, and returns the most realistic Monet works almost unchanged.
5. **Metric asymmetries.** A2B precision is low (0.364) but coverage is high (0.807): the translations spread over the Monet feature manifold, but many fall outside the tight k-NN balls of only 300 real paintings, so precision is pessimistic with such a small reference set. B2A shows the opposite pattern (precision 0.670, recall 0.338): generated photos look realistic but only cover the part of photo space reachable from 300 paintings.

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

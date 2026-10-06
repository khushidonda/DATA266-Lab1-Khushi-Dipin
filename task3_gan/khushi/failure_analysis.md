# Failure analysis — Task 3 (Khushi), CycleGAN `epoch_28.pt` (completed epoch 29)

## Overview
Failure cases come from the 30 blinded audit samples (`outputs/human_audit/panels/S01-S30.png`; direction and source files in `outputs/human_audit/audit_mapping_private.csv`), which were drawn at random with seed 42 from the fixed in-domain evaluation set, not hand-picked. The cases below are the samples the two human raters scored worst, checked against the saved panels, and complemented by the per-image metrics in `outputs/final_eval_epoch029/per_image_*.csv`. Rater scores are written as rater 1 / rater 2 (style, content, artifacts; artifacts 1 = none, 5 = severe).

Pattern across the audit: raters scored content preservation (combined mean 4.32) above style conversion (3.48), and the two raters disagreed mainly on photo -> Monet (B2A) artifacts (rater 1 mean 1.47, rater 2 mean 3.20; for Monet -> photo, A2B, 1.73 and 1.67). Failures cluster into three kinds: texture artifacts on smooth regions, loss of detail or contrast, and partial style conversion.

## Cases

### Case 1 — Periodic texture over smooth regions (B2A)
- **Examples:** `S09` (input `050da171c0`, ratings 5,4,1 / 2,2,4), `S27` (input `0466d4cdf9`, 5,5,1 / 2,3,4), `S14` (input `099159901a`, 4,4,3 / 2,2,4).
- **Failure/error type:** repetitive grid or stripe texture artifacts.
- **Observation:** In `S09` the dark cloud mass and the sky are covered by a fine woven or grid-like pattern, and the foreground turns saturated green and red. In `S27` the sky and sea show a faint regular grid and streaks. In `S14` the smooth night sky is replaced by vertical streaks and mottled colour, while the snow foreground and mountain outline are kept. Rater 2 explicitly noted grid or checkerboard texture for these samples.
- **Possible cause (hypothesis, not tested here):** the generator upsamples with `ConvTranspose2d`, which is known to produce checkerboard-like patterns (Odena et al., 2016), and large smooth regions give the generator little structure to anchor on. A resize-convolution upsampling layer or a smoothness loss would be a direct test.

### Case 2 — Loss of detail and contrast (A2B and B2A)
- **Examples:** `S11` (A2B, input `5e357ad790`, 3,3,3 / 2,3,3), `S20` (B2A, input `0529dd48c8`, 5,5,1 / 2,2,4).
- **Failure/error type:** loss of fine detail, darkening or washed-out colour.
- **Observation:** In `S11` (a Monet painting of a willow translated to a photo) the output is darker and blurrier, and the fine foliage structure of the tree is lost. In `S20` (photo -> Monet) the strongly saturated orange terrace flattens to a pale, washed-out pastel and the steam and background become smeared, although the hill silhouette is kept.
- **Possible cause:** a Monet-domain bias towards soft, low-contrast palettes (only 300 Monet training images) and an L1 cycle/identity loss that favours averaged, smooth reconstructions. Not verified beyond these examples.

### Case 3 — Incomplete style conversion (A2B)
- **Example:** `S18` (A2B, input `8044a92484`, 2,4,2 / 1,5,1; rater 2: "basically unchanged").
- **Failure/error type:** under-stylization.
- **Observation:** The input is an abstract, heavily textured Monet painting and the translation is visually very close to it, apart from a slight blue-grey shift in the upper area. Content preservation is rated high (4 and 5) while style conversion is rated 2 and 1. Paintings that already look unlike a typical photograph give the generator little to convert.
- **Possible cause:** weakly photographic target statistics for abstract inputs; the 7,038-photo target domain contains few abstract, high-texture scenes. Not verified beyond this example.

### Case 4 — Direction-specific rater disagreement (B2A artifacts)
- **Examples:** `S09`, `S14`, `S20`, `S25`, `S27`.
- **Failure/error type:** artifact perception differs between raters.
- **Observation:** For these five photo -> Monet samples rater 1 gave artifact severity 1 or 3 and rater 2 gave 4. Averaged over all 15 B2A samples the artifact means differ by 1.73 points. The quadratic-weighted kappa for artifact severity is -0.025, so the artifact scores should be read as two raters with different sensitivity to fine texture, not as a precise consensus.

### Case 5 — Metric view of the same behaviour
- A2B has higher precision and lower recall (0.717 / 0.390): generated photos tend to look realistic but cover less of the real-photo variety. B2A shows the opposite (0.400 / 0.613). Cycle reconstruction is good in both directions (L1 0.042 / 0.051), so the artifacts in Cases 1 and 2 are not cycle-consistency failures: an artifact-laden translation can still be reconstructed well by the inverse generator.

## Summary
The selected model preserves scene content well (cosine 0.77-0.80, cycle L1 about 0.05), and the most visible failures are texture artifacts on smooth regions in photo -> Monet translation, loss of detail or contrast, and weak conversion of already painterly or abstract inputs in Monet -> photo translation. Suggested next steps, none of which were run: replace transposed convolutions with resize-convolutions, add a smoothness or perceptual cycle term, and evaluate on images that were not part of the training domains. All observations are from 30 audited samples and the 300 + 300 in-domain evaluation set.

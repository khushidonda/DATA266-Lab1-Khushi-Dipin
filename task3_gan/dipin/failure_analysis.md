# Failure / Error Analysis: Task 3 (Dipin), CycleGAN photo ↔ Monet

## Overview
Failure candidates were **selected automatically, not hand-picked**. From the 500 held-out test photos, notebook §8 takes the 5 with the lowest content cosine, the 5 with the highest cycle-reconstruction L1, and the 5 with the highest LPIPS change (`outputs/eval/failure_candidates.csv`). We then inspected them visually (`outputs/plots/failures_*.png`) together with the per-epoch sample grids and the 30 held-out Monet translations. Per-image numbers come from `outputs/eval/per_image_A2B.csv` and `per_image_B2A.csv`.

Reference medians over the 500 test photos (photo→Monet): cycle L1 0.042, translation L1 0.134, LPIPS 0.352, content cosine 0.771.

Failures are rare in cycle terms: only 14 of 500 test photos (2.8%) have cycle L1 above 0.1. What they share is **input statistics far from the typical training photo**: near-uniform regions, extreme saturation, or synthetic borders.

## Cases

### Case 1: Low-texture scenes break the cycle
- **Example:** `029854a790` (white salt flat), `eedab28822` (plain blue sky with contrail), `f1032a4bd8` (calm pale sea).
- **Failure/Error type:** cycle-consistency failure and hallucinated texture.
- **Observation:** These have the highest cycle L1 in the test set: 0.242, 0.238 and 0.130, against a median of 0.042. In the translation, G_AB fills the empty areas with vertical streaks and mottled "paint" texture. The reconstruction comes back the **wrong colour**: white becomes tan, bright blue becomes navy, and the pale sea gains a pink band. The salt flat failed the same way at every checkpoint (epoch 5 grid onward), so this is systematic, not a late-training regression. Large uniform regions give G_AB almost no structure to work with, and the Monet set has almost no large white or saturated-blue fields for D_B to anchor on. The generator therefore paints in Monet's dominant warm palette, and G_BA cannot tell which original colour to restore.
- **Possible improvement:** Increase λ_cyc or add a colour-histogram or low-frequency reconstruction term. Train at higher resolution, or with crops that include more context. Oversample low-texture photos.

### Case 2: Saturated sunsets get the wrong hue, yet reconstruct perfectly ("steganography")
- **Example:** `1f252ba827` (orange sunset → acid green/yellow), `45f63c2653` (red sky and silhouetted trees → yellow-green), `ea29af7178` (green aurora → mustard haze).
- **Failure/Error type:** wrong colour mapping, with the cycle loss hiding the error.
- **Observation:** These are among the **largest** translation changes (L1 0.261 / 0.352 / 0.273, LPIPS 0.58–0.70) and low content cosine (0.49–0.63). Yet their cycle L1 is 0.035 / 0.029 / 0.037, at or **below** the median. `45f63c2653` reconstructs better than a typical photo even though its translation is the wrong colour. So the orange and red cannot be visible in the translation, yet G_BA recovers them; the information must be hidden as a low-amplitude, high-frequency signal. This is the "CycleGAN is a master of steganography" effect (Chu et al., 2017). A low cycle loss alone **does not prove a faithful translation**, which is why we report LPIPS, content cosine and the human audit next to it.
- **Possible improvement:** Add small noise or JPEG compression to G_AB's output before G_BA, which destroys hidden low-amplitude signals and forces the colour to be encoded visibly. Add a perceptual cycle loss or a hue-consistency term.

### Case 3: Letterbox borders and other synthetic structure
- **Example:** `338fec22de` (landscape between thick black letterbox bars).
- **Failure/Error type:** out-of-distribution input, structured artifacts.
- **Observation:** This has the highest translation L1 in the gallery (0.359) and LPIPS 0.688. The black bars become yellow-brown horizontal stripe texture that bleeds into the mountains, because nothing in either domain's training data looks like a hard black border. The reconstruction restores the bars (cycle L1 0.081), so again the error is visible only in the translation. Similar small artifacts appear near watermarks and photographer signatures (visible in `1f252ba827`).
- **Possible improvement:** Pre-process by detecting and cropping letterbox bars, or add bordered images to training. A data-cleaning pass over `photo_jpg`.

### Case 4: Streak / checkerboard artifacts in flat skies
- **Example:** Sky regions in the fixed sample grids (railway scene, `outputs/samples/epoch_025.png` and later), `b7126e52cb`, `f1032a4bd8`.
- **Failure/Error type:** upsampling artifact.
- **Observation:** A faint regular grid and vertical streaks appear in smooth areas of photo→Monet outputs from epoch 5 onward and never fully disappear. This is the known checkerboard pattern of stride-2 transposed convolutions (Odena et al., 2016). In textured regions the painting style hides it, but in skies it reads as a canvas-like grid rather than brushwork. The human audit's *artifacts* criterion is meant to capture this.
- **Possible improvement:** Replace `ConvTranspose2d` with nearest-neighbour upsampling followed by a 3×3 convolution.

### Case 5: Under-translation (identity shortcut), and metrics that reward it
- **Example:** Photo→Monet `e54604f159` (lake with bare trees), and Monet→photo `de6f71b00f` (Monet's forest road).
- **Failure/Error type:** too little style transfer.
- **Observation:** `e54604f159` has the highest content cosine in the test set (0.959), with LPIPS only 0.101 and translation L1 0.075: the output is almost the input photo. In the other direction, `de6f71b00f` barely changes (translation L1 0.021, cosine 0.963), because many of Monet's more realistic works already look photographic. The identity loss (λ_id = 5) explicitly rewards leaving such images alone. Across the 30 held-out paintings, Monet→photo changes images far less than photo→Monet does (mean translation L1 0.088 vs 0.148). As a consequence, **content cosine and LPIPS favour doing nothing**, and the "best content preservation" gallery is partly a gallery of weak translations.
- **Possible improvement:** Reduce λ_id to 0.5–2, or decay it during training. Report content metrics only together with FID/KID and human style ratings.

### Case 6: Monet→photo keeps brushwork and occasionally changes content
- **Example:** `7017e6caa1`, `0260d15306` (held-out Monet paintings with the largest changes).
- **Failure/Error type:** incomplete domain transfer, content drift.
- **Observation:** The most-changed paintings get a photographic sky and stronger contrast, but brush-texture remains visible in foliage. In `7017e6caa1`, the pale sea below a tree is turned into dark grey ground, changing the scene's content. `0260d15306` (misty river) becomes a convincing dusk scene, but has a higher cycle error of 0.097 against a held-out Monet mean of 0.060. D_A has 6,538 real photos, so it is a strong critic, but G_BA only ever sees 270 distinct paintings as input, which limits how varied its learned mapping can be. That matches the low B2A recall (0.338).
- **Possible improvement:** More Monet data (other Impressionist paintings as auxiliary domain-B data), stronger augmentation on B, or DiffAugment on the discriminators.

## Summary
- **Common thread.** The model fails where the input is far from the typical training photo: near-uniform regions (Case 1), extreme colour (Case 2), or synthetic borders (Case 3).
- **The key analytical lesson (Case 2).** The cycle-consistency loss can be satisfied by hiding information, so a low cycle loss does not guarantee a faithful translation. Low cycle loss must always be checked against perceptual and human evaluation.
- **The metrics have biases of their own (Case 5).** Content cosine and LPIPS reward under-translation, and the small 300-painting Monet reference set makes FID and precision pessimistic and high-variance.
- **Artifacts are architectural (Case 4)** and would be fixed by resize-convolution upsampling.
- **Training instability** is analysed separately in `results.md`: three one-step discriminator-gradient spikes that collapsed D_B, and D_B overfitting its 270 paintings.

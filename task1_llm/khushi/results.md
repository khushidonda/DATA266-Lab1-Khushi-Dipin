# Results

## Model / Approach
A character-level, decoder-only GPT-style language model trained from scratch on a 100,000-story
subset of `roneneldan/TinyStories`, with a 10,000-story held-out validation set (same seeded split,
zero overlap). Tokenization is a from-scratch character-level vocabulary (110 characters) built only
from the training split. No pretrained embeddings, tokenizers, or language models are used anywhere.

Run ID: `task1_full_20260925_204011` (full 10-epoch GPU run). Trained on an NVIDIA GeForce RTX 5090
inside a Docker container (`khushi-task1-gpu:torch2.14-cu130`, PyTorch 2.14.0+cu130 / CUDA 13.0).

## Architecture
Trainable token embedding + learnable positional embedding → N Transformer blocks (pre-LayerNorm:
`x = x + Attn(LN(x))`, `x = x + FFN(LN(x))`) → final LayerNorm → linear LM head over the 110-character
vocabulary. Multi-head causal self-attention is implemented manually (`CausalSelfAttention` in
`task1_llm/khushi/src/model.py`): explicit Q/K/V linear projections, scaled dot-product scores,
a registered lower-triangular boolean mask applied via `masked_fill(..., -inf)` before softmax, and
manual multi-head reshape/merge. No `nn.MultiheadAttention`, `nn.Transformer`, `nn.TransformerEncoder`,
or any other prebuilt attention/Transformer module is used anywhere in the model.

Feed-forward sub-layer: `Linear(embedding_dim → feed_forward_dim) → GELU → Linear(feed_forward_dim →
embedding_dim) → Dropout`.

## Hyperparameters
(from `task1_llm/khushi/configs/task1_config.json`, the config used by the committed run)

| Group | Value |
|---|---|
| sequence_length | 128 |
| embedding_dim | 128 |
| num_heads | 4 |
| num_transformer_blocks | 4 |
| feed_forward_dim | 256 |
| dropout | 0.1 |
| batch_size | 64 |
| optimizer | AdamW |
| learning_rate | 3e-4 |
| weight_decay | 0.01 |
| warmup_steps | 1000 |
| scheduler | cosine decay (to 0 at the final step) |
| epochs | 10 |
| parameter_count | 574,830 |

## Training Setup
- **Hardware:** NVIDIA GeForce RTX 5090 (32,607 MiB), lab machine `ADS-A51-836-10`; Intel Core Ultra 9
  285K (24 cores), 63.5 GiB host RAM (31.1 GiB available to the Docker VM). Full details:
  `reproducibility/manifests/khushi/environment.txt`.
- **Software / library versions:** Docker image `khushi-task1-gpu:torch2.14-cu130`
  (recipe: `reproducibility/manifests/khushi/task1_gpu.Dockerfile`); Python 3.12.3; PyTorch 2.14.0+cu130;
  CUDA 13.0 runtime; cuDNN 9.24.0; numpy 2.5.2; datasets 5.0.1.
- **Training time:** 2,705.1 s total (10 epochs, ~270 s/epoch average).
- **Random seed:** 42 (Python `random`, NumPy, and PyTorch CPU/CUDA; also the seed for the 100k/10k
  TinyStories split and for generation sampling).

## Evaluation Results
All values from `task1_llm/khushi/outputs/task1_full_20260925_204011/eval_epoch_9/metrics.json`,
primary checkpoint = `epoch_9.pt` (lowest validation CE; also the final epoch).
See `metrics_report.csv` in this folder for the same values in flat form.

| Metric | Validation | Training (eval mode) |
|---|---|---|
| Cross-entropy (nats/char) | 0.8642926901 | 0.8638971224 |
| Perplexity | 2.3733 | 2.3724 |
| Bits per character | 1.2469 | 1.2463 |
| Top-1 next-character accuracy | 72.75% | 72.76% |

- **Generalization gap** (validation eval CE − training eval CE): **+0.0003956 nats** — i.e. essentially
  zero, no meaningful overfitting at epoch 9.
- **Generation diagnostics** (50 samples, 5 prompts × 10 samples, 200 new chars, temperature 0.8, seed 42):
  character Distinct-1/2/3 = 0.0051 / 0.0436 / 0.1592; character repeated-4-gram rate = 0.1309 (primary
  diversity metrics); word Distinct-1/2/3 = 0.2395 / 0.6463 / 0.8569; word repeated-4-gram rate = 0.0020
  (secondary diagnostics).
- **Throughput / resources:** training 307,817 tokens/sec; generation 566.4 tokens/sec (batch size 1);
  peak GPU memory 444.1 MiB allocated / 480.0 MiB reserved; 0 NaN/Inf events over the full run.

### Per-epoch train/validation loss (nats/char)
| Epoch | Train CE | Val CE |
|---|---|---|
| 0 | 1.5118 | 1.0431 |
| 1 | 1.0775 | 0.9535 |
| 2 | 1.0173 | 0.9183 |
| 3 | 0.9887 | 0.8985 |
| 4 | 0.9710 | 0.8854 |
| 5 | 0.9591 | 0.8767 |
| 6 | 0.9507 | 0.8708 |
| 7 | 0.9451 | 0.8662 |
| 8 | 0.9419 | 0.8646 |
| 9 | 0.9404 | 0.8643 |

Loss curves: `task1_llm/khushi/outputs/task1_full_20260925_204011/curves/loss_curves.png` and
`training_dynamics.png`.

## Observations
- Validation cross-entropy decreased monotonically from 1.0431 after epoch 0 to 0.8643 after epoch 9,
  so the model continued improving throughout the full required 10 epochs.
- The full-split eval-mode training CE was 0.863897 and validation CE was 0.864293, producing a very
  small generalization gap of +0.000396 nats/character.
- Validation perplexity was 2.3733, BPC was 1.2469, and top-1 next-character accuracy was 72.75%.
- Training was numerically stable: no NaN/Inf failures occurred in 100,760 optimizer steps.
- Generated samples generally learned the surface structure of short children's stories, including
  names, dialogue, common narrative openings, and simple sentence structure.
- Generation quality still showed weaknesses in repeated phrases, semantic inconsistency,
  invented/unusual words, and loss of narrative coherence.

## Strengths and Limitations

**Strengths**
- Stable optimization across all 10 epochs with no NaN failures.
- Very small train-validation evaluation gap, indicating little evidence of overfitting on this split.
- Manual causal Transformer successfully learned strong character-level next-token structure while
  remaining compact at 574,830 parameters.
- Reproducible run with deterministic seed, fixed split, raw step-level logging, checkpoint hashing,
  and recorded environment information.

**Limitations**
- Sequence length 128 restricts how much story context the model can condition on at once.
- The model is intentionally small, so its capacity for longer-range semantic consistency is limited.
- Character-level modeling can generate plausible-looking but invalid or contextually inappropriate
  words.
- Generated stories sometimes repeat phrases or drift semantically even when local sentence structure
  looks reasonable.
- Only one locked architecture/hyperparameter configuration was evaluated in this run, so these results
  do not establish that the configuration is optimal.

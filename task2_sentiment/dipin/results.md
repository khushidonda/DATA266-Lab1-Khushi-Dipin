# Task 2 Results: Yelp Polarity Sentiment (Dipin)

## Model lineup
| Model | Architecture | Params | Change vs previous |
|---|---|---|---|
| Baseline | Embedding(V, 200) → 1-layer LSTM(128) → hidden state at true length → dropout 0.3 → Linear(128→1) | 8,169,489 | — |
| Experimental 1 | Embedding(V, 200) → 2-layer BiLSTM(128/dir, inter-layer dropout 0.3) → concat final fwd+bwd → dropout → Linear(256→1) | 8,733,841 | direction + depth |
| Experimental 2 | same 2-layer BiLSTM → additive (Bahdanau) attention pooling over non-PAD steps (att dim 128) → dropout → Linear(256→1) | 8,766,865 | pooling only |

### Why these architectures and embeddings
I built the three models as a controlled progression, with each experimental model adding a specific capability to the previous design. The **baseline** is a one-layer unidirectional LSTM (128 units) whose hidden state at the true review length goes to a linear classifier; it is the simplest recurrent model and the reference point. **Experimental 1** changes direction and depth together: a two-layer BiLSTM whose final forward and backward states are concatenated, so the representation can use context on both sides of a word, which matters for phrases such as "X but Y". **Experimental 2** keeps that encoder and changes only the pooling, from final states to masked additive (Bahdanau) attention over the non-padding steps, so the classifier can weight informative tokens instead of relying on the last state; it also lets me read which tokens drove a prediction (used in the error review). All models use 200-dimensional embeddings learned from scratch, because the task forbids pretrained embeddings. I fixed the embedding size (200) and the hidden size (128 per direction) for all three models and did not run an ablation over either, so I cannot say these values are optimal. The 40,002 x 200 embedding table accounts for about 8.0 million of the 8.2 to 8.8 million parameters, so the three architectures differ by only about 0.6 million parameters.

## Preprocessing
Unescape literal `\n` → lowercase → expand negation contractions → strip punctuation → whitespace tokenize → NLTK stopwords removed (negations kept) → WordNet lemmatization (noun, then verb).
Vocabulary built on train only (min freq 3, cap 40k, plus `<PAD>`/`<UNK>`). max_len 200 (covers ~96% of reviews); longer reviews keep the first 150 and the last 50 tokens.
Each preprocessing step addresses something I saw in the data audit. The raw text contains literal `\n` sequences (about half of the reviews), which punctuation stripping would turn into a stray `n` token, so I replace them with spaces first. I expand negation contractions (`didn't` to `did not`) and keep eleven negation words when removing NLTK stopwords (no, not, nor, never, none, nobody, nothing, neither, nowhere, cannot, without), because negations flip sentiment. I used WordNet lemmatization (noun, then verb); the task allows lemmatization or stemming and I did not run a stemming comparison, so I do not claim it is better. `max_len = 200` covers about 96% of reviews and keeps LSTM compute bounded; the 4% that are longer keep the first 150 and last 50 tokens, because Yelp reviews often end with the overall verdict. The vocabulary (frequency at least 3, capped at 40,000) is built on the training split only, and padding is masked out of the LSTM by packing. The split is the same stratified 90/10 split as Khushi's, so validation and test sets are identical across the team. One consequence only became clear in the error review: the standard stopword list also removes contrast and comparison words (*but, than, other, now, before*), and reviews that contain "but" have a higher error rate (5.16%, n = 20,227) than the rest (4.11%, n = 17,773).

## Hyperparameters (shared by all three)
| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 2e-3, weight decay 1e-4 |
| Schedule | 5% linear warm-up, then cosine decay |
| Batch size | 256 |
| Epochs | up to 4, early stopping on val macro-F1 (patience 2), best checkpoint kept |
| Loss | BCEWithLogits (single logit) |
| Regularization | embedding dropout 0.2, dropout 0.3, grad-clip 1.0 |
| Seed | 42 |

## Training setup / hardware
All three models were trained on the same machine: **NVIDIA GeForce RTX 5090 (32 GB)**, **Intel Core Ultra 9 285K**, Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19).

| Model | GPU | CPU | Train time | Train examples/sec | Inference examples/sec | Peak GPU mem | Peak host RAM | Best epoch (of 4) | Best val macro-F1 | Checkpoint | Raw log |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | RTX 5090 | Core Ultra 9 285K | 167.4 s | 12,046 | 86,703 | 577 MiB | 3.66 GiB | 3 | 0.9492 | `checkpoints/baseline.pt` | `reproducibility/raw_logs/dipin/task2_baseline_20260925_163551.jsonl` |
| Experimental 1 | RTX 5090 | Core Ultra 9 285K | 543.1 s | 3,712 | 32,300 | 979 MiB | 3.68 GiB | 3 | 0.9497 | `checkpoints/experimental_1.pt` | `reproducibility/raw_logs/dipin/task2_experimental_1_20260925_163845.jsonl` |
| Experimental 2 | RTX 5090 | Core Ultra 9 285K | 693.3 s | 2,908 | 27,278 | 980 MiB | 3.79 GiB | 3 | 0.9508 | `checkpoints/experimental_2.pt` | `reproducibility/raw_logs/dipin/task2_experimental_2_20260925_164755.jsonl` |

Run IDs → checkpoints → metrics are recorded in `reproducibility/manifests/dipin/task2_manifest.json`.

## Test-set results (official 38k)
Source: `metrics_report.csv`, `outputs/run_summary.json`. Threshold 0.5; ECE with 10 bins.

| Metric | Baseline | Experimental 1 | Experimental 2 |
|---|---|---|---|
| Accuracy | 0.9518 | 0.9531 | 0.9533 |
| Precision (macro) | 0.9518 | 0.9531 | 0.9533 |
| Recall (macro) | 0.9518 | 0.9531 | 0.9533 |
| F1 (macro) | 0.9518 | 0.9531 | 0.9533 |
| Precision / Recall / F1 (micro) | 0.9518 | 0.9531 | 0.9533 |
| Precision (weighted) | 0.9518 | 0.9531 | 0.9533 |
| Recall (weighted) | 0.9518 | 0.9531 | 0.9533 |
| F1 (weighted) | 0.9518 | 0.9531 | 0.9533 |
| ROC-AUC | 0.9890 | 0.9901 | 0.9903 |
| PR-AUC | 0.9889 | 0.9902 | 0.9903 |
| MCC | 0.9036 | 0.9062 | 0.9066 |
| Brier score | 0.0376 | 0.0364 | 0.0360 |
| Expected calibration error | 0.0149 | 0.0157 | 0.0144 |
| Accuracy 95% CI | [0.9496, 0.9539] | [0.9511, 0.9551] | [0.9512, 0.9554] |
| Macro-F1 95% CI | [0.9496, 0.9539] | [0.9511, 0.9551] | [0.9512, 0.9554] |
| MCC 95% CI | [0.8993, 0.9078] | [0.9021, 0.9102] | [0.9023, 0.9108] |

CIs: 1,000 bootstrap resamples of the test set.

### Confusion matrices (rows = true, cols = predicted; negative, positive)
| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Baseline | 18,064 | 936 | 896 | 18,104 |
| Experimental 1 | 18,150 | 850 | 932 | 18,068 |
| Experimental 2 | 18,167 | 833 | 941 | 18,059 |

Plots: `outputs/plots/confusion_*.png`, `outputs/plots/roc_pr_calibration.png`, `outputs/plots/training_curves.png`.

### McNemar (paired)
| Comparison | A right / B wrong | A wrong / B right | p-value |
|---|---|---|---|
| Baseline vs Experimental 1 | 373 | 423 | 0.0824 |
| Baseline vs Experimental 2 | 430 | 488 | 0.0599 |
| Experimental 1 vs Experimental 2 | 431 | 439 | 0.8124 |

### Per-slice robustness
| Slice | n | Baseline macro-F1 | Exp 1 macro-F1 | Exp 2 macro-F1 | Baseline error | Exp 1 error | Exp 2 error |
|---|---|---|---|---|---|---|---|
| all | 38,000 | 0.9518 | 0.9531 | 0.9533 | 4.82% | 4.69% | 4.67% |
| raw length short (≤ 65 words) | 12,776 | 0.9488 | 0.9506 | 0.9492 | 4.97% | 4.80% | 4.93% |
| raw length medium (65–142) | 12,649 | 0.9529 | 0.9534 | 0.9553 | 4.71% | 4.66% | 4.47% |
| raw length long (> 142 words) | 12,575 | 0.9507 | 0.9523 | 0.9524 | 4.78% | 4.61% | 4.60% |
| contains negation | 28,526 | 0.9494 | 0.9506 | 0.9515 | 4.91% | 4.79% | 4.70% |
| contains contrast word | 22,551 | 0.9467 | 0.9484 | 0.9483 | 5.28% | 5.11% | 5.12% |
| truncated (> 200 tokens) | 1,694 | 0.9303 | 0.9361 | 0.9250 | 6.14% | 5.61% | 6.61% |
| high UNK rate (> 10%) | 231 | 0.8442 | 0.8421 | 0.8333 | 13.85% | 13.85% | 14.72% |

## Observations, strengths, limitations
The three models are close. Test accuracy is 0.9518 (baseline), 0.9531 (experimental 1) and 0.9533 (experimental 2), and the paired McNemar test does not separate them at alpha = 0.05: p = 0.0824 for baseline vs experimental 1, p = 0.0599 for baseline vs experimental 2 (borderline) and p = 0.8124 for experimental 1 vs experimental 2. The accuracy gains of 0.0013 and 0.0015 are about the size of the bootstrap interval half-width (about 0.002), so I treat them as small and not statistically significant; the deeper models also cost 3.3 and 4.1 times the training time of the baseline. The BiLSTM models are marginally better in probability quality (Brier score 0.0376, 0.0364, 0.0360; ROC-AUC 0.9890, 0.9901, 0.9903), but expected calibration error does not improve monotonically (0.0149, 0.0157, 0.0144). All three models reach their best validation macro-F1 at epoch 3, while the training loss keeps falling in epoch 4 (to 0.065, 0.062 and 0.056) and the validation loss rises (0.152, 0.151, 0.154), so extra epochs would mainly add overfitting; checkpoint selection on validation macro-F1 handles this. Experimental 2 has the best aggregate numbers but is not best on every slice: on the truncated slice (1,694 reviews) its error is 6.61% against 5.61% for experimental 1, and on the high-unknown-token slice (231 reviews) the models differ by only two reviews (error 13.85% to 14.72%). The 20-error review of experimental 2 (1,774 errors in total, 4.67%) shows that most errors are hard rather than random: 7 of the 20 are mixed reviews, 5 involve text the model cannot read (non-English words, links, accents damaged by my cleaner), 3 are strong sentiment about something other than the business, and 2 are label noise where the model's answer is the sensible one. The nearest-neighbour check of the learned embeddings shows that sentiment was partly learned ("terrible" is near "horrible" and "aweful"; "never" is near "not", "no" and "nor") but not everywhere (the neighbours of "recommend" are unrelated words).

## Future work
The next steps come directly from these errors and limits. (1) Remove *but, than, other, now, before, only, too* from the stopword list, retrain experimental 2 with the same configuration and seed, and test whether the error rate on "but" reviews (now 5.16%) falls without raising it elsewhere, using a paired McNemar test against the current model, as laid out in `failure_analysis.md`. (2) Decode `\uXXXX` escape codes before cleaning and replace links with a single `<url>` token; 879 test reviews (2.3%) contain such codes and their error rate is 7.39%. (3) Flag or separately handle non-English and link-heavy reviews: all five reviewed errors from the high-unknown-token slice (14.72% error for experimental 2) were of this kind. (4) Look at the truncated slice (6.61% error for experimental 2) with a longer `max_len` or a model that reads the middle of long reviews, since head-plus-tail truncation may drop evidence from that part of the text. (5) Because the observed accuracy differences are small relative to the uncertainty in this single-seed comparison, repeating the three runs with several seeds would provide stronger evidence before drawing conclusions about depth or attention; since validation loss rises after epoch 3, stronger regularisation should also be compared instead of more epochs. (6) Two of the 20 reviewed errors are label noise, so a manual audit of a random sample of the test labels would estimate how much of the remaining 4.7% error cannot be reduced by any model.

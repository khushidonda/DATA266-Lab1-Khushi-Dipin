# Task 2 Results: Yelp Polarity Sentiment (Dipin)

## Model lineup
| Model | Architecture | Params | Change vs previous |
|---|---|---|---|
| Baseline | Embedding(V, 200) → 1-layer LSTM(128) → hidden state at true length → dropout 0.3 → Linear(128→1) | 8,169,489 | — |
| Experimental 1 | Embedding(V, 200) → 2-layer BiLSTM(128/dir, inter-layer dropout 0.3) → concat final fwd+bwd → dropout → Linear(256→1) | 8,733,841 | direction + depth |
| Experimental 2 | same 2-layer BiLSTM → additive (Bahdanau) attention pooling over non-PAD steps (att dim 128) → dropout → Linear(256→1) | 8,766,865 | pooling only |

### Why these architectures and embeddings
<!-- Your own justification: why an LSTM baseline, why BiLSTM + depth, why attention pooling, why dim 200, why learned-from-scratch embeddings -->

## Preprocessing
Unescape literal `\n` → lowercase → expand negation contractions → strip punctuation → whitespace tokenize → NLTK stopwords removed (negations kept) → WordNet lemmatization (noun, then verb).
Vocabulary built on train only (min freq 3, cap 40k, plus `<PAD>`/`<UNK>`). max_len 200 (covers ~96% of reviews); longer reviews keep the first 150 and the last 50 tokens.
<!-- Your justification for lemmatization over stemming, max_len, head+tail truncation -->

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
<!-- Your own comparison of the three models -->

## Future work
<!-- Your own -->

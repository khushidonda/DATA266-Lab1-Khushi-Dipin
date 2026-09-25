# Task 2 Results: Yelp Polarity Sentiment (Dipin)

## Model lineup
| Model | Architecture | Params | Change vs previous |
|---|---|---|---|
| Baseline | Embedding(V, 200) → 1-layer LSTM(128) → hidden state at true length → dropout 0.3 → Linear(128→1) | | — |
| Experimental 1 | Embedding(V, 200) → 2-layer BiLSTM(128/dir, inter-layer dropout 0.3) → concat final fwd+bwd → dropout → Linear(256→1) | | direction + depth |
| Experimental 2 | same 2-layer BiLSTM → additive (Bahdanau) attention pooling over non-PAD steps (att dim 128) → dropout → Linear(256→1) | | pooling only |

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
| Model | GPU | CPU | Train time | Examples/sec | Peak GPU mem | Checkpoint | Raw log |
|---|---|---|---|---|---|---|---|
| Baseline | | | | | | `checkpoints/baseline.pt` | |
| Experimental 1 | | | | | | `checkpoints/experimental_1.pt` | |
| Experimental 2 | | | | | | `checkpoints/experimental_2.pt` | |

## Test-set results (official 38k)
<!-- Copy from the notebook / metrics_report.csv: accuracy, P/R/F1 macro/micro/weighted, ROC-AUC, PR-AUC, MCC, Brier, ECE, 95% CIs -->

### McNemar (paired)
<!-- baseline vs exp1, baseline vs exp2 -->

### Per-slice robustness
<!-- macro-F1 / error rate per slice -->

## Observations, strengths, limitations
<!-- Your own comparison of the three models -->

## Future work
<!-- Your own -->

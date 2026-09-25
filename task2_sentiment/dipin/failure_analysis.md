# Task 2: Error Review (Dipin)

Model reviewed: <!-- best model by validation macro-F1, printed in the notebook -->
Source: `outputs/error_review/error_candidates_<model>.csv` (official test set)

Error-type vocabulary (used below): <!-- e.g. define your own labels, such as sarcasm, mixed sentiment, negation scope, label noise, domain-specific term, truncation, OOV-heavy -->

## A. Confident false positives
_true negative, predicted positive with high confidence_

| # | Test idx | P(pos) | Snippet | Error type | Observation |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

## B. Confident false negatives
_true positive, predicted negative with high confidence_

| # | Test idx | P(pos) | Snippet | Error type | Observation |
|---|---|---|---|---|---|
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |

## C. Near-threshold errors
_P(positive) close to 0.5_

| # | Test idx | P(pos) | Snippet | Error type | Observation |
|---|---|---|---|---|---|
| 11 | | | | | |
| 12 | | | | | |
| 13 | | | | | |
| 14 | | | | | |
| 15 | | | | | |

## D. Slice-specific failures
_worst slice: fill in from the notebook output_

| # | Test idx | P(pos) | Snippet | Error type | Observation |
|---|---|---|---|---|---|
| 16 | | | | | |
| 17 | | | | | |
| 18 | | | | | |
| 19 | | | | | |
| 20 | | | | | |

## Patterns across the 20 errors
<!-- your own analysis -->

## Proposed testable fix
- Fix:
- Hypothesis:
- How to test it (metric, slice, expected change):

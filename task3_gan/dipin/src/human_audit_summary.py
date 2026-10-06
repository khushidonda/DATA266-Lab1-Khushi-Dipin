"""Task 3 (Dipin): summary of the completed two-rater human audit (works without the private id->image key).

Reads outputs/human_audit/ratings_rater1.csv and ratings_rater2.csv (criteria style, content, artifacts; 1-5, higher is
better, artifacts: 5 = no visible artifacts) and writes outputs/human_audit/audit_summary.json and audit_summary.csv:
mean and SD per rater, combined mean and SD, quadratic-weighted Cohen's kappa, exact agreement and mean absolute difference.
Usage: python task3_gan/dipin/src/human_audit_summary.py
"""
import csv
import json
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parent.parent / "outputs" / "human_audit"
CRIT = ["style", "content", "artifacts"]


def load(name):
    rows = list(csv.DictReader(open(AUDIT / f"ratings_{name}.csv", newline="", encoding="utf8")))
    out = {}
    for r in rows:
        out[r["sample_id"]] = [int(r[c]) for c in CRIT]   # raises on empty / non-integer cells
    return out


def qwk(x, y, k=5):
    O = np.zeros((k, k))
    for a, b in zip(x, y): O[a - 1, b - 1] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else None


r1, r2 = load("rater1"), load("rater2")
ids = [f"S{i:02d}" for i in range(1, 31)]
assert sorted(r1) == sorted(r2) == ids
res = {"n_samples": 30, "raters": 2, "scale": "1-5, higher is better (artifacts: 5 = no visible artifacts)", "criteria": {}}
for j, c in enumerate(CRIT):
    x = np.array([r1[i][j] for i in ids]); y = np.array([r2[i][j] for i in ids]); both = np.concatenate([x, y])
    res["criteria"][c] = {"rater1_mean": float(x.mean()), "rater1_sd": float(x.std(ddof=1)), "rater2_mean": float(y.mean()), "rater2_sd": float(y.std(ddof=1)),
                          "combined_mean": float(both.mean()), "combined_sd": float(both.std(ddof=1)), "quadratic_weighted_kappa": qwk(x, y),
                          "exact_agreement_pct": float((x == y).mean() * 100), "mean_abs_difference": float(np.abs(x - y).mean())}
json.dump(res, open(AUDIT / "audit_summary.json", "w"), indent=2)
keys = ["rater1_mean", "rater1_sd", "rater2_mean", "rater2_sd", "combined_mean", "combined_sd", "quadratic_weighted_kappa", "exact_agreement_pct", "mean_abs_difference"]
with open(AUDIT / "audit_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["criterion"] + keys)
    for c, v in res["criteria"].items(): w.writerow([c] + [v[k] for k in keys])
print(json.dumps(res, indent=1))

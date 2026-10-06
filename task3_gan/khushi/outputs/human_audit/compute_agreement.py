"""Inter-rater agreement for the Task 3 (Khushi) human audit. Run after BOTH rating sheets are filled in.

Per criterion (style_quality, content_preservation, artifact_severity; ordinal 1..5): mean/SD per rater, combined
mean, quadratic-weighted Cohen's kappa, exact agreement %, mean absolute rater difference.
Writes agreement_results.json and human_audit_summary.csv. SD uses ddof=1.
Usage: python compute_agreement.py   (reads ratings_rater1.csv / ratings_rater2.csv next to this file)
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CRITERIA = ["style_quality_1_to_5", "content_preservation_1_to_5", "artifact_severity_1_to_5"]
LEVELS = [1, 2, 3, 4, 5]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        vals = []
        for c in CRITERIA:
            v = r[c].strip()
            if v == "" or int(v) not in LEVELS:
                raise SystemExit(f"{path.name}: {r['sample_id']} has a missing/invalid '{c}' ({v!r}); fill every cell with 1-5")
            vals.append(int(v))
        out[r["sample_id"]] = vals
    return out


def quadratic_kappa(x, y):
    k = len(LEVELS)
    obs = np.zeros((k, k))
    for a, b in zip(x, y):
        obs[a - 1, b - 1] += 1
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / (k - 1) ** 2
    if (w * exp).sum() == 0:
        return None          # undefined: a rater used a single level for every sample
    return float(1 - (w * obs).sum() / (w * exp).sum())


r1, r2 = load(HERE / "ratings_rater1.csv"), load(HERE / "ratings_rater2.csv")
if set(r1) != set(r2) or len(r1) != 30:
    raise SystemExit("both sheets must contain exactly the same 30 sample_ids")
ids = sorted(r1)
res = {"n_samples": len(ids), "criteria": {}}
for j, c in enumerate(CRITERIA):
    x = np.array([r1[i][j] for i in ids]); y = np.array([r2[i][j] for i in ids])
    res["criteria"][c] = {
        "rater1_mean": float(x.mean()), "rater1_sd": float(x.std(ddof=1)),
        "rater2_mean": float(y.mean()), "rater2_sd": float(y.std(ddof=1)),
        "combined_mean": float(np.concatenate([x, y]).mean()),
        "combined_sd": float(np.concatenate([x, y]).std(ddof=1)),
        "quadratic_weighted_kappa": quadratic_kappa(x, y),
        "exact_agreement_pct": float((x == y).mean() * 100),
        "mean_abs_difference": float(np.abs(x - y).mean()),
    }
with open(HERE / "agreement_results.json", "w") as f:
    json.dump(res, f, indent=2)
with open(HERE / "human_audit_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["criterion", "rater1_mean", "rater1_sd", "rater2_mean", "rater2_sd", "combined_mean", "combined_sd",
                "quadratic_weighted_kappa", "exact_agreement_pct", "mean_abs_difference"])
    for c, v in res["criteria"].items():
        w.writerow([c] + [v[k] for k in ("rater1_mean", "rater1_sd", "rater2_mean", "rater2_sd", "combined_mean", "combined_sd",
                                         "quadratic_weighted_kappa", "exact_agreement_pct", "mean_abs_difference")])
print(json.dumps(res, indent=2))

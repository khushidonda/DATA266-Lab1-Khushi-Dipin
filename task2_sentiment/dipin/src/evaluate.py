"""Task 2 (Dipin): full evaluation suite.

Metric definitions follow the team protocol (same as Khushi's evaluate_task2.py): accuracy,
P/R/F1 (macro/micro/weighted), confusion matrix, ROC-AUC, PR-AUC, MCC, Brier, ECE,
95% bootstrap CIs, exact-binomial McNemar, per-slice macro-F1 / error rate.
Slices are defined on the raw review text so they are independent of preprocessing.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


NEGATION_RE = re.compile(r"\b(not|no|never|nothing|none|nobody|nor|neither|without|cannot)\b|n't\b", re.I)
CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet|except|despite)\b", re.I)


def macro_f1(y, p):
    return precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]


def expected_calibration_error(y, prob, pred, n_bins=10):
    conf = np.where(pred == 1, prob, 1 - prob)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(y, pred, fn, seed, n_boot=1000):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(y[idx], pred[idx]))
    return {"point": float(fn(y, pred)), "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5))}


def compute_metrics(y, prob, cfg):
    ecfg = cfg["evaluation"]
    pred = (prob >= ecfg["decision_threshold"]).astype(int)
    out = {"accuracy": float(accuracy_score(y, pred))}
    for avg in ["macro", "micro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        out.update({f"precision_{avg}": float(p), f"recall_{avg}": float(r), f"f1_{avg}": float(f)})
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out.update({"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                "roc_auc": float(roc_auc_score(y, prob)), "pr_auc": float(average_precision_score(y, prob)),
                "mcc": float(matthews_corrcoef(y, pred)), "brier_score": float(brier_score_loss(y, prob)),
                "expected_calibration_error": expected_calibration_error(y, prob, pred, ecfg["ece_bins"])})
    for key, fn in [("accuracy", accuracy_score), ("macro_f1", macro_f1), ("mcc", matthews_corrcoef)]:
        ci = bootstrap_ci(y, pred, fn, cfg["seed"], ecfg["bootstrap_samples"])
        out[f"{key}_ci95_lower"], out[f"{key}_ci95_upper"] = ci["ci_lower"], ci["ci_upper"]
    return out, pred


def mcnemar(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    n10, n01 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = 1.0 if n10 + n01 == 0 else float(stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return {"a_right_b_wrong": n10, "a_wrong_b_right": n01, "p_value": p}


def build_slices(raw_texts, unks, lengths, truncated):
    wc = np.array([len(t.split()) for t in raw_texts])
    t1, t2 = np.percentile(wc, [33.33, 66.67])
    unk_rate = unks / np.maximum(1, lengths)
    return {
        "all": np.ones(len(wc), bool),
        f"raw_length_short(<= {int(t1)} words)": wc <= t1,
        f"raw_length_medium({int(t1)}-{int(t2)})": (wc > t1) & (wc <= t2),
        f"raw_length_long(> {int(t2)} words)": wc > t2,
        "contains_negation": np.array([bool(NEGATION_RE.search(t)) for t in raw_texts]),
        "contains_contrast_word": np.array([bool(CONTRAST_RE.search(t)) for t in raw_texts]),
        "truncated": truncated.astype(bool),
        "high_unk_rate(>10%)": unk_rate > 0.10,
    }


def slice_metrics(y, pred, slices):
    return {name: {"n": int(m.sum()),
                   "macro_f1": float(macro_f1(y[m], pred[m])) if m.any() else float("nan"),
                   "error_rate": float((pred[m] != y[m]).mean()) if m.any() else float("nan")}
            for name, m in slices.items()}


def error_candidates(y, prob, pred, raw_texts, slices, k=5):
    """Selects the 20 errors for manual review: 5 per category."""
    err = pred != y
    idx = np.arange(len(y))
    fp = idx[err & (pred == 1)]
    fn = idx[err & (pred == 0)]
    picks = {
        "confident_false_positive": fp[np.argsort(-prob[fp])][:k],
        "confident_false_negative": fn[np.argsort(prob[fn])][:k],
        "near_threshold_error": idx[err][np.argsort(np.abs(prob[err] - 0.5))][:k],
    }
    used = set(np.concatenate(list(picks.values())).tolist())
    rates = {n: (pred[m] != y[m]).mean() for n, m in slices.items() if n != "all" and m.sum() >= 100}
    worst = max(rates, key=rates.get)
    cand = [i for i in idx[err & slices[worst]][np.argsort(-np.abs(prob[err & slices[worst]] - 0.5))] if i not in used]
    picks[f"slice_failure[{worst}]"] = np.array(cand[:k])
    rows = []
    for cat, ids in picks.items():
        for i in ids:
            rows.append({"category": cat, "test_index": int(i), "true_label": int(y[i]), "predicted_label": int(pred[i]),
                         "prob_positive": round(float(prob[i]), 4), "text": raw_texts[i].replace("\\n", " ")})
    return rows, worst


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["neg", "pos"])
    ax.set_yticks([0, 1], ["neg", "pos"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(results, y, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, prob in results.items():
        fpr, tpr, _ = roc_curve(y, prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, prob):.4f})")
        prec, rec, _ = precision_recall_curve(y, prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y, prob):.4f})")
        bins = np.linspace(0, 1, 16)
        which = np.clip(np.digitize(prob, bins) - 1, 0, 14)
        xs = [prob[which == b].mean() for b in range(15) if (which == b).any()]
        ys = [y[which == b].mean() for b in range(15) if (which == b).any()]
        axes[2].plot(xs, ys, marker="o", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    for ax, t, xl, yl in zip(axes, ["ROC", "Precision-Recall", "Reliability (calibration)"],
                             ["FPR", "Recall", "Mean predicted P(pos)"], ["TPR", "Precision", "Observed P(pos)"]):
        ax.set_title(t)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_history(summaries, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, s in summaries.items():
        ep = [h["epoch"] for h in s["history"]]
        axes[0].plot(ep, [h["train_loss"] for h in s["history"]], "-o", label=f"{name} train")
        axes[0].plot(ep, [h["val_loss"] for h in s["history"]], "--o", label=f"{name} val")
        axes[1].plot(ep, [h["val_macro_f1"] for h in s["history"]], "-o", label=name)
    axes[0].set_title("BCE loss")
    axes[1].set_title("Validation macro-F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

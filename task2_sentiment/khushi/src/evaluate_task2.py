"""Task 2 (Step 6): evaluation script for Khushi's three classifiers.

Loads a trained checkpoint, runs inference on a chosen data split, and
computes the full required metrics suite: accuracy; precision/recall/F1
(macro/micro/weighted); confusion matrix; ROC-AUC; PR-AUC; MCC; Brier
score; ECE; 95% bootstrap CIs for accuracy/macro-F1/MCC; paired McNemar
tests between two models' predictions; macro-F1 and error rate per data
slice; and reports parameter count / training time / examples-per-sec /
peak memory from the checkpoint. Exports predictions, true labels, and
positive-class probabilities to CSV for later manual error review.

This script does NOT train anything. It only evaluates an existing
checkpoint -- see check_evaluate_task2.py for a lightweight validation
run using the (deliberately undertrained) smoke-test checkpoints.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from models import build_model
from train_task2 import DATA_DIR, build_dataloader, load_checkpoint, load_config

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def get_predictions(model, input_ids, labels, device, batch_size=256):
    loader = build_dataloader(input_ids, labels, batch_size, shuffle=False)
    model.eval()
    all_preds, all_probs_pos, all_labels = [], [], []
    with torch.no_grad():
        for batch_ids, batch_labels in loader:
            batch_ids = batch_ids.to(device)
            logits = model(batch_ids)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_probs_pos.append(probs[:, 1].cpu().numpy())
            all_labels.append(batch_labels.numpy())
    return (np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_probs_pos))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def expected_calibration_error(y_true, y_prob_pos, y_pred, n_bins=10):
    confidences = np.where(y_pred == 1, y_prob_pos, 1 - y_prob_pos)
    accuracies = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def bootstrap_ci(y_true, y_pred, metric_fn, seed, n_bootstrap=1000, confidence=0.95):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            values.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:
            continue  # e.g. resample landed on a single class; skip that draw
    alpha = (1 - confidence) / 2 * 100
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "ci_lower": float(np.percentile(values, alpha)) if values else float("nan"),
        "ci_upper": float(np.percentile(values, 100 - alpha)) if values else float("nan"),
        "n_bootstrap_used": len(values),
    }


def macro_f1(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]


def compute_metrics(y_true, y_pred, y_prob_pos, seed):
    accuracy = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    n_classes_present = len(set(y_true.tolist()))
    if n_classes_present > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob_pos))
        pr_auc = float(average_precision_score(y_true, y_prob_pos))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    mcc = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob_pos))
    ece = expected_calibration_error(y_true, y_prob_pos, y_pred)

    bootstrap = {
        "accuracy": bootstrap_ci(y_true, y_pred, accuracy_score, seed=seed),
        "macro_f1": bootstrap_ci(y_true, y_pred, macro_f1, seed=seed),
        "mcc": bootstrap_ci(y_true, y_pred, matthews_corrcoef, seed=seed),
    }

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f1_macro),
        "precision_micro": float(p_micro), "recall_micro": float(r_micro), "f1_micro": float(f1_micro),
        "precision_weighted": float(p_weighted), "recall_weighted": float(r_weighted), "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "brier_score": brier,
        "ece": ece,
        "bootstrap_95ci": bootstrap,
    }


def mcnemar_test(y_true, preds_a, preds_b):
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    n10 = int(np.sum(correct_a & ~correct_b))  # a right, b wrong
    n01 = int(np.sum(~correct_a & correct_b))  # a wrong, b right
    n_discordant = n10 + n01
    if n_discordant == 0:
        return {"n10": n10, "n01": n01, "n_discordant": 0, "p_value": 1.0, "method": "exact_binomial"}
    result = stats.binomtest(min(n10, n01), n_discordant, p=0.5)
    return {"n10": n10, "n01": n01, "n_discordant": n_discordant,
            "p_value": float(result.pvalue), "method": "exact_binomial"}


# --------------------------------------------------------------------------
# Per-slice metrics -- computed directly from input_ids, no extra metadata
# needed. Starting slice set; can be extended later.
# --------------------------------------------------------------------------

def build_default_slices(input_ids, pad_id):
    non_pad_counts = (input_ids != pad_id).sum(axis=1)
    max_length = input_ids.shape[1]

    slices = {"all": np.ones(len(input_ids), dtype=bool)}
    slices["empty_after_cleaning"] = non_pad_counts == 0
    slices["truncated_at_max_length"] = non_pad_counts == max_length

    non_empty = non_pad_counts > 0
    if non_empty.sum() > 0:
        lengths_non_empty = non_pad_counts[non_empty]
        t1, t2 = np.percentile(lengths_non_empty, [33.33, 66.67])
        slices["short_length"] = non_empty & (non_pad_counts <= t1)
        slices["medium_length"] = non_empty & (non_pad_counts > t1) & (non_pad_counts <= t2)
        slices["long_length"] = non_empty & (non_pad_counts > t2)

    return slices


def per_slice_metrics(y_true, y_pred, slices):
    results = {}
    for name, mask in slices.items():
        n = int(mask.sum())
        if n == 0:
            results[name] = {"n": 0, "macro_f1": float("nan"), "error_rate": float("nan")}
            continue
        results[name] = {
            "n": n,
            "macro_f1": float(macro_f1(y_true[mask], y_pred[mask])),
            "error_rate": float((y_pred[mask] != y_true[mask]).mean()),
        }
    return results


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export_predictions(model_name, split_name, y_true, y_pred, y_prob_pos, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_{split_name}_predictions.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "predicted_label", "prob_positive"])
        for i, (t, p, prob) in enumerate(zip(y_true, y_pred, y_prob_pos)):
            writer.writerow([i, int(t), int(p), float(prob)])
    return path


# --------------------------------------------------------------------------
# Main evaluation entry point
# --------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint_path, model_name, config, split_name, device, batch_size=256,
                         max_examples=None, subset_seed=42):
    """max_examples takes a random (seeded) subset of that many rows --
    used by check_evaluate_task2.py to keep the lightweight validation
    run cheap. Random, not a head-slice: the saved split arrays are
    ordered by class block (all label-0 rows first, then all label-1),
    so a head-slice would silently be single-class. Leave max_examples
    None for a real evaluation over the full split."""
    model = build_model(model_name, config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)

    input_ids = np.load(DATA_DIR / f"{split_name}_input_ids.npy")
    labels = np.load(DATA_DIR / f"{split_name}_labels.npy")
    if max_examples is not None:
        rng = np.random.default_rng(subset_seed)
        idx = rng.choice(len(labels), size=max_examples, replace=False)
        input_ids = input_ids[idx]
        labels = labels[idx]

    y_true, y_pred, y_prob_pos = get_predictions(model, input_ids, labels, device, batch_size)

    seed = config.get("seed", 42)
    metrics = compute_metrics(y_true, y_pred, y_prob_pos, seed=seed)

    pad_id = config["models"][model_name]["pad_id"]
    slices = build_default_slices(input_ids, pad_id)
    metrics["per_slice"] = per_slice_metrics(y_true, y_pred, slices)

    metrics["parameter_count"] = checkpoint.get("num_parameters")
    metrics["training_seconds"] = checkpoint.get("training_seconds")
    metrics["examples_per_sec"] = checkpoint.get("examples_per_sec")
    metrics["peak_memory_bytes"] = checkpoint.get("peak_memory_bytes")

    export_path = export_predictions(model_name, split_name, y_true, y_pred, y_prob_pos)
    metrics["predictions_export_path"] = str(export_path)

    return metrics, y_true, y_pred, y_prob_pos


if __name__ == "__main__":
    print("evaluate_task2.py defines evaluation functions but does not run "
          "anything on import. Call evaluate_checkpoint(...) once a real "
          "trained checkpoint exists. See check_evaluate_task2.py for a "
          "lightweight validation run using smoke-test checkpoints.")

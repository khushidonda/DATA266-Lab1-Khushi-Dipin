"""Task 2 (Step 6): lightweight validation of evaluate_task2.py.

Uses ONLY the deliberately-undertrained smoke-test checkpoints (1 tiny
epoch each, from smoke_test_task2.py) on the same tiny 100-example
validation subset. This exercises every metric/export code path end to
end and confirms it runs without error and produces sane-shaped output --
it does NOT produce meaningful final metrics, since the models are
untrained. Real metrics come only after the actual training run.
"""

import math

import numpy as np
import torch

from evaluate_task2 import compute_metrics, evaluate_checkpoint, mcnemar_test
from train_task2 import CHECKPOINT_DIR, load_config

TINY_VAL_SIZE = 100
MODEL_NAMES = ["baseline", "experimental_1", "experimental_2"]


def sanity_check_metrics(model_name, metrics):
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert -1.0 <= metrics["mcc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert math.isnan(metrics["roc_auc"]) or 0.0 <= metrics["roc_auc"] <= 1.0
    assert math.isnan(metrics["pr_auc"]) or 0.0 <= metrics["pr_auc"] <= 1.0
    for name in ["accuracy", "macro_f1", "mcc"]:
        ci = metrics["bootstrap_95ci"][name]
        assert ci["n_bootstrap_used"] > 0, f"Bootstrap produced zero usable resamples for {name}."
    assert metrics["parameter_count"] is not None and metrics["parameter_count"] > 0
    assert metrics["training_seconds"] is not None and metrics["training_seconds"] >= 0
    assert metrics["examples_per_sec"] is not None and metrics["examples_per_sec"] > 0
    for slice_name, slice_metrics in metrics["per_slice"].items():
        if slice_metrics["n"] > 0:
            assert 0.0 <= slice_metrics["error_rate"] <= 1.0
    print(f"  [PASS] {model_name}: all metric values are finite and within valid ranges.")


def main():
    print("SMOKE TEST ONLY -- these are NOT final metrics. Models are undertrained "
          "(1 tiny epoch on 200 train examples). This validates that the evaluation "
          "code runs correctly end to end.\n")

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for model_name in MODEL_NAMES:
        print(f"=== {model_name} ===")
        checkpoint_path = CHECKPOINT_DIR / "smoke_test" / f"{model_name}_epoch_0.pt"
        metrics, y_true, y_pred, y_prob_pos = evaluate_checkpoint(
            checkpoint_path, model_name, config, split_name="validation",
            device=device, max_examples=TINY_VAL_SIZE,
        )
        sanity_check_metrics(model_name, metrics)
        print(f"  [PASS] accuracy={metrics['accuracy']:.4f}, f1_macro={metrics['f1_macro']:.4f}, "
              f"mcc={metrics['mcc']:.4f}, roc_auc={metrics['roc_auc']:.4f}, "
              f"brier={metrics['brier_score']:.4f}, ece={metrics['ece']:.4f}")
        print(f"  [PASS] bootstrap 95% CI (accuracy): "
              f"[{metrics['bootstrap_95ci']['accuracy']['ci_lower']:.4f}, "
              f"{metrics['bootstrap_95ci']['accuracy']['ci_upper']:.4f}] "
              f"({metrics['bootstrap_95ci']['accuracy']['n_bootstrap_used']} usable resamples)")
        print(f"  [PASS] per-slice metrics computed for: {list(metrics['per_slice'].keys())}")
        print(f"  [PASS] predictions exported to {metrics['predictions_export_path']}")
        results[model_name] = (y_true, y_pred)

    print("\n=== McNemar tests (paired, on the same tiny validation subset) ===")
    y_true_baseline, preds_baseline = results["baseline"]
    for other in ["experimental_1", "experimental_2"]:
        y_true_other, preds_other = results[other]
        assert np.array_equal(y_true_baseline, y_true_other), (
            "y_true mismatch between models -- McNemar test requires identical example sets."
        )
        result = mcnemar_test(y_true_baseline, preds_baseline, preds_other)
        assert 0.0 <= result["p_value"] <= 1.0
        print(f"  [PASS] baseline vs {other}: n10={result['n10']}, n01={result['n01']}, "
              f"p_value={result['p_value']:.4f} ({result['method']})")

    print("\nAll Step 6 evaluation-code validation checks passed (smoke-test predictions only). "
          "No final metrics computed -- models remain untrained.")


if __name__ == "__main__":
    main()

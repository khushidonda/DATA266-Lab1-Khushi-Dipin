"""Task 2 (Step 5/6): local smoke test for Khushi's three classifiers.

This is NOT full training. For each model (baseline, TextCNN,
bidirectional GRU), it runs one tiny epoch via the real run_training()
driver on a small subset of the real preprocessed data -- which also
exercises and validates the raw JSONL logger -- then verifies checkpoint
save/reload produces identical parameters.
"""

import json

import numpy as np
import torch

from models import build_model
from train_task2 import (
    CHECKPOINT_DIR,
    DATA_DIR,
    RAW_LOG_DIR,
    build_optimizer,
    load_checkpoint,
    load_config,
    run_training,
)

TINY_TRAIN_SIZE = 200
TINY_VAL_SIZE = 100


def load_tiny_arrays(split_name, n, seed):
    """A random (seeded) subset, not the first n rows -- the saved split
    arrays are ordered by class block (all label-0 rows first, then all
    label-1), so a head-slice would silently be single-class."""
    input_ids = np.load(DATA_DIR / f"{split_name}_input_ids.npy")
    labels = np.load(DATA_DIR / f"{split_name}_labels.npy")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), size=n, replace=False)
    return input_ids[idx], labels[idx]


def check_raw_log(log_path, expected_fields):
    with open(log_path) as f:
        lines = f.readlines()
    assert len(lines) == 1, f"Expected 1 log record for a 1-epoch smoke run, got {len(lines)}."
    record = json.loads(lines[0])
    missing = [field for field in expected_fields if field not in record]
    assert not missing, f"Raw log record is missing fields: {missing}"
    return record


def run_smoke_test_for_model(model_name, config, device):
    print(f"\n=== {model_name} ===")

    seed = config.get("seed", 42)
    model = build_model(model_name, config).to(device)
    train_arrays = load_tiny_arrays("train", TINY_TRAIN_SIZE, seed=seed)
    val_arrays = load_tiny_arrays("validation", TINY_VAL_SIZE, seed=seed + 1)

    smoke_checkpoint_dir = CHECKPOINT_DIR / "smoke_test"
    history, log_path = run_training(
        model, model_name, config, device,
        max_epochs=1,
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        checkpoint_dir=smoke_checkpoint_dir,
        run_type="smoke_test",
    )
    print(f"  [PASS] tiny 1-epoch run via run_training(): "
          f"train_loss={history[0]['train_loss']:.4f}, val_loss={history[0]['validation_loss']:.4f}, "
          f"val_acc={history[0]['validation_accuracy']:.4f} (all finite).")

    expected_fields = [
        "run_type", "model_name", "config", "epoch", "train_loss", "validation_loss",
        "validation_accuracy", "learning_rate", "training_time_seconds",
        "examples_per_sec", "peak_memory_bytes", "checkpoint_path",
    ]
    record = check_raw_log(log_path, expected_fields)
    assert record["run_type"] == "smoke_test", f"Expected run_type='smoke_test', got {record['run_type']!r}"
    print(f"  [PASS] raw JSONL log written to {log_path}, contains all required fields "
          f"(run_type={record['run_type']!r}): {list(record.keys())}")

    checkpoint_path = smoke_checkpoint_dir / f"{model_name}_epoch_0.pt"
    reloaded_model = build_model(model_name, config).to(device)
    reloaded_optimizer = build_optimizer(reloaded_model, config["models"][model_name])
    checkpoint = load_checkpoint(checkpoint_path, reloaded_model, reloaded_optimizer)

    params_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model.state_dict().values(), reloaded_model.state_dict().values())
    )
    assert params_match, f"Reloaded checkpoint parameters do not match for {model_name}."
    print(f"  [PASS] checkpoint reloaded and verified identical "
          f"(epoch={checkpoint['epoch']}, num_parameters={checkpoint['num_parameters']:,}).")

    return checkpoint_path


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Smoke test running on device: {device}")
    print(f"[INFO] Raw logs will be written under: {RAW_LOG_DIR}")

    checkpoint_paths = {}
    for model_name in ["baseline", "experimental_1", "experimental_2"]:
        checkpoint_paths[model_name] = run_smoke_test_for_model(model_name, config, device)

    print("\nAll Step 5/6 smoke-test checks passed for all three models "
          "(training loop, raw logging, checkpointing). This was NOT full training.")
    return checkpoint_paths


if __name__ == "__main__":
    main()

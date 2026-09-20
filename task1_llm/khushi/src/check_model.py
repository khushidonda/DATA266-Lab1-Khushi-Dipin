"""Task 1 (Step 5): lightweight correctness checks for Khushi's from-scratch
GPT implementation in model.py. These are sanity checks only, NOT training."""

from pathlib import Path

import numpy as np
import torch

from model import build_model_from_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data_processed"


def check_initializes():
    model, cfg = build_model_from_config()
    print("[PASS] Model initializes.")
    return model, cfg


def check_shapes(model, cfg):
    batch_size = 4
    T = cfg["sequence_length"]
    idx = torch.randint(0, cfg["vocab_size"], (batch_size, T))
    logits, loss = model(idx, targets=idx)
    expected_logits_shape = (batch_size, T, cfg["vocab_size"])
    assert logits.shape == expected_logits_shape, (
        f"Expected logits shape {expected_logits_shape}, got {logits.shape}"
    )
    print(f"[PASS] Input shape {tuple(idx.shape)} -> logits shape {tuple(logits.shape)} "
          f"(expected {expected_logits_shape}).")


def check_causal_mask(model, cfg):
    model.eval()
    T = cfg["sequence_length"]
    idx = torch.randint(0, cfg["vocab_size"], (1, T))

    # Change only the very last token. Because of causal masking, no earlier
    # position can attend to it, so every logit before the last position
    # must be completely unaffected.
    idx_modified = idx.clone()
    idx_modified[0, -1] = (idx_modified[0, -1] + 1) % cfg["vocab_size"]

    with torch.no_grad():
        logits_a, _ = model(idx)
        logits_b, _ = model(idx_modified)

    earlier_positions_match = torch.allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-6)
    last_position_differs = not torch.allclose(logits_a[:, -1, :], logits_b[:, -1, :], atol=1e-6)

    assert earlier_positions_match, (
        "Causal mask violated: changing the last token changed earlier-position logits."
    )
    assert last_position_differs, (
        "Sanity check inconclusive: changing the last token had no effect on its own logits."
    )

    print("[PASS] Causal mask verified: changing the last token leaves all earlier-position "
          "logits unchanged, but changes the last position's own logits.")
    model.train()


def check_parameter_count(model):
    n_params = model.num_parameters()
    print(f"[PASS] Parameter count reported: {n_params:,} parameters.")
    return n_params


def check_forward_pass_finite(model, cfg):
    train_inputs_path = DATA_DIR / "train_inputs.npy"
    train_targets_path = DATA_DIR / "train_targets.npy"

    if train_inputs_path.exists() and train_targets_path.exists():
        train_inputs = np.load(train_inputs_path)
        train_targets = np.load(train_targets_path)
        batch_inputs = torch.from_numpy(train_inputs[:8]).long()
        batch_targets = torch.from_numpy(train_targets[:8]).long()
        source = "real Step 3 training sequences"
    else:
        batch_inputs = torch.randint(0, cfg["vocab_size"], (8, cfg["sequence_length"]))
        batch_targets = torch.randint(0, cfg["vocab_size"], (8, cfg["sequence_length"]))
        source = "random fallback data (Step 3 .npy files not found)"

    logits, loss = model(batch_inputs, targets=batch_targets)

    assert torch.isfinite(logits).all(), "Forward pass produced non-finite logits."
    assert torch.isfinite(loss).all(), "Forward pass produced non-finite loss."

    print(f"[PASS] Forward pass on {source}: logits finite={torch.isfinite(logits).all().item()}, "
          f"loss={loss.item():.4f} (finite={torch.isfinite(loss).item()}).")


def main():
    model, cfg = check_initializes()
    check_shapes(model, cfg)
    check_causal_mask(model, cfg)
    check_parameter_count(model)
    check_forward_pass_finite(model, cfg)
    print("\nAll Step 5 lightweight correctness checks passed.")


if __name__ == "__main__":
    main()

"""Task 1 (Step 7): local smoke test for Khushi's training pipeline.

This is NOT the full 10-epoch training run. It exercises the pipeline
end-to-end on a tiny subset of the real preprocessed data: forward pass,
backward pass, finite loss, validation pass, gradient-norm tracking, NaN
detection, checkpoint save/reload, and text generation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import build_model_from_config
from train import (
    CHECKPOINT_DIR,
    DATA_DIR,
    build_optimizer,
    build_scheduler,
    check_finite,
    compute_grad_norm,
    evaluate,
    generate,
    load_checkpoint,
    load_full_config,
    save_checkpoint,
    train_one_epoch,
    RawLogger,
)

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "tokenizer.json"
TINY_TRAIN_SIZE = 200
TINY_VAL_SIZE = 100
SMOKE_TEST_BATCH_SIZE = 8
SMOKE_TEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "smoke_test.pt"


def build_tiny_dataloaders(batch_size):
    train_inputs = np.load(DATA_DIR / "train_inputs.npy")[:TINY_TRAIN_SIZE]
    train_targets = np.load(DATA_DIR / "train_targets.npy")[:TINY_TRAIN_SIZE]
    val_inputs = np.load(DATA_DIR / "validation_inputs.npy")[:TINY_VAL_SIZE]
    val_targets = np.load(DATA_DIR / "validation_targets.npy")[:TINY_VAL_SIZE]

    train_dataset = TensorDataset(torch.from_numpy(train_inputs).long(), torch.from_numpy(train_targets).long())
    val_dataset = TensorDataset(torch.from_numpy(val_inputs).long(), torch.from_numpy(val_targets).long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def decode(indices, vocab):
    return "".join(vocab[i] for i in indices)


def main():
    full_cfg = load_full_config()
    with open(TOKENIZER_PATH) as f:
        vocab = json.load(f)["vocab"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Smoke test running on device: {device}")

    model, _ = build_model_from_config()
    model.to(device)

    train_loader, val_loader = build_tiny_dataloaders(SMOKE_TEST_BATCH_SIZE)
    print(f"[INFO] Tiny dataset subset: {len(train_loader.dataset)} train sequences, "
          f"{len(val_loader.dataset)} validation sequences "
          f"({len(train_loader)} train batches, {len(val_loader)} val batches).")

    optimizer = build_optimizer(model, full_cfg["training"])
    total_steps = len(train_loader)  # a single tiny "epoch" for this smoke test only
    scheduler = build_scheduler(optimizer, full_cfg["training"], total_steps)
    logger = RawLogger()
    print(f"[INFO] Raw log file: {logger.path}")

    # Forward pass + backward pass + finite loss + gradient norm, over the tiny train loader.
    train_loss, global_step = train_one_epoch(
        model, train_loader, optimizer, scheduler, logger, device, epoch=0, global_step=0,
    )
    print(f"[PASS] Forward + backward pass over {global_step} tiny-subset steps: "
          f"mean train loss={train_loss:.4f} (finite).")

    # Validation pass.
    val_loss = evaluate(model, val_loader, device)
    print(f"[PASS] Validation pass: mean val loss={val_loss:.4f} (finite).")

    # Gradient norm is available.
    grad_norm = compute_grad_norm(model)
    print(f"[PASS] Gradient norm available: {grad_norm:.4f}")

    # NaN/Inf detector actually raises on non-finite values (isolated check).
    try:
        check_finite(float("nan"), 1.0, step=-1)
        raise AssertionError("NaN detector failed to raise on a non-finite loss.")
    except RuntimeError:
        print("[PASS] NaN/Inf detector correctly raises on non-finite loss.")

    # Checkpoint save + reload round-trip.
    save_checkpoint(model, optimizer, scheduler, epoch=0, global_step=global_step,
                     full_cfg=full_cfg, output_path=SMOKE_TEST_CHECKPOINT_PATH)
    print(f"[PASS] Checkpoint saved to {SMOKE_TEST_CHECKPOINT_PATH}")

    reloaded_model, _ = build_model_from_config()
    reloaded_optimizer = build_optimizer(reloaded_model, full_cfg["training"])
    reloaded_scheduler = build_scheduler(reloaded_optimizer, full_cfg["training"], total_steps)
    checkpoint = load_checkpoint(SMOKE_TEST_CHECKPOINT_PATH, reloaded_model, reloaded_optimizer, reloaded_scheduler)

    params_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model.state_dict().values(), reloaded_model.state_dict().values())
    )
    assert params_match, "Reloaded checkpoint parameters do not match the saved model."
    print(f"[PASS] Checkpoint reloaded and verified identical "
          f"(epoch={checkpoint['epoch']}, global_step={checkpoint['global_step']}).")

    # Text generation executes.
    generation_cfg = full_cfg["generation"]
    sequence_length = full_cfg["data"]["sequence_length"]
    prompt_indices = train_loader.dataset[0][0][:10].tolist()  # first 10 chars of a real training input
    generated = generate(
        model, prompt_indices,
        max_new_tokens=generation_cfg["max_new_tokens"],
        temperature=generation_cfg["temperature"],
        sequence_length=sequence_length,
        device=device,
    )
    expected_len = len(prompt_indices) + generation_cfg["max_new_tokens"]
    assert len(generated) == expected_len, f"Expected generated length {expected_len}, got {len(generated)}"
    print(f"[PASS] Text generation executes: produced {len(generated)} tokens "
          f"(prompt {len(prompt_indices)} + {generation_cfg['max_new_tokens']} generated).")
    print(f"        prompt : {decode(prompt_indices, vocab)!r}")
    print(f"        output : {decode(generated, vocab)!r}")
    print("        (model is untrained, so generated text is expected to be gibberish)")

    print("\nAll Step 7 local smoke-test checks passed. This was NOT the full 10-epoch training run.")


if __name__ == "__main__":
    main()

"""Task 2 (Step 5): lightweight correctness checks for Khushi's three
from-scratch classifiers. These are sanity checks only, NOT training."""

import torch
import torch.nn as nn

from models import build_model
from train_task2 import load_config

MODEL_NAMES = ["baseline", "experimental_1", "experimental_2"]
SEQUENCE_LENGTH = 256


def check_initializes(model_name, config):
    model = build_model(model_name, config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [PASS] {model_name} initializes ({n_params:,} parameters).")
    return model


def make_batch(vocab_size, pad_id, batch_size=4, seq_len=SEQUENCE_LENGTH, real_lengths=None):
    """Builds a batch where row i has `real_lengths[i]` real (non-PAD)
    tokens at the start, followed by PAD. real_lengths[i] == 0 gives an
    all-PAD row."""
    if real_lengths is None:
        real_lengths = [50, 5, seq_len, 0]  # normal, short, full, all-PAD
    batch = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    for i, length in enumerate(real_lengths):
        if length > 0:
            batch[i, :length] = torch.randint(2, vocab_size, (length,))  # avoid PAD(0)/UNK(1) ids
    return batch, real_lengths


def check_forward_backward_shape_and_finite(model_name, model, vocab_size, pad_id):
    batch, real_lengths = make_batch(vocab_size, pad_id)
    labels = torch.randint(0, 2, (batch.shape[0],))

    logits = model(batch)
    assert logits.shape == (batch.shape[0], 2), f"Expected shape (4, 2), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Forward pass produced non-finite logits."

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    assert torch.isfinite(loss).item(), "Loss is non-finite."
    loss.backward()

    grad_norm_sq = sum(p.grad.detach().float().pow(2).sum().item()
                        for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(torch.tensor(grad_norm_sq)), "Gradient norm is non-finite."

    print(f"  [PASS] {model_name}: forward+backward OK, output shape={tuple(logits.shape)}, "
          f"loss={loss.item():.4f} (finite), grad_norm={grad_norm_sq ** 0.5:.4f} (finite).")
    model.zero_grad()


def check_all_pad_finite(model_name, model, vocab_size, pad_id):
    all_pad_batch, _ = make_batch(vocab_size, pad_id, batch_size=2, real_lengths=[0, 0])
    with torch.no_grad():
        logits = model(all_pad_batch)
    assert torch.isfinite(logits).all(), "All-PAD input produced non-finite logits."
    print(f"  [PASS] {model_name}: all-PAD batch -> finite logits {logits.tolist()}.")


def check_pad_masking_invariance(model_name, model, vocab_size, pad_id):
    """A real prefix fed two ways -- (a) padded out to 256 with PAD, and
    (b) as its own unpadded length-20 tensor -- must produce identical
    output, proving PAD positions don't influence the result."""
    model.eval()
    real_len = 20
    real_tokens = torch.randint(2, vocab_size, (1, real_len))

    padded = torch.full((1, SEQUENCE_LENGTH), pad_id, dtype=torch.long)
    padded[0, :real_len] = real_tokens[0]

    with torch.no_grad():
        logits_padded = model(padded)
        logits_unpadded = model(real_tokens)

    match = torch.allclose(logits_padded, logits_unpadded, atol=1e-5)
    assert match, (f"PAD masking invariance failed for {model_name}: "
                    f"padded={logits_padded.tolist()} vs unpadded={logits_unpadded.tolist()}")
    print(f"  [PASS] {model_name}: padded-to-256 vs unpadded-length-{real_len} give identical "
          f"logits (max abs diff={ (logits_padded - logits_unpadded).abs().max().item():.2e}).")
    model.train()


def main():
    config = load_config()
    vocab_size = config["_vocab_size"]

    for model_name in MODEL_NAMES:
        print(f"\n=== {model_name} ===")
        pad_id = config["models"][model_name]["pad_id"]
        model = check_initializes(model_name, config)
        check_forward_backward_shape_and_finite(model_name, model, vocab_size, pad_id)
        check_all_pad_finite(model_name, model, vocab_size, pad_id)
        check_pad_masking_invariance(model_name, model, vocab_size, pad_id)

    print("\nAll Step 5 lightweight correctness checks passed for all three models.")


if __name__ == "__main__":
    main()

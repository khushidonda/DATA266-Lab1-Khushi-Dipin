"""Task 1: GPU smoke test of the real-run code path (NOT the 10-epoch run).

Calls run_task1.launch("smoke"): the same seeding, run ID, raw logging,
config batch_size (64) and CUDA device as the full run, capped at one epoch
of 100 steps and 20 validation batches. Then checks CUDA placement, finite
losses, NaN status and the NaN/Inf detector, causal masking on the GPU,
checkpoint save/reload and generation, and reports peak GPU memory.

Smoke checkpoints (checkpoints/smoke/<run_id>/) are disposable and removed
only when every check passes; the raw log in reproducibility/raw_logs/khushi/
is always kept.
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

from model import build_model_from_config
from run_task1 import SMOKE_OVERRIDES, launch
from train import CHECKPOINT_DIR, build_optimizer, build_scheduler, check_finite, generate, load_checkpoint, sha256_file

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "data_processed" / "tokenizer.json"
PROMPT = "Once upon a time"


def causal_mask_check(model, vocab_size, sequence_length, device):
    """Changing the token at position p must leave the logits of every
    earlier position unchanged and change the logits at p. Checked for the
    last position and the middle position."""
    model.eval()
    generator = torch.Generator().manual_seed(0)
    idx = torch.randint(0, vocab_size, (1, sequence_length), generator=generator).to(device)
    results = []
    with torch.no_grad():
        base, _ = model(idx)
        for p in (sequence_length - 1, sequence_length // 2):
            modified = idx.clone()
            modified[0, p] = (modified[0, p] + 1) % vocab_size
            out, _ = model(modified)
            earlier = (out[:, :p] - base[:, :p]).abs().max().item()
            at_p = (out[:, p] - base[:, p]).abs().max().item()
            results.append((p, earlier, at_p))
    model.train()
    return results


def main():
    checks = []

    def record(name, ok, detail):
        checks.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    result = launch("smoke")
    model, device, full_cfg = result["model"], result["device"], result["full_cfg"]
    log_path, checkpoint_dir = Path(result["raw_log"]), Path(result["checkpoint_dir"])
    print("\n=== GPU smoke checks (smoke-run numbers are NOT results) ===")

    with open(log_path) as f:
        records = [json.loads(line) for line in f]
    by_type = {}
    for r in records:
        by_type.setdefault(r["type"], []).append(r)
    run_start, epoch_end, run_end = by_type["run_start"][0], by_type["epoch_end"][0], by_type["run_end"][0]
    steps = by_type.get("train_step", [])
    env = run_start["environment"]
    batch_size = full_cfg["training"]["batch_size"]
    sequence_length = full_cfg["data"]["sequence_length"]
    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]

    # 1. Device.
    param_devices = sorted({p.device.type for p in model.parameters()})
    record("model on CUDA", param_devices == ["cuda"],
           f"parameters on {param_devices}; {env['gpu_name']} (compute {env['gpu_compute_capability']}), "
           f"torch {env['torch']}, CUDA {env['torch_cuda']}")

    # 2. Forward/backward at the configured batch size, finite loss.
    losses = [s["train_loss"] for s in steps]
    grad_norms = [s["grad_norm"] for s in steps]
    finite = all(math.isfinite(v) for v in losses + grad_norms) and math.isfinite(epoch_end["val_loss"])
    record("forward/backward, finite loss", finite and len(steps) == SMOKE_OVERRIDES["max_steps_per_epoch"],
           f"{len(steps)} steps; train loss {losses[0]:.4f} -> {losses[-1]:.4f}; "
           f"val loss {epoch_end['val_loss']:.4f} ({epoch_end['val_batches']} batches); "
           f"grad norm {min(grad_norms):.3f}-{max(grad_norms):.3f}")
    tokens_per_step = epoch_end["train_tokens"] / epoch_end["train_steps"]
    record("batch_size 64 from config", batch_size == 64 and tokens_per_step == batch_size * sequence_length,
           f"batch_size={batch_size}; {tokens_per_step:.0f} tokens/step = {batch_size} x {sequence_length}")

    # 3. NaN status and the NaN/Inf detector itself.
    nan_records = by_type.get("nan_detected", [])
    record("NaN status", not nan_records and epoch_end["nan_detected"] is False and run_end["nan_detected"] is False,
           f"{len(nan_records)} nan_detected records; epoch_end/run_end nan_detected=False")
    raised = 0
    for loss_value, grad_norm in ((float("nan"), 1.0), (1.0, float("inf"))):
        try:
            check_finite(loss_value, grad_norm, step=-1)
        except RuntimeError:
            raised += 1
    record("NaN/Inf detector", raised == 2, "check_finite raises on a NaN loss and on an Inf gradient norm")

    # 4. Causal mask on the GPU (same criterion as check_model.py, atol 1e-6).
    mask = causal_mask_check(model, len(vocab), sequence_length, device)
    record("causal mask (GPU)", all(earlier <= 1e-6 and at_p > 1e-6 for _, earlier, at_p in mask),
           "; ".join(f"changed pos {p}: max |d logit| before it = {earlier:.1e}, at it = {at_p:.3f}"
                     for p, earlier, at_p in mask))

    # 5. Checkpoint save/reload.
    saved = by_type["checkpoint_saved"][0]
    checkpoint_path = Path(saved["path"])
    reloaded, _ = build_model_from_config()
    reloaded.to(device)
    optimizer = build_optimizer(reloaded, full_cfg["training"])
    scheduler = build_scheduler(optimizer, full_cfg["training"], run_start["total_steps"])
    ckpt = load_checkpoint(checkpoint_path, reloaded, optimizer, scheduler, map_location=device)
    params_equal = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), reloaded.state_dict().values()))
    probe = torch.randint(0, len(vocab), (4, sequence_length), generator=torch.Generator().manual_seed(1)).to(device)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        logits_a, logits_b = model(probe)[0], reloaded(probe)[0]
    model.train()
    logits_close = torch.allclose(logits_a, logits_b, rtol=0, atol=1e-6)
    hash_ok = sha256_file(checkpoint_path) == saved["sha256"]
    state_ok = ckpt["global_step"] == epoch_end["global_step"] and scheduler.get_last_lr()[0] == epoch_end["lr"]
    record("checkpoint save/reload", params_equal and logits_close and hash_ok and state_ok,
           f"{checkpoint_path.name} ({saved['size_bytes']:,} bytes, sha256 {saved['sha256'][:12]}...); "
           f"params identical={params_equal}, logits identical={torch.equal(logits_a, logits_b)}, "
           f"hash matches log={hash_ok}, epoch={ckpt['epoch']}, global_step={ckpt['global_step']}, "
           f"optimizer/scheduler state restored={state_ok}")

    # 6. Generation with the configured sampling settings.
    gen_cfg = full_cfg["generation"]
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    prompt_idx = [char_to_idx[ch] for ch in PROMPT]
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = generate(model, prompt_idx, gen_cfg["max_new_tokens"], gen_cfg["temperature"], sequence_length, device)
    torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - start
    record("generation", len(out) == len(prompt_idx) + gen_cfg["max_new_tokens"],
           f"{len(out)} tokens (prompt {len(prompt_idx)} + {gen_cfg['max_new_tokens']} new), "
           f"temperature {gen_cfg['temperature']}, {gen_cfg['max_new_tokens'] / gen_seconds:.0f} new tokens/sec")
    print(f"    sample (100-step smoke model, gibberish expected): {''.join(vocab[i] for i in out)!r}")

    # 7. Peak GPU memory at batch size 64.
    total = env["gpu_total_memory_bytes"]
    allocated = epoch_end["peak_gpu_memory_allocated_bytes"]
    reserved = epoch_end["peak_gpu_memory_reserved_bytes"]
    record("batch 64 fits / peak GPU memory", reserved < 0.5 * total,
           f"peak allocated {allocated / 2**20:.0f} MiB, peak reserved {reserved / 2**20:.0f} MiB "
           f"of {total / 2**30:.2f} GiB ({100 * reserved / total:.1f}%)")

    # Informational: step time, raw-log write cost on this filesystem, rough full-run duration.
    probe_log = checkpoint_dir / "log_write_probe.jsonl"
    line = json.dumps(steps[-1])
    start = time.perf_counter()
    for _ in range(200):
        with open(probe_log, "a") as f:
            f.write(line + "\n")
    write_ms = (time.perf_counter() - start) / 200 * 1000
    step_ms = epoch_end["train_seconds"] / epoch_end["train_steps"] * 1000
    epochs = full_cfg["training"]["epochs"]
    est_seconds = (run_start["train_sequences"] * sequence_length * epochs / epoch_end["train_tokens_per_sec"]
                   + run_start["val_sequences"] * sequence_length * epochs / epoch_end["val_tokens_per_sec"])
    print(f"[INFO] mean train step {step_ms:.2f} ms, of which ~{write_ms:.2f} ms is the per-step raw-log write")
    print(f"[INFO] smoke throughput: train {epoch_end['train_tokens_per_sec']:,.0f} tokens/s, "
          f"val {epoch_end['val_tokens_per_sec']:,.0f} tokens/s (includes CUDA warm-up)")
    print(f"[INFO] rough full-run estimate: ~{est_seconds / 60:.0f} min for {epochs} epochs")

    all_ok = all(ok for _, ok in checks)
    if all_ok:
        smoke_root = CHECKPOINT_DIR / "smoke"
        assert checkpoint_dir.parent == smoke_root  # only ever delete smoke checkpoints
        shutil.rmtree(checkpoint_dir)
        if not any(smoke_root.iterdir()):
            smoke_root.rmdir()
        print(f"\nRemoved disposable smoke checkpoints: {checkpoint_dir}")
    else:
        print(f"\nKept smoke checkpoints for debugging: {checkpoint_dir}")
    print(f"Raw log kept: {log_path}")
    print("ALL GPU SMOKE CHECKS PASSED" if all_ok else "SOME GPU SMOKE CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

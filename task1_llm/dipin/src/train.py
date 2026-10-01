"""Task 1 (Dipin): GPT training with cross-entropy loss, LR warm-up + cosine schedule.

    python task1_llm/dipin/src/train.py            # full run from configs/task1_config.json
    python task1_llm/dipin/src/train.py --fresh    # ignore any resume state

Writes an append-only JSONL raw log, saves resume state after every epoch
(checkpoints/latest.pt), the best-validation weights (best.pt) and the final weights (model.pt).
"""

import argparse
import math
import os
import time
from datetime import datetime

import torch

from data import DATA_DIR, REPO_DIR, TASK_DIR, load_config, load_prepared, prepare, set_seed, window_starts, get_batch
from model import build_model, count_parameters
from utils import RawLogger, autocast, get_device, hardware_info, peak_memory, sync

CKPT_DIR = TASK_DIR / "checkpoints"
LOG_DIR = REPO_DIR / "reproducibility" / "raw_logs" / "dipin"
SMOKE_DIR = TASK_DIR / "outputs" / "smoke"


def lr_at(step, tcfg, total_steps):
    """Linear warm-up to learning_rate, then cosine decay to min_learning_rate."""
    warm = tcfg["warmup_steps"]
    if step < warm:
        return tcfg["learning_rate"] * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total_steps - warm))
    return tcfg["min_learning_rate"] + 0.5 * (tcfg["learning_rate"] - tcfg["min_learning_rate"]) * (
        1 + math.cos(math.pi * progress))


def make_optimizer(model, tcfg):
    # Weight decay on weight matrices only; LayerNorm gains/biases and embeddings' 1-D params are exempt.
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": tcfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=tcfg["learning_rate"], betas=(tcfg["beta1"], tcfg["beta2"]))


@torch.no_grad()
def score_stream(model, stream, seq_len, batch_size, device, amp):
    """Token-weighted cross-entropy (nats/char) and top-1 accuracy over every window, eval mode."""
    model.eval()
    starts = window_starts(len(stream), seq_len)
    nll, correct = 0.0, 0  # accumulated in Python (float64): exact enough and works on every backend
    for i in range(0, len(starts), batch_size):
        x, y = get_batch(stream, starts[i:i + batch_size], seq_len)
        with autocast(device, amp):
            logits, _ = model(x)
        logits = logits.float()
        nll += torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    n = len(starts) * seq_len
    ce = nll / n
    return {"ce_nats": ce, "perplexity": math.exp(ce), "bpc": ce / math.log(2),
            "top1_accuracy": correct / n, "tokens": n}


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def weights_payload(model, cfg, tok, extra):
    return {"state_dict": model.state_dict(), "config": cfg, "char_to_idx": tok.char_to_idx, **extra}


def train(cfg, data, device, epochs=None, smoke=False, resume=True):
    tcfg, T = cfg["training"], cfg["data"]["sequence_length"]
    epochs = epochs or tcfg["epochs"]
    tok = data["tokenizer"]
    ckpt_dir = CKPT_DIR / "smoke" if smoke else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    amp = tcfg["amp_bf16"]

    set_seed(cfg["seed"])
    model = build_model(cfg, tok.vocab_size).to(device)
    opt = make_optimizer(model, tcfg)
    train_stream = torch.from_numpy(data["train"]).to(device)
    val_stream = torch.from_numpy(data["validation"]).to(device)
    steps_per_epoch = len(window_starts(len(train_stream), T, offset=T - 1)) // tcfg["batch_size"]
    total_steps = epochs * steps_per_epoch
    hw = hardware_info(device)
    n_params = count_parameters(model)

    state = {"epoch": 0, "step": 0, "history": [], "train_seconds": 0.0, "tokens_seen": 0, "nan_count": 0,
             "loss_spikes": 0, "grad_norm_sum": 0.0, "grad_norm_max": 0.0, "grad_norm_max_step": 0,
             "best_val": float("inf"), "best_epoch": 0}
    latest = ckpt_dir / "latest.pt"
    if resume and latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        opt.load_state_dict(saved["optimizer"])
        state = saved["state"]
        run_id, log_path = saved["run_id"], REPO_DIR / saved["raw_log"]
        print(f"resuming {run_id} after epoch {state['epoch']}")
    else:
        run_id = f"{cfg['run_id']}{'_smoke' if smoke else ''}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = (SMOKE_DIR / "raw_logs" if smoke else LOG_DIR) / f"{run_id}.jsonl"
    logger = RawLogger(log_path)
    raw_log_rel = log_path.relative_to(REPO_DIR).as_posix()
    logger.log("resume" if state["epoch"] else "start", run_id=run_id, config=cfg, epochs=epochs,
               steps_per_epoch=steps_per_epoch, total_steps=total_steps, parameter_count=n_params, hardware=hw,
               vocab_size=tok.vocab_size, train_chars=len(train_stream), val_chars=len(val_stream))
    print(f"[{run_id}] params={n_params:,} device={hw.get('gpu')} steps/epoch={steps_per_epoch}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ema = None
    for epoch in range(state["epoch"] + 1, epochs + 1):
        model.train()
        # Fresh generator per epoch so a resumed run sees the same batches as an uninterrupted one.
        g = torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch)
        offset = int(torch.randint(0, T, (1,), generator=g)) if cfg["data"]["random_window_offset_per_epoch"] else 0
        starts = window_starts(len(train_stream), T, offset)
        starts = starts[torch.randperm(len(starts), generator=g)]
        loss_sum, n_steps = 0.0, 0
        sync(device)
        t0 = time.perf_counter()
        for b in range(steps_per_epoch):
            x, y = get_batch(train_stream, starts[b * tcfg["batch_size"]:(b + 1) * tcfg["batch_size"]], T)
            lr = lr_at(state["step"], tcfg, total_steps)
            for group in opt.param_groups:
                group["lr"] = lr
            with autocast(device, amp):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["gradient_clip_max_norm"]).item()
            loss_val = loss.item()
            state["step"] += 1
            if not (math.isfinite(loss_val) and math.isfinite(grad_norm)):
                state["nan_count"] += 1
                logger.log("non_finite", epoch=epoch, step=state["step"], loss=loss_val, grad_norm=grad_norm)
                continue  # skip the update; gradients are discarded at the next zero_grad
            opt.step()
            loss_sum += loss_val
            n_steps += 1
            state["grad_norm_sum"] += grad_norm
            if grad_norm > state["grad_norm_max"]:
                state["grad_norm_max"], state["grad_norm_max_step"] = grad_norm, state["step"]
            # Loss spike = step loss more than 25% above its running average (after warm-up).
            if ema is not None and state["step"] > tcfg["warmup_steps"] and loss_val > 1.25 * ema:
                state["loss_spikes"] += 1
                logger.log("loss_spike", epoch=epoch, step=state["step"], loss=loss_val, running_avg=ema)
            ema = loss_val if ema is None else 0.98 * ema + 0.02 * loss_val
            if state["step"] % tcfg["log_every"] == 0:
                logger.log("step", epoch=epoch, step=state["step"], loss=loss_val, grad_norm=grad_norm, lr=lr)
        sync(device)
        epoch_seconds = time.perf_counter() - t0
        tokens = steps_per_epoch * tcfg["batch_size"] * T
        state["train_seconds"] += epoch_seconds
        state["tokens_seen"] += tokens

        val = score_stream(model, val_stream, T, tcfg["batch_size"] * 2, device, amp)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(1, n_steps), "val_loss": val["ce_nats"],
               "val_perplexity": val["perplexity"], "val_bpc": val["bpc"], "val_top1_accuracy": val["top1_accuracy"],
               "epoch_seconds": epoch_seconds, "tokens_per_sec": tokens / epoch_seconds, "lr_end": lr,
               "window_offset": offset}
        state["history"].append(rec)
        state["epoch"] = epoch
        logger.log("epoch", **rec)
        print(f"epoch {epoch:2d}: train_loss={rec['train_loss']:.4f} val_loss={rec['val_loss']:.4f} "
              f"ppl={rec['val_perplexity']:.3f} acc={rec['val_top1_accuracy']:.4f} "
              f"({epoch_seconds:.0f}s, {rec['tokens_per_sec']:,.0f} tok/s)")

        extra = {"run_id": run_id, "raw_log": raw_log_rel, "epoch": epoch, "val_loss": val["ce_nats"]}
        if val["ce_nats"] < state["best_val"]:
            state["best_val"], state["best_epoch"] = val["ce_nats"], epoch
            save_atomic(weights_payload(model, cfg, tok, extra), ckpt_dir / "best.pt")
        save_atomic({**weights_payload(model, cfg, tok, extra), "optimizer": opt.state_dict(), "state": state}, latest)

    steps_ok = max(1, state["step"] - state["nan_count"])
    summary = {"run_id": run_id, "raw_log": raw_log_rel, "parameter_count": n_params, "epochs": state["epoch"],
               "total_steps": state["step"], "best_epoch": state["best_epoch"], "best_val_loss": state["best_val"],
               "total_training_seconds": state["train_seconds"],
               "training_tokens_per_sec": state["tokens_seen"] / max(1e-9, state["train_seconds"]),
               "grad_norm_mean": state["grad_norm_sum"] / steps_ok, "grad_norm_max": state["grad_norm_max"],
               "grad_norm_max_step": state["grad_norm_max_step"], "grad_clip_max_norm": tcfg["gradient_clip_max_norm"],
               "nan_count": state["nan_count"], "loss_spikes": state["loss_spikes"], **peak_memory(device),
               "hardware": hw, "history": state["history"],
               "checkpoints": {"final": (ckpt_dir / "model.pt").relative_to(REPO_DIR).as_posix(),
                               "best": (ckpt_dir / "best.pt").relative_to(REPO_DIR).as_posix()}}
    save_atomic(weights_payload(model, cfg, tok, {"run_id": run_id, "raw_log": raw_log_rel, "epoch": state["epoch"],
                                                  "train_summary": summary}), ckpt_dir / "model.pt")
    logger.log("end", **{k: v for k, v in summary.items() if k != "history"})
    logger.close()
    return summary


def load_checkpoint(path, device):
    from data import CharTokenizer
    saved = torch.load(path, map_location=device, weights_only=False)
    tok = CharTokenizer(saved["char_to_idx"])
    model = build_model(saved["config"], tok.vocab_size).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, tok, saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="start over even if checkpoints/latest.pt exists")
    args = ap.parse_args()
    cfg = load_config()
    data = load_prepared() if (DATA_DIR / "train_stream.npy").exists() else prepare(cfg)
    summary = train(cfg, data, get_device(), resume=not args.fresh)
    print({k: v for k, v in summary.items() if k not in ("history", "hardware")})


if __name__ == "__main__":
    main()

"""Task 1 (Step 6): training pipeline for Khushi's from-scratch GPT model.

Implements the reusable pieces (dataloaders, optimizer/scheduler from
config, training/validation loops, gradient-norm tracking, NaN/Inf
detection, checkpointing, raw log writing, and temperature-based text
generation). All values come from task1_config.json — nothing here
introduces a new architecture or hyperparameter choice.

Running this file directly does NOT start the full 10-epoch training run.
hardware.device/hardware.gpu are intentionally left blank in the config
until that real run happens. See check_pipeline.py for lightweight wiring
checks and smoke_test.py for a tiny real run on a subset of the data.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import build_model_from_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task1_config.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data_processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
RAW_LOG_DIR = Path(__file__).resolve().parents[3] / "reproducibility" / "raw_logs" / "khushi"

OPTIMIZERS = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


def load_full_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


class RawLogger:
    """Appends one JSON record per line to a uniquely-named, timestamped
    log file. The file is opened in append mode only and is never
    truncated or rewritten, so raw logs from a run are preserved exactly
    as produced."""

    def __init__(self, log_dir=RAW_LOG_DIR):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = log_dir / f"task1_train_log_{timestamp}.jsonl"

    def log(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def build_dataloaders(training_cfg):
    train_inputs = np.load(DATA_DIR / "train_inputs.npy")
    train_targets = np.load(DATA_DIR / "train_targets.npy")
    val_inputs = np.load(DATA_DIR / "validation_inputs.npy")
    val_targets = np.load(DATA_DIR / "validation_targets.npy")

    train_dataset = TensorDataset(torch.from_numpy(train_inputs).long(), torch.from_numpy(train_targets).long())
    val_dataset = TensorDataset(torch.from_numpy(val_inputs).long(), torch.from_numpy(val_targets).long())

    train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg["batch_size"], shuffle=False)
    return train_loader, val_loader


def build_optimizer(model, training_cfg):
    name = training_cfg["optimizer"]
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{name}' in config.")
    kwargs = {"lr": training_cfg["learning_rate"]}
    if "weight_decay" in training_cfg:
        kwargs["weight_decay"] = training_cfg["weight_decay"]
    return OPTIMIZERS[name](model.parameters(), **kwargs)


def build_scheduler(optimizer, training_cfg, total_steps):
    name = training_cfg["scheduler"]
    warmup_steps = training_cfg["warmup_steps"]

    if name != "cosine":
        raise ValueError(f"Unsupported scheduler '{name}' in config.")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_grad_norm(model):
    """Global L2 norm across all parameter gradients. This is reported
    only — it is not used to clip gradients, since no clipping threshold
    exists in task1_config.json."""
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().pow(2).sum().item()
    return total_sq ** 0.5


def check_finite(loss_value, grad_norm, step):
    if not math.isfinite(loss_value):
        raise RuntimeError(f"Non-finite loss detected at step {step}: {loss_value}")
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"Non-finite gradient norm detected at step {step}: {grad_norm}")


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, full_cfg, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": full_cfg,
    }, output_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def train_one_epoch(model, loader, optimizer, scheduler, logger, device, epoch, global_step, max_steps=None):
    model.train()
    epoch_losses = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        _, loss = model(inputs, targets=targets)
        loss.backward()

        loss_value = loss.item()
        grad_norm = compute_grad_norm(model)
        check_finite(loss_value, grad_norm, global_step)

        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss_value)
        logger.log({
            "type": "train_step",
            "epoch": epoch,
            "global_step": global_step,
            "batch_idx": batch_idx,
            "train_loss": loss_value,
            "grad_norm": grad_norm,
            "lr": scheduler.get_last_lr()[0],
        })

        global_step += 1
        if max_steps is not None and global_step >= max_steps:
            break

    mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
    return mean_loss, global_step


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets=targets)
            losses.append(loss.item())
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def generate(model, prompt_indices, max_new_tokens, temperature, sequence_length, device):
    """Autoregressive temperature-sampling generation, matching
    generation.method == "temperature" in task1_config.json."""
    model.eval()
    idx = torch.tensor(prompt_indices, dtype=torch.long, device=device).unsqueeze(0)  # (1, T0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = idx[:, -sequence_length:]  # crop to the model's context window
            logits, _ = model(context)
            next_logits = logits[:, -1, :] / temperature  # (1, vocab_size)
            probs = F.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx = torch.cat([idx, next_idx], dim=1)
    model.train()
    return idx[0].tolist()


def run_training(model, full_cfg, device, checkpoint_dir=CHECKPOINT_DIR, logger=None,
                  max_epochs=None, max_steps_per_epoch=None, max_eval_batches=None):
    """Full training driver. Epoch count defaults to
    full_cfg['training']['epochs']; max_epochs/max_steps_per_epoch/
    max_eval_batches are execution-only overrides for smoke testing, not
    new hyperparameter choices."""
    training_cfg = full_cfg["training"]
    train_loader, val_loader = build_dataloaders(training_cfg)

    optimizer = build_optimizer(model, training_cfg)
    steps_per_epoch = len(train_loader) if max_steps_per_epoch is None else min(len(train_loader), max_steps_per_epoch)
    epochs = training_cfg["epochs"] if max_epochs is None else max_epochs
    total_steps = steps_per_epoch * epochs
    scheduler = build_scheduler(optimizer, training_cfg, total_steps)

    logger = logger or RawLogger()

    n_params = model.num_parameters()
    logger.log({"type": "run_start", "num_parameters": n_params, "config": full_cfg})
    print(f"Parameter count: {n_params:,}")

    global_step = 0
    history = []
    start_time = datetime.now()

    for epoch in range(epochs):
        epoch_start = datetime.now()
        step_limit = (global_step + max_steps_per_epoch) if max_steps_per_epoch is not None else None
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, logger, device, epoch, global_step,
            max_steps=step_limit,
        )
        val_loss = evaluate(model, val_loader, device, max_batches=max_eval_batches)
        epoch_seconds = (datetime.now() - epoch_start).total_seconds()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": epoch_seconds})
        logger.log({
            "type": "epoch_end",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_seconds,
        })
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({epoch_seconds:.1f}s)")

        checkpoint_path = Path(checkpoint_dir) / f"epoch_{epoch}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, full_cfg, checkpoint_path)
        logger.log({"type": "checkpoint_saved", "epoch": epoch, "path": str(checkpoint_path)})

    total_seconds = (datetime.now() - start_time).total_seconds()
    logger.log({"type": "run_end", "total_seconds": total_seconds})
    print(f"Total training time: {total_seconds:.1f}s")

    return history, logger.path


if __name__ == "__main__":
    # Intentionally does NOT start the full run automatically. Running this
    # file directly only reports the config-driven training plan; the real
    # 10-epoch run is deferred until the actual GPU run.
    cfg = load_full_config()
    train_loader, val_loader = build_dataloaders(cfg["training"])
    print("Training pipeline loaded (not executed). Full run deferred until the GPU run:")
    print(f"  train batches/epoch: {len(train_loader)}, val batches: {len(val_loader)}")
    print(f"  planned epochs (from config): {cfg['training']['epochs']}")

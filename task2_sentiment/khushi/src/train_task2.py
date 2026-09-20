"""Task 2 (Step 5): training pipeline shared by Khushi's three Yelp
Polarity classifiers (baseline, TextCNN, bidirectional GRU).

Loads the already-finalized encoded arrays from data_processed/ (produced
by build_processed_data.py), builds the requested model from
task2_config.json, and provides train/eval loops, checkpoint save/load,
and gradient-clipping support (used only by the GRU, per its config).

Running this file directly does NOT start full training -- it only
reports the training plan. See smoke_test_task2.py for a tiny real run.
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_model

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task2_config.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data_processed"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
RAW_LOG_DIR = Path(__file__).resolve().parents[3] / "reproducibility" / "raw_logs" / "khushi"

OPTIMIZERS = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD}


class RawLogger:
    """Appends one JSON record per line to a uniquely-named, timestamped,
    per-model log file. The file is opened in append mode only and is
    never truncated or rewritten, so raw logs from a run are preserved
    exactly as produced.

    Every record automatically gets a "run_type" field (e.g. "full_train"
    for a real GPU training run, "smoke_test" for a tiny local check), so
    later log analysis can filter real runs from smoke tests without
    touching existing log files."""

    def __init__(self, model_name, run_type, log_dir=RAW_LOG_DIR):
        self.run_type = run_type
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = log_dir / f"task2_{model_name}_train_log_{timestamp}.jsonl"

    def log(self, record):
        record = {"run_type": self.run_type, **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    with open(DATA_DIR / "vocab.json", "r") as f:
        vocab = json.load(f)
    config["_vocab_size"] = vocab["vocab_size"]
    return config


def load_split_arrays(split_name):
    input_ids = np.load(DATA_DIR / f"{split_name}_input_ids.npy")
    labels = np.load(DATA_DIR / f"{split_name}_labels.npy")
    return input_ids, labels


def build_dataloader(input_ids, labels, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(input_ids).long(),
        torch.from_numpy(labels).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_optimizer(model, model_cfg):
    name = model_cfg["optimizer"]
    if name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{name}' in config.")
    kwargs = {"lr": model_cfg["learning_rate"]}
    if "weight_decay" in model_cfg:
        kwargs["weight_decay"] = model_cfg["weight_decay"]
    return OPTIMIZERS[name](model.parameters(), **kwargs)


def train_one_epoch(model, loader, optimizer, device, grad_clip_max_norm=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss detected: {loss_value}")

        if grad_clip_max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

        optimizer.step()
        losses.append(loss_value)

    return sum(losses) / len(losses) if losses else float("nan")


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    accuracy = correct / total if total > 0 else float("nan")
    return mean_loss, accuracy


def save_checkpoint(model, optimizer, epoch, global_step, model_name, config, output_path,
                     num_parameters=None, training_seconds=None, examples_per_sec=None,
                     peak_memory_bytes=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config["models"][model_name],
        "num_parameters": num_parameters,
        "training_seconds": training_seconds,
        "examples_per_sec": examples_per_sec,
        "peak_memory_bytes": peak_memory_bytes,
    }, output_path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def run_training(model, model_name, config, device, max_epochs=None, logger=None,
                  train_arrays=None, val_arrays=None, checkpoint_dir=CHECKPOINT_DIR,
                  run_type="full_train"):
    """Full training driver for one model. Epoch count defaults to
    config['models'][model_name]['epochs']; max_epochs is an
    execution-only override for smoke testing, not a new hyperparameter
    choice. train_arrays/val_arrays let callers pass a tiny data subset
    for smoke testing instead of the full saved arrays.

    Writes one timestamped, append-only JSONL raw log record per epoch
    to reproducibility/raw_logs/khushi/ (never overwritten -- a fresh
    RawLogger is created per call, each with its own timestamped file).
    Every record's run_type defaults to "full_train" (a real run); pass
    run_type="smoke_test" (or supply your own `logger`) for tiny local
    checks, so real GPU runs stay easy to distinguish from smoke tests."""
    model_cfg = config["models"][model_name]

    train_ids, train_labels = train_arrays if train_arrays is not None else load_split_arrays("train")
    val_ids, val_labels = val_arrays if val_arrays is not None else load_split_arrays("validation")

    train_loader = build_dataloader(train_ids, train_labels, model_cfg["batch_size"], shuffle=True)
    val_loader = build_dataloader(val_ids, val_labels, model_cfg["batch_size"], shuffle=False)

    optimizer = build_optimizer(model, model_cfg)
    grad_clip = model_cfg.get("gradient_clip_max_norm")
    epochs = model_cfg["epochs"] if max_epochs is None else max_epochs

    logger = logger or RawLogger(model_name, run_type=run_type)
    n_params = sum(p.numel() for p in model.parameters())

    history = []
    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, grad_clip_max_norm=grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, device)

        epoch_seconds = time.time() - epoch_start
        n_examples = len(train_loader.dataset)
        examples_per_sec = n_examples / epoch_seconds if epoch_seconds > 0 else float("inf")
        peak_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        current_lr = optimizer.param_groups[0]["lr"]

        checkpoint_path = Path(checkpoint_dir) / f"{model_name}_epoch_{epoch}.pt"
        save_checkpoint(
            model, optimizer, epoch, len(train_loader) * (epoch + 1), model_name, config, checkpoint_path,
            num_parameters=n_params, training_seconds=epoch_seconds,
            examples_per_sec=examples_per_sec, peak_memory_bytes=peak_memory_bytes,
        )

        record = {
            "model_name": model_name,
            "config": model_cfg,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "learning_rate": current_lr,
            "training_time_seconds": epoch_seconds,
            "examples_per_sec": examples_per_sec,
            "peak_memory_bytes": peak_memory_bytes,
            "checkpoint_path": str(checkpoint_path),
        }
        logger.log(record)
        history.append(record)
        print(f"[{model_name}] epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} ({epoch_seconds:.2f}s, {examples_per_sec:.1f} ex/s)")

    return history, logger.path


if __name__ == "__main__":
    # Intentionally does NOT start full training. Reports the plan only.
    config = load_config()
    for model_name in ["baseline", "experimental_1", "experimental_2"]:
        model_cfg = config["models"][model_name]
        train_ids, train_labels = load_split_arrays("train")
        loader = build_dataloader(train_ids, train_labels, model_cfg["batch_size"], shuffle=True)
        print(f"{model_name}: architecture={model_cfg['architecture']}, "
              f"batch_size={model_cfg['batch_size']}, batches/epoch={len(loader)}, "
              f"planned epochs={model_cfg['epochs']}")

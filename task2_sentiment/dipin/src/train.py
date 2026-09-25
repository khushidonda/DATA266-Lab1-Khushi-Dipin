"""Task 2 (Dipin): training loop shared by all three models.

Every run writes an append-only JSONL raw log (one line per event) and saves the
best-validation-macro-F1 checkpoint with its config, hardware and timing metadata.
"""

import json
import math
import platform
import resource
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from models import build_model, count_parameters

TASK_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = TASK_DIR / "checkpoints"
REPO_DIR = TASK_DIR.parent.parent
# Real runs log straight into the team evidence folder; smoke runs stay in git-ignored outputs/smoke/.
LOG_DIR = REPO_DIR / "reproducibility" / "raw_logs" / "dipin"
SMOKE_LOG_DIR = TASK_DIR / "outputs" / "smoke" / "raw_logs"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def hardware_info(device):
    info = {"device": str(device), "cpu": cpu_name(), "platform": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 2), cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version())
    elif device.type == "mps":
        info["gpu"] = "Apple MPS (integrated GPU)"
    else:
        info["gpu"] = "none"
    return info


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    """Peak accelerator memory allocated by tensors, plus process peak RSS (host RAM)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if platform.system() == "Darwin" else rss * 1024
    acc = None
    if device.type == "cuda":
        acc = torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        acc = torch.mps.driver_allocated_memory()
    return {"peak_accelerator_bytes": acc, "peak_host_rss_bytes": rss_bytes}


def make_loader(ids, lengths, labels, batch_size, shuffle, seed, num_workers=0):
    ds = TensorDataset(torch.from_numpy(ids), torch.from_numpy(lengths), torch.from_numpy(labels))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


class RawLogger:
    def __init__(self, run_id, log_dir=LOG_DIR):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{run_id}.jsonl"
        self.f = open(self.path, "a", buffering=1)

    def log(self, event, **kw):
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **kw}
        self.f.write(json.dumps(rec) + "\n")

    def close(self):
        self.f.close()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for ids, lengths, y in loader:
        logits = model(ids.to(device).long(), lengths)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def lr_lambda_factory(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


def train_model(name, cfg, data, vocab_size, device, epochs=None, tag=""):
    """Trains one model. `data` holds ids/lengths/labels dicts keyed by split name.

    Returns a summary dict and the path to the best checkpoint.
    """
    tcfg = cfg["training"]
    epochs = epochs or tcfg["epochs"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    run_id = f"task2_{name}{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = RawLogger(run_id, SMOKE_LOG_DIR if "smoke" in tag else LOG_DIR)
    hw = hardware_info(device)

    model = build_model(name, cfg, vocab_size).to(device)
    n_params = count_parameters(model)
    train_loader = make_loader(data["ids"]["train"], data["lengths"]["train"], data["labels"]["train"],
                               tcfg["batch_size"], True, cfg["seed"], tcfg.get("num_workers", 0))
    val_loader = make_loader(data["ids"]["validation"], data["lengths"]["validation"], data["labels"]["validation"],
                             1024, False, cfg["seed"])

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    total_steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda_factory(total_steps, max(1, int(tcfg["warmup_fraction"] * total_steps))))
    loss_fn = nn.BCEWithLogitsLoss()

    logger.log("start", run_id=run_id, model=name, config=cfg["models"][name], training=tcfg,
               epochs=epochs, parameter_count=n_params, hardware=hw,
               n_train=len(data["labels"]["train"]), n_val=len(data["labels"]["validation"]))
    print(f"[{name}] params={n_params:,} device={hw.get('gpu')} run_id={run_id}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{name}{tag}.pt"
    history, best_f1, bad_epochs = [], -1.0, 0
    reset_peak_memory(device)
    train_seconds, n_seen, step = 0.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, count, nan_steps, grad_norms = 0.0, 0, 0, []
        for ids, lengths, y in train_loader:
            ids, y = ids.to(device, non_blocking=True).long(), y.to(device, non_blocking=True).float()
            logits = model(ids, lengths)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["gradient_clip_max_norm"]).item()
            if not math.isfinite(loss.item()) or not math.isfinite(gn):
                nan_steps += 1
                continue
            opt.step()
            sched.step()
            step += 1
            running += loss.item() * len(y)
            count += len(y)
            grad_norms.append(gn)
            if step % 200 == 0:
                logger.log("step", epoch=epoch, step=step, loss=loss.item(), grad_norm=gn, lr=sched.get_last_lr()[0])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        epoch_time = time.perf_counter() - t0
        train_seconds += epoch_time
        n_seen += count

        y_val, p_val = predict(model, val_loader, device)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(p_val).clamp(1e-7, 1 - 1e-7),
                                                             torch.tensor(y_val).float()))
        val_pred = (p_val >= 0.5).astype(int)
        val_acc = float((val_pred == y_val).mean())
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        rec = {"epoch": epoch, "train_loss": running / max(1, count), "val_loss": val_loss, "val_accuracy": val_acc,
               "val_macro_f1": val_f1, "epoch_seconds": epoch_time, "examples_per_sec": count / epoch_time,
               "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
               "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
               "nan_steps": nan_steps, "lr": sched.get_last_lr()[0]}
        history.append(rec)
        logger.log("epoch", **rec)
        print(f"[{name}] epoch {epoch}: train_loss={rec['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({epoch_time:.0f}s, {rec['examples_per_sec']:.0f} ex/s)")

        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            torch.save({"model_name": name, "state_dict": model.state_dict(), "config": cfg,
                        "vocab_size": vocab_size, "epoch": epoch, "val_macro_f1": val_f1}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stopping_patience"]:
                logger.log("early_stop", epoch=epoch)
                print(f"[{name}] early stop at epoch {epoch}")
                break

    mem = peak_memory(device)
    summary = {"run_id": run_id, "model": name, "parameter_count": n_params, "best_val_macro_f1": best_f1,
               "best_epoch": max(history, key=lambda r: r["val_macro_f1"])["epoch"],
               "epochs_run": len(history), "training_seconds": train_seconds,
               "train_examples_per_sec": n_seen / train_seconds, **mem, "hardware": hw,
               "checkpoint": str(ckpt_path.relative_to(REPO_DIR)), "raw_log": str(logger.path.relative_to(REPO_DIR)),
               "history": history}
    # Attach run metadata to the best checkpoint so it is self-describing.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt.update({k: v for k, v in summary.items() if k != "history"})
    ckpt["history"] = history
    torch.save(ckpt, ckpt_path)
    logger.log("end", **{k: v for k, v in summary.items() if k != "history"})
    logger.close()
    return summary, ckpt_path


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["config"], ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

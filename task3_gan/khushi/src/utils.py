"""Task 3: shared infrastructure -- config loading, seeding, raw JSONL
logging, checkpoint save/load with SHA-256, device selection, and
finite-value / gradient-norm helpers. Used by train.py, generate.py, and
smoke_test.py so none of them duplicate this logic."""

import hashlib
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task3_config.json"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
RAW_LOG_DIR = Path(__file__).resolve().parents[3] / "reproducibility" / "raw_logs" / "khushi"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_id(prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_grad_norm(parameters):
    total_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq += p.grad.detach().float().pow(2).sum().item()
    return total_sq ** 0.5


def check_finite(value, name, step):
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite value detected for '{name}' at step {step}: {value}")


def gpu_memory_bytes(device):
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
        }
    return {"allocated": None, "reserved": None}


class RawLogger:
    """Appends one JSON record per line to a run_id-named log file under
    reproducibility/raw_logs/khushi/. Refuses to reuse an existing log
    file name (never overwrites/edits a prior run's log). Every record
    gets a wall-clock timestamp and the run_id automatically."""

    def __init__(self, run_id, log_dir=RAW_LOG_DIR):
        self.run_id = run_id
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{run_id}.jsonl"
        if self.path.exists():
            raise FileExistsError(f"Raw log already exists, refusing to reuse it: {self.path}")

    def log(self, record):
        stamped = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
        }
        stamped.update(record)
        with open(self.path, "a") as f:
            f.write(json.dumps(stamped, default=str) + "\n")


def get_rng_states():
    state = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_states(state):
    """torch.set_rng_state()/set_rng_state_all() require plain CPU
    ByteTensors regardless of what device the rest of the checkpoint was
    loaded onto -- if load_checkpoint was called with map_location set to
    a non-CPU device, torch.load relocates every tensor in the payload,
    including these, which would otherwise raise
    "RNG state must be a torch.ByteTensor". Force .cpu() here explicitly
    so restoring RNG state never depends on map_location."""
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def save_checkpoint(path, epoch, global_step, config, models, optimizers, schedulers,
                     include_rng_state=True):
    """models / optimizers / schedulers: dicts of {name: module_or_optimizer_or_scheduler}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "model_state_dicts": {name: m.state_dict() for name, m in models.items()},
        "optimizer_state_dicts": {name: o.state_dict() for name, o in optimizers.items()},
        "scheduler_state_dicts": {name: s.state_dict() for name, s in schedulers.items()},
    }
    if include_rng_state:
        payload["rng_states"] = get_rng_states()
    torch.save(payload, path)
    return sha256_file(path)


def load_checkpoint(path, models, optimizers=None, schedulers=None, map_location="cpu",
                     restore_rng_state=False):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    for name, m in models.items():
        m.load_state_dict(checkpoint["model_state_dicts"][name])
    if optimizers is not None:
        for name, o in optimizers.items():
            o.load_state_dict(checkpoint["optimizer_state_dicts"][name])
    if schedulers is not None:
        for name, s in schedulers.items():
            s.load_state_dict(checkpoint["scheduler_state_dicts"][name])
    if restore_rng_state and "rng_states" in checkpoint:
        set_rng_states(checkpoint["rng_states"])
    return checkpoint

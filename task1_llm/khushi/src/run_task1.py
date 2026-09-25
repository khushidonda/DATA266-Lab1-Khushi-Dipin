"""Task 1: launcher for Khushi's GPU runs of the existing training pipeline.

    python run_task1.py --mode full    # the real 10-epoch run (all values from task1_config.json)
    python run_task1.py --mode smoke   # short GPU smoke run of the same code path

Execution/reproducibility only: no architecture or hyperparameter value is
chosen here. What it adds around train.run_training():
  - seeds random / NumPy / PyTorch (CPU + CUDA) with config["seed"] right
    before the model is built, so weight init, shuffling and dropout are seeded;
  - one run ID (task1_<mode>_<UTC timestamp>) shared by the raw log, the
    checkpoint folder and the outputs folder; smoke runs are kept apart under
    checkpoints/smoke/ and never write to outputs/;
  - environment info (GPU, library versions, Docker image, git commit) and
    SHA-256 hashes of the code, config, tokenizer, split and sequence arrays
    in the raw log's run_start record.
"""

import argparse
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from model import build_model_from_config
from train import CHECKPOINT_DIR, CONFIG_PATH, DATA_DIR, RawLogger, load_full_config, run_training, sha256_file

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[2]
OUTPUTS_DIR = SRC_DIR.parent / "outputs"

# Execution-only limits for --mode smoke (not hyperparameters): one short
# epoch at the configured batch size, just enough to exercise the full path.
SMOKE_OVERRIDES = {"max_epochs": 1, "max_steps_per_epoch": 100, "max_eval_batches": 20}

HASHED_FILES = [
    SRC_DIR / "model.py",
    SRC_DIR / "train.py",
    SRC_DIR / "run_task1.py",
    CONFIG_PATH,
    DATA_DIR / "tokenizer.json",
    DATA_DIR / "split_indices.json",
    DATA_DIR / "sequences_metadata.json",
    DATA_DIR / "train_inputs.npy",
    DATA_DIR / "train_targets.npy",
    DATA_DIR / "validation_inputs.npy",
    DATA_DIR / "validation_targets.npy",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment_info(device):
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "gpu_name": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_compute_capability": f"{props.major}.{props.minor}",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        # Set by the host-side `docker run -e ...` command (the container has no git).
        "docker_image": os.environ.get("TASK1_DOCKER_IMAGE"),
        "git_commit": os.environ.get("TASK1_GIT_COMMIT"),
        "git_worktree_dirty": os.environ.get("TASK1_GIT_DIRTY"),
    }


def launch(mode):
    """Runs train.run_training() in the given mode and returns the run's
    identifiers plus the trained model (used by smoke_test_gpu.py)."""
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to run the GPU launcher on CPU.")
    device = torch.device("cuda")

    full_cfg = load_full_config()
    seed = full_cfg["seed"]
    overrides = SMOKE_OVERRIDES if mode == "smoke" else {}

    run_id = f"task1_{mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if mode == "smoke":
        checkpoint_dir = CHECKPOINT_DIR / "smoke" / run_id
        output_dir = None
    else:
        checkpoint_dir = CHECKPOINT_DIR / run_id
        output_dir = OUTPUTS_DIR / run_id
    for d in (checkpoint_dir, output_dir):
        if d is not None and d.exists():
            raise FileExistsError(f"{d} already exists; refusing to reuse a run folder.")
    logger = RawLogger(run_id=run_id)  # raises if this run's raw log already exists
    print(f"Run ID: {run_id}\nRaw log: {logger.path}\nCheckpoints: {checkpoint_dir}")

    run_info = {
        "mode": mode,
        "seed": seed,
        "seeded": ["random", "numpy", "torch", "torch.cuda"],
        "execution_overrides": overrides,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir) if output_dir else None,
        "environment": environment_info(device),
        "sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in HASHED_FILES},
    }

    set_seed(seed)
    model, _ = build_model_from_config()
    model.to(device)

    history, log_path = run_training(model, full_cfg, device, checkpoint_dir=checkpoint_dir, logger=logger,
                                     run_info=run_info, **overrides)

    summary = {
        "run_id": run_id,
        "mode": mode,
        "raw_log": str(log_path),
        "checkpoint_dir": str(checkpoint_dir),
        "history": history,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True)
        with open(output_dir / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Run summary: {output_dir / 'run_summary.json'}")
    print(f"Run {run_id} finished.")
    return {**summary, "model": model, "device": device, "full_cfg": full_cfg}


def main():
    parser = argparse.ArgumentParser(description="Launch a Task 1 GPU run (full or smoke).")
    parser.add_argument("--mode", choices=["full", "smoke"], required=True)
    launch(parser.parse_args().mode)


if __name__ == "__main__":
    main()

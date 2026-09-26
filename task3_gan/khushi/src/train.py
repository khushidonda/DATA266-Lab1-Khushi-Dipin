"""Task 3: CycleGAN training pipeline for Khushi's locked configuration.

Implements the full per-step training sequence (forward passes, adversarial
+ cycle + identity losses, generator update, both discriminator updates via
independent replay pools), the exact locked LR schedule, checkpointing, and
raw JSONL logging (one record per optimizer step, matching the granularity
used for Task 1's real GPU run, plus per-epoch and special-event records).

Running this file directly does NOT start the 200-epoch training run --
it only builds the pieces and reports the plan. See smoke_test.py for a
tiny, non-training-scale exercise of this same code path.
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import UnalignedMonetPhotoDataset
from image_pool import ImagePool
from losses import (
    cycle_consistency_loss,
    identity_loss,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)
from models import build_discriminator, build_generator, count_parameters, set_requires_grad
from utils import (
    CHECKPOINT_DIR,
    RawLogger,
    compute_grad_norm,
    gpu_memory_bytes,
    load_checkpoint,
    load_config,
    make_run_id,
    save_checkpoint,
    select_device,
    set_global_seed,
    sha256_file,
)


class TrainingDiverged(RuntimeError):
    """Raised after a NaN/Inf event has already been logged, so callers
    never need to log it themselves -- the failure record survives even
    though this then propagates and aborts the run."""


def build_models(device):
    models = {
        "G_A2B": build_generator().to(device),
        "G_B2A": build_generator().to(device),
        "D_A": build_discriminator().to(device),
        "D_B": build_discriminator().to(device),
    }
    return models


def build_optimizers(models, training_cfg):
    lr = training_cfg["learning_rate"]
    betas = (training_cfg["beta1"], training_cfg["beta2"])
    optimizer_G = torch.optim.Adam(
        list(models["G_A2B"].parameters()) + list(models["G_B2A"].parameters()),
        lr=lr, betas=betas,
    )
    optimizer_D_A = torch.optim.Adam(models["D_A"].parameters(), lr=lr, betas=betas)
    optimizer_D_B = torch.optim.Adam(models["D_B"].parameters(), lr=lr, betas=betas)
    return {"G": optimizer_G, "D_A": optimizer_D_A, "D_B": optimizer_D_B}


def make_lr_lambda(constant_epochs, decay_epochs):
    """Exact CycleGAN convention. `epoch` here is PyTorch's 0-indexed
    last_epoch (epoch 0 = before any .step() call, corresponds to the
    user-facing "epoch 1"). Verified boundary behavior:
      epoch 1   (last_epoch=0)   -> multiplier 1.0
      epoch 100 (last_epoch=99)  -> multiplier 1.0   (last constant epoch)
      epoch 101 (last_epoch=100) -> multiplier ~0.9901 (decay begins)
      epoch 150 (last_epoch=149) -> multiplier ~0.5050
      epoch 200 (last_epoch=199) -> multiplier ~0.0099 (approaches, doesn't hit, zero)
    """
    def lr_lambda(last_epoch):
        return 1.0 - max(0, last_epoch + 1 - constant_epochs) / float(decay_epochs + 1)
    return lr_lambda


def build_schedulers(optimizers, training_cfg):
    schedule_cfg = training_cfg["lr_schedule"]
    constant_epochs = schedule_cfg["constant_epochs"][1]  # e.g. [1, 100] -> 100
    decay_epochs = schedule_cfg["decay_epochs"][1] - schedule_cfg["decay_epochs"][0] + 1  # [101,200] -> 100
    lr_lambda = make_lr_lambda(constant_epochs, decay_epochs)
    return {
        name: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        for name, opt in optimizers.items()
    }


def build_dataloader(monet_dir, photo_dir, config, seed):
    dataset = UnalignedMonetPhotoDataset(
        monet_dir, photo_dir,
        load_size=config["training"]["augmentation"]["resize"],
        crop_size=config["image_size"],
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    return dataset, loader


def _check_finite_logged(value, component, epoch, global_step, logger):
    """Like utils.check_finite, but first appends a "nan_detected" JSONL
    record (surviving the crash) before raising, so the failure is never
    silently lost when the process aborts."""
    if math.isfinite(value):
        return
    if logger is not None:
        logger.log({
            "event": "nan_detected",
            "epoch": epoch,
            "global_step": global_step,
            "component": component,
            "value": value,
        })
    raise TrainingDiverged(f"Non-finite value detected for '{component}' at step {global_step}: {value}")


def train_one_step(batch, models, optimizers, pools, training_cfg, device, epoch, global_step, logger=None):
    real_A = batch["A"].to(device)  # Monet
    real_B = batch["B"].to(device)  # Photo

    lambda_cycle = training_cfg["cycle_consistency_weight"]
    lambda_identity = training_cfg["identity_loss_weight"]

    # ---- Forward passes (exact locked sequence) ----
    fake_B = models["G_A2B"](real_A)
    rec_A = models["G_B2A"](fake_B)

    fake_A = models["G_B2A"](real_B)
    rec_B = models["G_A2B"](fake_A)

    idt_A = models["G_B2A"](real_A)
    idt_B = models["G_A2B"](real_B)

    # ---- Generator update ----
    # Freeze D_A/D_B parameters (requires_grad=False, NOT torch.no_grad()) so
    # loss_G.backward() cannot accumulate gradients on the discriminators'
    # own parameters. The forward calls below still build a graph back
    # through D into fake_A/fake_B, so gradients still flow into the
    # generators exactly as before -- only D's own .grad stays untouched.
    set_requires_grad([models["D_A"], models["D_B"]], False)

    optimizers["G"].zero_grad()

    loss_G_A2B_adv = lsgan_generator_loss(models["D_B"](fake_B))
    loss_G_B2A_adv = lsgan_generator_loss(models["D_A"](fake_A))

    loss_cycle_A_raw = cycle_consistency_loss(real_A, rec_A)
    loss_cycle_B_raw = cycle_consistency_loss(real_B, rec_B)
    loss_idt_A_raw = identity_loss(real_A, idt_A)
    loss_idt_B_raw = identity_loss(real_B, idt_B)

    loss_G = (
        loss_G_A2B_adv + loss_G_B2A_adv
        + lambda_cycle * (loss_cycle_A_raw + loss_cycle_B_raw)
        + lambda_identity * (loss_idt_A_raw + loss_idt_B_raw)
    )
    loss_G.backward()
    grad_norm_G = compute_grad_norm(
        list(models["G_A2B"].parameters()) + list(models["G_B2A"].parameters())
    )
    _check_finite_logged(loss_G.item(), "loss_G", epoch, global_step, logger)
    _check_finite_logged(grad_norm_G, "grad_norm_G", epoch, global_step, logger)
    optimizers["G"].step()

    # ---- Discriminator updates: unfreeze D_A/D_B ----
    set_requires_grad([models["D_A"], models["D_B"]], True)

    # ---- Discriminator A update (judges Monet real vs. fake) ----
    optimizers["D_A"].zero_grad()
    fake_A_for_D = pools["fake_A"].query(fake_A.detach())
    loss_D_A, loss_D_A_real, loss_D_A_fake = lsgan_discriminator_loss(
        models["D_A"](real_A), models["D_A"](fake_A_for_D)
    )
    loss_D_A.backward()
    grad_norm_D_A = compute_grad_norm(models["D_A"].parameters())
    _check_finite_logged(loss_D_A.item(), "loss_D_A", epoch, global_step, logger)
    optimizers["D_A"].step()

    # ---- Discriminator B update (judges Photo real vs. fake) ----
    optimizers["D_B"].zero_grad()
    fake_B_for_D = pools["fake_B"].query(fake_B.detach())
    loss_D_B, loss_D_B_real, loss_D_B_fake = lsgan_discriminator_loss(
        models["D_B"](real_B), models["D_B"](fake_B_for_D)
    )
    loss_D_B.backward()
    grad_norm_D_B = compute_grad_norm(models["D_B"].parameters())
    _check_finite_logged(loss_D_B.item(), "loss_D_B", epoch, global_step, logger)
    optimizers["D_B"].step()

    return {
        "loss_G": loss_G.item(),
        "loss_G_A2B_adv": loss_G_A2B_adv.item(),
        "loss_G_B2A_adv": loss_G_B2A_adv.item(),
        "loss_cycle_A": loss_cycle_A_raw.item(),
        "loss_cycle_B": loss_cycle_B_raw.item(),
        "loss_idt_A": loss_idt_A_raw.item(),
        "loss_idt_B": loss_idt_B_raw.item(),
        "loss_D_A": loss_D_A.item(),
        "loss_D_A_real": loss_D_A_real.item(),
        "loss_D_A_fake": loss_D_A_fake.item(),
        "loss_D_B": loss_D_B.item(),
        "loss_D_B_real": loss_D_B_real.item(),
        "loss_D_B_fake": loss_D_B_fake.item(),
        "grad_norm_G": grad_norm_G,
        "grad_norm_D_A": grad_norm_D_A,
        "grad_norm_D_B": grad_norm_D_B,
    }


def run_training(monet_dir, photo_dir, config, device, logger, run_id,
                  max_epochs=None, max_steps_per_epoch=None,
                  checkpoint_dir=CHECKPOINT_DIR, checkpoint_every_epochs=1,
                  resume_from=None):
    """Full training driver. max_epochs / max_steps_per_epoch are
    execution-only overrides for smoke testing -- not new hyperparameter
    choices. Epoch/step counts otherwise come entirely from the locked
    config.

    resume_from: optional path to a checkpoint saved by save_checkpoint().
    When given, all 4 model states, all 3 optimizer states, all 3 scheduler
    states, and RNG state (Python/NumPy/CPU-torch/CUDA-torch) are restored,
    and training continues from the next epoch/global_step -- it does not
    restart the LR schedule or step count. `logger` must already be a
    *fresh* RawLogger (a new run_id) -- resuming never reuses or rewrites
    the original run's log file; the new log instead records `resume_from`,
    `resumed_epoch`, `resumed_global_step`, and `checkpoint_sha256` so the
    two logs are traceably linked. `run_id` still controls where new
    checkpoints are written (checkpoints/<run_id>/...) and may be the same
    as the original run's, since epoch numbers only advance from here."""
    training_cfg = config["training"]
    seed = config["seed"]

    dataset, loader = build_dataloader(monet_dir, photo_dir, config, seed)
    models = build_models(device)
    optimizers = build_optimizers(models, training_cfg)
    schedulers = build_schedulers(optimizers, training_cfg)
    pools = {
        "fake_A": ImagePool(pool_size=training_cfg["replay_buffer"]["size_per_domain"], seed=seed),
        "fake_B": ImagePool(pool_size=training_cfg["replay_buffer"]["size_per_domain"], seed=seed + 1),
    }

    param_counts = {name: count_parameters(m) for name, m in models.items()}
    param_counts["total"] = sum(param_counts.values())

    epochs = training_cfg["epochs"] if max_epochs is None else max_epochs

    start_epoch = 0
    global_step = 0

    if resume_from is not None:
        checkpoint_sha256 = sha256_file(resume_from)
        checkpoint = load_checkpoint(
            resume_from, models=models, optimizers=optimizers, schedulers=schedulers,
            map_location=device, restore_rng_state=True,
        )
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        logger.log({
            "type": "resumed",
            "resume_from": str(resume_from),
            "resumed_epoch": checkpoint["epoch"],
            "resumed_global_step": global_step,
            "checkpoint_sha256": checkpoint_sha256,
            "parameter_counts": param_counts,
            "device": str(device),
        })
    else:
        logger.log({
            "type": "run_start",
            "config": config,
            "parameter_counts": param_counts,
            "device": str(device),
            "dataset_sizes": {"monet": len(dataset.monet_paths), "photo": len(dataset.photo_paths)},
            "epoch_length": len(dataset),
        })

    for epoch in range(start_epoch, epochs):  # epoch here is 0-indexed; user-facing "epoch N" = epoch index N-1
        epoch_start = time.time()
        epoch_losses = {}
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                break

            step_start = time.time()
            step_result = train_one_step(batch, models, optimizers, pools, training_cfg, device,
                                          epoch, global_step, logger=logger)
            step_seconds = time.time() - step_start

            mem = gpu_memory_bytes(device)
            current_lr = optimizers["G"].param_groups[0]["lr"]

            record = {
                "type": "train_step",
                "epoch": epoch,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "learning_rate": current_lr,
                "images_per_sec": (1.0 / step_seconds) if step_seconds > 0 else float("inf"),
                "gpu_memory_allocated_bytes": mem["allocated"],
                "gpu_memory_reserved_bytes": mem["reserved"],
            }
            record.update(step_result)
            logger.log(record)

            for k, v in step_result.items():
                epoch_losses.setdefault(k, []).append(v)
            n_batches += 1
            global_step += 1

        for scheduler in schedulers.values():
            scheduler.step()

        epoch_seconds = time.time() - epoch_start
        epoch_summary = {k: sum(v) / len(v) for k, v in epoch_losses.items() if v}
        logger.log({
            "type": "epoch_end",
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "n_batches": n_batches,
            "images_per_sec": (n_batches / epoch_seconds) if epoch_seconds > 0 else float("inf"),
            **{f"mean_{k}": v for k, v in epoch_summary.items()},
        })
        print(f"[epoch {epoch}] loss_G={epoch_summary.get('loss_G', float('nan')):.4f} "
              f"loss_D_A={epoch_summary.get('loss_D_A', float('nan')):.4f} "
              f"loss_D_B={epoch_summary.get('loss_D_B', float('nan')):.4f} ({epoch_seconds:.1f}s)")

        if (epoch + 1) % checkpoint_every_epochs == 0 or epoch == epochs - 1:
            checkpoint_path = Path(checkpoint_dir) / run_id / f"epoch_{epoch}.pt"
            checksum = save_checkpoint(
                checkpoint_path, epoch, global_step, config,
                models=models, optimizers=optimizers, schedulers=schedulers,
            )
            latest_path = Path(checkpoint_dir) / run_id / "latest.pt"
            save_checkpoint(
                latest_path, epoch, global_step, config,
                models=models, optimizers=optimizers, schedulers=schedulers,
            )
            logger.log({
                "type": "checkpoint_saved",
                "epoch": epoch,
                "path": str(checkpoint_path),
                "sha256": checksum,
            })

    logger.log({"type": "run_end", "global_step": global_step})
    return models, optimizers, schedulers, param_counts


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Task 3 CycleGAN training (locked config).")
    parser.add_argument("--monet-dir", type=str, default=None)
    parser.add_argument("--photo-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint.pt to resume from. Restores all 4 model states, "
                              "all 3 optimizer states, all 3 scheduler states, and RNG state; "
                              "continues from the next epoch/global_step. Always writes to a NEW "
                              "raw log (never reuses/overwrites the original run's log).")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Checkpoint directory name for this invocation. Defaults to the "
                              "resumed checkpoint's parent directory name when --resume is given "
                              "(so new checkpoints land alongside the old ones), or a fresh "
                              "timestamped ID for a brand-new run.")
    parser.add_argument("--run", action="store_true",
                         help="Actually start training. Without this flag, only the plan is printed "
                              "(this is the safe default -- see smoke_test.py for a tiny real exercise).")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    config = load_config()
    training_cfg = config["training"]

    if not args.run:
        # Intentionally does NOT start the 200-epoch run. Reports the plan only.
        print("Task 3 training pipeline loaded (not executed). Pass --run to actually start training.")
        print(f"  epochs (locked): {training_cfg['epochs']}")
        print(f"  steps_per_epoch (locked, Policy A = larger domain): {training_cfg['steps_per_epoch']}")
        print(f"  total_steps (locked): {training_cfg['total_steps']}")
        print(f"  batch_size (locked): {training_cfg['batch_size']}")
        if args.resume:
            print(f"  --resume given ({args.resume}) but --run was not passed -- nothing started.")
        print("  Provide --monet-dir/--photo-dir at runtime; see smoke_test.py for a tiny real exercise.")
    else:
        if not args.monet_dir or not args.photo_dir:
            raise SystemExit("--monet-dir and --photo-dir are required with --run.")

        set_global_seed(config["seed"])
        device = select_device()

        if args.resume:
            original_run_id = args.run_id or Path(args.resume).parent.name
            log_run_id = make_run_id(f"{original_run_id}_resumed")
            logger = RawLogger(log_run_id)
            print(f"Resuming from {args.resume} (checkpoint dir: {original_run_id}, new log: {log_run_id})")
            run_training(args.monet_dir, args.photo_dir, config, device, logger,
                         run_id=original_run_id, resume_from=args.resume)
        else:
            run_id = args.run_id or make_run_id("task3_full")
            logger = RawLogger(run_id)
            print(f"Starting new run {run_id}")
            run_training(args.monet_dir, args.photo_dir, config, device, logger, run_id=run_id)

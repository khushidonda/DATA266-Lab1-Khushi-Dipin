"""Task 3: local smoke test for Khushi's locked CycleGAN configuration.

This is NOT the 200-epoch training run. It exercises the full pipeline
(config, seeding, dataset, both generators, both discriminators, all
losses, both optimizer updates, replay pools, checkpointing, image
export, LR schedule, NaN detection) on a tiny number of real batches.

Dataset location is supplied at runtime via environment variables
(TASK3_MONET_DIR, TASK3_PHOTO_DIR) -- never hardcoded here, since this
file is committed to the repo and must not contain a personal path.
"""

import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import UnalignedMonetPhotoDataset, list_images
from generate import denormalize_to_image
from image_pool import ImagePool
from losses import cycle_consistency_loss, identity_loss, lsgan_discriminator_loss, lsgan_generator_loss
from models import build_discriminator, build_generator, count_parameters, set_requires_grad
from train import (
    TrainingDiverged,
    _check_finite_logged,
    build_optimizers,
    build_schedulers,
    make_lr_lambda,
    run_training,
)
from utils import (
    RawLogger,
    check_finite,
    compute_grad_norm,
    load_checkpoint,
    load_config,
    save_checkpoint,
    select_device,
    set_global_seed,
    sha256_file,
)

RESULTS = []


def report(check_num, description, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((check_num, description, status))
    print(f"[{status}] {check_num}. {description}" + (f" -- {detail}" if detail else ""))
    if not passed:
        raise AssertionError(f"Check {check_num} FAILED: {description} {detail}")


def main():
    monet_dir = os.environ.get("TASK3_MONET_DIR")
    photo_dir = os.environ.get("TASK3_PHOTO_DIR")
    if not monet_dir or not photo_dir:
        raise SystemExit(
            "Set TASK3_MONET_DIR and TASK3_PHOTO_DIR environment variables to the local "
            "dataset paths before running this smoke test (not hardcoded, per repo rules)."
        )

    # ---- 1. Config parses correctly ----
    config = load_config()
    required_keys = ["seed", "image_size", "generator", "discriminator", "training", "hardware"]
    missing = [k for k in required_keys if k not in config]
    report(1, "config parses correctly", not missing, f"missing keys: {missing}" if missing else "")

    seed = config["seed"]
    training_cfg = config["training"]

    # ---- 2. Seed applied to Python/NumPy/PyTorch ----
    set_global_seed(seed)
    py_val_1 = random.random()
    np_val_1 = np.random.rand()
    torch_val_1 = torch.rand(1).item()
    set_global_seed(seed)
    py_val_2 = random.random()
    np_val_2 = np.random.rand()
    torch_val_2 = torch.rand(1).item()
    seed_ok = (py_val_1 == py_val_2) and (np_val_1 == np_val_2) and (torch_val_1 == torch_val_2)
    report(2, "seed 42 applied to Python/NumPy/PyTorch (re-seeding reproduces identical draws)", seed_ok)

    device = select_device()
    print(f"    device: {device}")

    # ---- 3. Dataset finds exactly Monet=300, Photo=7038 ----
    monet_paths = list_images(monet_dir)
    photo_paths = list_images(photo_dir)
    counts_ok = (len(monet_paths) == 300) and (len(photo_paths) == 7038)
    report(3, "dataset finds exactly Monet=300, Photo=7038",
           counts_ok, f"found monet={len(monet_paths)}, photo={len(photo_paths)}")

    dataset = UnalignedMonetPhotoDataset(
        monet_dir, photo_dir,
        load_size=training_cfg["augmentation"]["resize"],
        crop_size=config["image_size"],
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch_iter = iter(loader)
    batch = next(batch_iter)

    # ---- 4. One batch shapes ----
    shapes_ok = (tuple(batch["A"].shape) == (1, 3, 256, 256)) and (tuple(batch["B"].shape) == (1, 3, 256, 256))
    report(4, "one batch has A=[1,3,256,256], B=[1,3,256,256]", shapes_ok,
           f"A={tuple(batch['A'].shape)}, B={tuple(batch['B'].shape)}")

    # ---- 5. Normalized values approximately within [-1, 1] ----
    a_min, a_max = batch["A"].min().item(), batch["A"].max().item()
    b_min, b_max = batch["B"].min().item(), batch["B"].max().item()
    range_ok = (-1.001 <= a_min <= 1.001) and (-1.001 <= a_max <= 1.001) and \
               (-1.001 <= b_min <= 1.001) and (-1.001 <= b_max <= 1.001)
    report(5, "normalized values approximately within [-1, 1]", range_ok,
           f"A range=[{a_min:.3f},{a_max:.3f}], B range=[{b_min:.3f},{b_max:.3f}]")

    real_A = batch["A"].to(device)
    real_B = batch["B"].to(device)

    g_a2b = build_generator().to(device)
    g_b2a = build_generator().to(device)
    d_a = build_discriminator().to(device)
    d_b = build_discriminator().to(device)

    param_counts = {
        "G_A2B": count_parameters(g_a2b), "G_B2A": count_parameters(g_b2a),
        "D_A": count_parameters(d_a), "D_B": count_parameters(d_b),
    }
    param_counts["total"] = sum(param_counts.values())
    print(f"    parameter counts: {param_counts}")

    # ---- 6. Both generators forward successfully ----
    fake_B = g_a2b(real_A)
    fake_A = g_b2a(real_B)
    gen_fwd_ok = (tuple(fake_B.shape) == (1, 3, 256, 256)) and (tuple(fake_A.shape) == (1, 3, 256, 256))
    report(6, "both generators forward successfully", gen_fwd_ok,
           f"fake_B={tuple(fake_B.shape)}, fake_A={tuple(fake_A.shape)}")

    # ---- 7. Both discriminators forward successfully ----
    d_a_out = d_a(real_A)
    d_b_out = d_b(real_B)
    disc_fwd_ok = d_a_out.dim() == 4 and d_b_out.dim() == 4 and d_a_out.shape[1] == 1 and d_b_out.shape[1] == 1
    report(7, "both discriminators forward successfully (PatchGAN score maps)", disc_fwd_ok,
           f"D_A output={tuple(d_a_out.shape)}, D_B output={tuple(d_b_out.shape)}")

    # ---- 8. Reconstructed tensors correct dims ----
    rec_A = g_b2a(fake_B)
    rec_B = g_a2b(fake_A)
    rec_ok = (tuple(rec_A.shape) == (1, 3, 256, 256)) and (tuple(rec_B.shape) == (1, 3, 256, 256))
    report(8, "reconstructed tensors have correct dimensions", rec_ok,
           f"rec_A={tuple(rec_A.shape)}, rec_B={tuple(rec_B.shape)}")

    # ---- 9. Identity passes correct dims ----
    idt_A = g_b2a(real_A)
    idt_B = g_a2b(real_B)
    idt_ok = (tuple(idt_A.shape) == (1, 3, 256, 256)) and (tuple(idt_B.shape) == (1, 3, 256, 256))
    report(9, "identity passes have correct dimensions", idt_ok,
           f"idt_A={tuple(idt_A.shape)}, idt_B={tuple(idt_B.shape)}")

    # ---- 10. All losses finite ----
    lambda_cycle = training_cfg["cycle_consistency_weight"]
    lambda_identity = training_cfg["identity_loss_weight"]

    loss_G_A2B_adv = lsgan_generator_loss(d_b(fake_B))
    loss_G_B2A_adv = lsgan_generator_loss(d_a(fake_A))
    loss_cycle_A = cycle_consistency_loss(real_A, rec_A)
    loss_cycle_B = cycle_consistency_loss(real_B, rec_B)
    loss_idt_A = identity_loss(real_A, idt_A)
    loss_idt_B = identity_loss(real_B, idt_B)
    loss_G = (loss_G_A2B_adv + loss_G_B2A_adv
              + lambda_cycle * (loss_cycle_A + loss_cycle_B)
              + lambda_identity * (loss_idt_A + loss_idt_B))

    all_loss_values = [loss_G_A2B_adv, loss_G_B2A_adv, loss_cycle_A, loss_cycle_B,
                        loss_idt_A, loss_idt_B, loss_G]
    losses_finite = all(math.isfinite(v.item()) for v in all_loss_values)
    report(10, "all losses are finite", losses_finite,
           f"loss_G={loss_G.item():.4f}")

    # ---- 11 + 20. Generator backward/update, with D_A/D_B frozen (no retain_graph needed) ----
    optimizers = build_optimizers({"G_A2B": g_a2b, "G_B2A": g_b2a, "D_A": d_a, "D_B": d_b}, training_cfg)

    set_requires_grad([d_a, d_b], False)
    optimizers["G"].zero_grad()
    optimizers["D_A"].zero_grad()
    optimizers["D_B"].zero_grad()

    before = g_a2b.model[1].weight.detach().clone()
    loss_G.backward()  # D frozen, so no retain_graph needed and no stray D grad accumulates
    grad_norm_G = compute_grad_norm(list(g_a2b.parameters()) + list(g_b2a.parameters()))
    check_finite(loss_G.item(), "loss_G", 0)
    check_finite(grad_norm_G, "grad_norm_G", 0)
    optimizers["G"].step()
    after = g_a2b.model[1].weight.detach().clone()
    gen_update_ok = not torch.equal(before, after)
    report(11, "one generator backward/update succeeds (weights changed)", gen_update_ok,
           f"grad_norm_G={grad_norm_G:.4f}")

    d_frozen_no_grad = all(p.grad is None for p in d_a.parameters()) and all(p.grad is None for p in d_b.parameters())
    report(20, "discriminator parameters receive no G-update gradients while frozen", d_frozen_no_grad,
           f"D_A params with grad={sum(p.grad is not None for p in d_a.parameters())}, "
           f"D_B params with grad={sum(p.grad is not None for p in d_b.parameters())}")

    # ---- 12 + 21. Both discriminator backward/updates succeed after unfreezing ----
    set_requires_grad([d_a, d_b], True)

    pool_fake_A = ImagePool(pool_size=training_cfg["replay_buffer"]["size_per_domain"], seed=seed)
    pool_fake_B = ImagePool(pool_size=training_cfg["replay_buffer"]["size_per_domain"], seed=seed + 1)

    before_da = d_a.model[0].weight.detach().clone()
    optimizers["D_A"].zero_grad()
    fake_A_pooled = pool_fake_A.query(fake_A.detach())
    loss_D_A, _, _ = lsgan_discriminator_loss(d_a(real_A), d_a(fake_A_pooled))
    loss_D_A.backward()
    grad_norm_D_A = compute_grad_norm(d_a.parameters())
    check_finite(loss_D_A.item(), "loss_D_A", 0)
    optimizers["D_A"].step()
    after_da = d_a.model[0].weight.detach().clone()

    before_db = d_b.model[0].weight.detach().clone()
    optimizers["D_B"].zero_grad()
    fake_B_pooled = pool_fake_B.query(fake_B.detach())
    loss_D_B, _, _ = lsgan_discriminator_loss(d_b(real_B), d_b(fake_B_pooled))
    loss_D_B.backward()
    grad_norm_D_B = compute_grad_norm(d_b.parameters())
    check_finite(loss_D_B.item(), "loss_D_B", 0)
    optimizers["D_B"].step()
    after_db = d_b.model[0].weight.detach().clone()

    disc_update_ok = (not torch.equal(before_da, after_da)) and (not torch.equal(before_db, after_db))
    report(12, "both discriminator backward/updates succeed (weights changed)", disc_update_ok,
           f"loss_D_A={loss_D_A.item():.4f}, loss_D_B={loss_D_B.item():.4f}")

    d_has_grad_after_unfreeze = (any(p.grad is not None for p in d_a.parameters())
                                 and any(p.grad is not None for p in d_b.parameters()))
    report(21, "discriminator gradients/updates work after unfreezing",
           d_has_grad_after_unfreeze and disc_update_ok)

    # ---- 13. Gradients finite ----
    grads_finite = math.isfinite(grad_norm_G) and math.isfinite(grad_norm_D_A) and math.isfinite(grad_norm_D_B)
    report(13, "gradients are finite", grads_finite,
           f"grad_norm_G={grad_norm_G:.4f}, grad_norm_D_A={grad_norm_D_A:.4f}, grad_norm_D_B={grad_norm_D_B:.4f}")

    # ---- 14. Replay pools behave correctly ----
    pool_test = ImagePool(pool_size=3, seed=1)
    q1 = pool_test.query(torch.zeros(1, 3, 2, 2))
    q2 = pool_test.query(torch.ones(1, 3, 2, 2))
    q3 = pool_test.query(torch.full((1, 3, 2, 2), 2.0))
    pool_fills_ok = len(pool_test.images) == 3
    q4 = pool_test.query(torch.full((1, 3, 2, 2), 3.0))  # pool now full; either swap or passthrough
    pool_behaves_ok = pool_fills_ok and q4.shape == (1, 3, 2, 2)
    report(14, "replay pools behave correctly (fill to capacity, then swap-or-passthrough)", pool_behaves_ok,
           f"pool size after fill={len(pool_test.images)}")

    # ---- 15. Checkpoint save/reload succeeds ----
    schedulers = build_schedulers(optimizers, training_cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "smoke_test_checkpoint.pt"
        models = {"G_A2B": g_a2b, "G_B2A": g_b2a, "D_A": d_a, "D_B": d_b}
        checksum = save_checkpoint(checkpoint_path, epoch=0, global_step=1, config=config,
                                    models=models, optimizers=optimizers, schedulers=schedulers)
        save_reload_ok = checkpoint_path.exists() and len(checksum) == 64
        report(15, "checkpoint save/reload succeeds", save_reload_ok, f"sha256={checksum[:16]}...")

        # ---- 16. Outputs before/after reload match in eval() mode ----
        g_a2b.eval()
        with torch.no_grad():
            output_before = g_a2b(real_A).clone()

        g_a2b_reloaded = build_generator().to(device)
        g_b2a_reloaded = build_generator().to(device)
        d_a_reloaded = build_discriminator().to(device)
        d_b_reloaded = build_discriminator().to(device)
        load_checkpoint(checkpoint_path, models={
            "G_A2B": g_a2b_reloaded, "G_B2A": g_b2a_reloaded,
            "D_A": d_a_reloaded, "D_B": d_b_reloaded,
        }, map_location=device)
        g_a2b_reloaded.eval()
        with torch.no_grad():
            output_after = g_a2b_reloaded(real_A).clone()

        outputs_match = torch.allclose(output_before, output_after, atol=1e-5)
        max_diff = (output_before - output_after).abs().max().item()
        report(16, "outputs before/after checkpoint reload match in eval() mode", outputs_match,
               f"max abs diff={max_diff:.2e}")
        g_a2b.train()

    # ---- 17. Image de-normalization/save/reload gives a valid RGB 256x256 image ----
    with torch.no_grad():
        sample_output = g_a2b(real_A)[0]
    image = denormalize_to_image(sample_output)
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "sample.jpg"
        image.save(image_path)
        reloaded_image = Image.open(image_path)
        image_ok = reloaded_image.size == (256, 256) and reloaded_image.convert("RGB").mode == "RGB"
        report(17, "image de-normalization/save/reload gives a valid RGB 256x256 image", image_ok,
               f"size={reloaded_image.size}, mode={reloaded_image.mode}")

    # ---- 18. LR scheduler boundary test ----
    schedule_cfg = training_cfg["lr_schedule"]
    constant_epochs = schedule_cfg["constant_epochs"][1]
    decay_epochs = schedule_cfg["decay_epochs"][1] - schedule_cfg["decay_epochs"][0] + 1
    lr_lambda = make_lr_lambda(constant_epochs, decay_epochs)
    base_lr = training_cfg["learning_rate"]

    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_opt = torch.optim.Adam([dummy_param], lr=base_lr)
    dummy_scheduler = torch.optim.lr_scheduler.LambdaLR(dummy_opt, lr_lambda)

    checkpoints_1indexed = [1, 100, 101, 150, 200]
    checkpoints_last_epoch = [e - 1 for e in checkpoints_1indexed]
    observed_lrs = {}
    current_last_epoch = 0
    for target_last_epoch in checkpoints_last_epoch:
        while current_last_epoch < target_last_epoch:
            dummy_scheduler.step()
            current_last_epoch += 1
        observed_lrs[checkpoints_1indexed[checkpoints_last_epoch.index(target_last_epoch)]] = dummy_opt.param_groups[0]["lr"]

    expected_multipliers = {e: lr_lambda(e - 1) for e in checkpoints_1indexed}
    lr_ok = True
    print("    LR schedule boundary values:")
    for e in checkpoints_1indexed:
        expected_lr = base_lr * expected_multipliers[e]
        observed = observed_lrs[e]
        close = abs(expected_lr - observed) < 1e-9
        lr_ok = lr_ok and close
        print(f"      epoch {e:>3}: lr={observed:.6e} (expected {expected_lr:.6e}, multiplier={expected_multipliers[e]:.6f})")
    # Sanity: constant phase equal, decay phase strictly decreasing, final epoch near zero.
    lr_ok = lr_ok and (observed_lrs[1] == observed_lrs[100])
    lr_ok = lr_ok and (observed_lrs[100] > observed_lrs[101] > observed_lrs[150] > observed_lrs[200])
    lr_ok = lr_ok and (observed_lrs[200] < base_lr * 0.02)
    report(18, "LR scheduler boundary test passes (no off-by-one)", lr_ok)

    # ---- 19. NaN/Inf detection works ----
    nan_detected_correctly = False
    try:
        check_finite(float("nan"), "test_value", step=0)
    except RuntimeError:
        try:
            check_finite(float("inf"), "test_value", step=0)
        except RuntimeError:
            nan_detected_correctly = True
    report(19, "NaN/Inf detection works", nan_detected_correctly)

    # ---- 22. Independent Monet resampling verified ----
    resample_test_dataset = UnalignedMonetPhotoDataset(
        monet_dir, photo_dir,
        load_size=training_cfg["augmentation"]["resize"],
        crop_size=config["image_size"],
        seed=seed,
    )
    fixed_photo_index = 0
    monet_names_seen = []
    for _ in range(20):
        item = resample_test_dataset[fixed_photo_index]
        monet_names_seen.append(item["A_name"])
    valid_monet_names = {p.name for p in resample_test_dataset.monet_paths}
    all_valid = all(name in valid_monet_names for name in monet_names_seen)
    distinct_count = len(set(monet_names_seen))
    resampling_ok = all_valid and distinct_count >= 2
    report(22, "independent Monet resampling verified (same photo index maps to varying Monet images)",
           resampling_ok,
           f"{distinct_count} distinct Monet images across 20 repeated draws at photo index {fixed_photo_index} "
           f"(all within valid range: {all_valid})")

    # ---- 23. Every photo traversed exactly once per Policy-A epoch ----
    full_loader = DataLoader(resample_test_dataset, batch_size=1, shuffle=True)
    seen_photo_names = [item["B_name"][0] for item in full_loader]
    expected_names = {p.name for p in resample_test_dataset.photo_paths}
    seen_set = set(seen_photo_names)
    traversal_ok = (
        len(seen_photo_names) == len(resample_test_dataset.photo_paths) == 7038
        and seen_set == expected_names
        and len(seen_photo_names) == len(seen_set)
    )
    report(23, "every photo traversed exactly once per Policy-A epoch (no duplication/omission)",
           traversal_ok,
           f"visited={len(seen_photo_names)}, distinct={len(seen_set)}, expected={len(expected_names)}")

    # ---- 24. NaN event is actually written to JSONL before abort ----
    with tempfile.TemporaryDirectory() as tmp_log_dir:
        nan_test_logger = RawLogger(run_id="smoke_nan_test", log_dir=tmp_log_dir)
        nan_logged_and_raised = False
        try:
            _check_finite_logged(float("nan"), "test_component", epoch=0, global_step=0, logger=nan_test_logger)
        except TrainingDiverged:
            with open(nan_test_logger.path) as f:
                lines = [json.loads(line) for line in f]
            nan_events = [line for line in lines if line.get("event") == "nan_detected"]
            nan_logged_and_raised = (
                len(nan_events) == 1
                and nan_events[0]["component"] == "test_component"
                and nan_events[0]["epoch"] == 0
                and nan_events[0]["global_step"] == 0
                and math.isnan(nan_events[0]["value"])
            )
        report(24, "NaN event is actually written to JSONL before abort", nan_logged_and_raised)

    # ---- 25 & 26. Checkpoint resume restores state and continues LR correctly ----
    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_checkpoint_dir = Path(tmp_root) / "checkpoints"
        tmp_log_dir = Path(tmp_root) / "raw_logs"
        run_id_a = "smoke_resume_test"

        logger_a = RawLogger(run_id=run_id_a, log_dir=tmp_log_dir)
        models_a, optimizers_a, schedulers_a, _ = run_training(
            monet_dir, photo_dir, config, device, logger_a, run_id_a,
            max_epochs=1, max_steps_per_epoch=2,
            checkpoint_dir=tmp_checkpoint_dir, checkpoint_every_epochs=1,
        )
        checkpoint_path = tmp_checkpoint_dir / run_id_a / "epoch_0.pt"

        # Restore into a completely fresh set of models/optimizers/schedulers
        # (not the ones still in memory from the tiny run above) to prove the
        # checkpoint file itself -- not leftover Python state -- carries
        # everything needed.
        fresh_models = {"G_A2B": build_generator().to(device), "G_B2A": build_generator().to(device),
                         "D_A": build_discriminator().to(device), "D_B": build_discriminator().to(device)}
        fresh_optimizers = build_optimizers(fresh_models, training_cfg)
        fresh_schedulers = build_schedulers(fresh_optimizers, training_cfg)
        checkpoint = load_checkpoint(checkpoint_path, models=fresh_models, optimizers=fresh_optimizers,
                                      schedulers=fresh_schedulers, map_location=device, restore_rng_state=True)

        models_match = all(
            torch.equal(a, b)
            for name in models_a
            for a, b in zip(models_a[name].state_dict().values(), fresh_models[name].state_dict().values())
        )
        optimizer_lrs_match = all(
            optimizers_a[name].param_groups[0]["lr"] == fresh_optimizers[name].param_groups[0]["lr"]
            for name in optimizers_a
        )
        scheduler_epochs_match = all(
            schedulers_a[name].last_epoch == fresh_schedulers[name].last_epoch
            for name in schedulers_a
        )
        global_step_restored = checkpoint["global_step"] == 2  # 1 epoch x 2 steps (max_steps_per_epoch)

        resume_state_ok = models_match and optimizer_lrs_match and scheduler_epochs_match and global_step_restored
        report(25, "checkpoint resume restores model/optimizer/scheduler/global-step state", resume_state_ok,
               f"models_match={models_match}, optimizer_lrs_match={optimizer_lrs_match}, "
               f"scheduler_epochs_match={scheduler_epochs_match}, global_step={checkpoint['global_step']}")

        # Continuing an actual resumed run must not silently overwrite the
        # original log, and must record the resume lineage.
        run_id_b = "smoke_resume_test_resumed"
        logger_b = RawLogger(run_id=run_id_b, log_dir=tmp_log_dir)
        expected_checkpoint_sha256 = sha256_file(checkpoint_path)
        run_training(
            monet_dir, photo_dir, config, device, logger_b, run_id_a,
            max_epochs=2, max_steps_per_epoch=2,
            checkpoint_dir=tmp_checkpoint_dir, checkpoint_every_epochs=1,
            resume_from=checkpoint_path,
        )
        with open(logger_b.path) as f:
            resumed_lines = [json.loads(line) for line in f]
        resumed_events = [line for line in resumed_lines if line.get("type") == "resumed"]
        continuation_ok = (
            len(resumed_events) == 1
            and resumed_events[0]["resumed_epoch"] == 0
            and resumed_events[0]["resumed_global_step"] == 2
            and resumed_events[0]["checkpoint_sha256"] == expected_checkpoint_sha256
            and Path(logger_a.path).exists()  # original log untouched/still present
        )

    # Isolated, decisive LR-continuation check: advance deep into the decay
    # phase (past epoch 101) so a reset-vs-continue divergence is obvious,
    # rather than relying on the tiny 1-epoch run above (which never leaves
    # the flat constant phase).
    schedule_cfg = training_cfg["lr_schedule"]
    constant_epochs_lr = schedule_cfg["constant_epochs"][1]
    decay_epochs_lr = schedule_cfg["decay_epochs"][1] - schedule_cfg["decay_epochs"][0] + 1
    lr_lambda_fn = make_lr_lambda(constant_epochs_lr, decay_epochs_lr)

    param_x = torch.nn.Parameter(torch.zeros(1))
    opt_x = torch.optim.Adam([param_x], lr=training_cfg["learning_rate"])
    sched_x = torch.optim.lr_scheduler.LambdaLR(opt_x, lr_lambda_fn)
    for _ in range(120):  # deep into the decay phase (epoch 101+)
        sched_x.step()

    with tempfile.TemporaryDirectory() as tmp_dir:
        sched_ckpt_path = Path(tmp_dir) / "scheduler_only.pt"
        torch.save({"optimizer_state_dict": opt_x.state_dict(),
                    "scheduler_state_dict": sched_x.state_dict()}, sched_ckpt_path)

        param_y = torch.nn.Parameter(torch.zeros(1))
        opt_y = torch.optim.Adam([param_y], lr=training_cfg["learning_rate"])
        sched_y = torch.optim.lr_scheduler.LambdaLR(opt_y, lr_lambda_fn)
        loaded = torch.load(sched_ckpt_path, weights_only=False)
        opt_y.load_state_dict(loaded["optimizer_state_dict"])
        sched_y.load_state_dict(loaded["scheduler_state_dict"])

    sched_x.step()  # ground-truth "never interrupted" continuation
    sched_y.step()  # "resumed" continuation
    lr_continuation_matches = math.isclose(
        opt_x.param_groups[0]["lr"], opt_y.param_groups[0]["lr"], rel_tol=1e-9
    )
    reset_lr = training_cfg["learning_rate"] * lr_lambda_fn(0)
    would_differ_if_reset = not math.isclose(opt_x.param_groups[0]["lr"], reset_lr, rel_tol=1e-6)

    lr_resume_ok = continuation_ok and lr_continuation_matches and would_differ_if_reset
    report(26, "resumed LR continues from expected position (not reset)", lr_resume_ok,
           f"continuation_ok={continuation_ok}, resumed_lr={opt_y.param_groups[0]['lr']:.6e} "
           f"matches_ground_truth={lr_continuation_matches}, would_differ_if_reset={would_differ_if_reset}")

    print(f"\nAll {len(RESULTS)} smoke-test checks passed. This was NOT the 200-epoch training run.")
    print(f"Device used: {device}")
    print(f"Parameter counts: {param_counts}")


if __name__ == "__main__":
    main()

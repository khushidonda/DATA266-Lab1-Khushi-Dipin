"""Task 1 (Step 6): lightweight implementation checks for the training
pipeline in train.py. These verify the pieces are wired correctly — they
do NOT run any real training steps (that's smoke_test.py, Step 7)."""

from model import build_model_from_config
from train import build_dataloaders, build_optimizer, build_scheduler, load_full_config, RawLogger


def main():
    full_cfg = load_full_config()
    training_cfg = full_cfg["training"]

    model, _ = build_model_from_config()
    print("[PASS] Model builds from config for pipeline use.")

    train_loader, val_loader = build_dataloaders(training_cfg)
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    train_shapes = [tuple(t.shape) for t in train_batch]
    val_shapes = [tuple(t.shape) for t in val_batch]
    print(f"[PASS] DataLoaders build: train batch shapes={train_shapes}, val batch shapes={val_shapes}, "
          f"train batches/epoch={len(train_loader)}, val batches={len(val_loader)}.")

    optimizer = build_optimizer(model, training_cfg)
    print(f"[PASS] Optimizer builds from config: {type(optimizer).__name__}, "
          f"lr={optimizer.defaults['lr']}, weight_decay={optimizer.defaults.get('weight_decay')}.")

    total_steps = len(train_loader) * training_cfg["epochs"]
    scheduler = build_scheduler(optimizer, training_cfg, total_steps)
    lr_at_start = scheduler.get_last_lr()[0]
    for _ in range(training_cfg["warmup_steps"]):
        optimizer.step()
        scheduler.step()
    lr_after_warmup = scheduler.get_last_lr()[0]
    print(f"[PASS] Scheduler builds from config ({training_cfg['scheduler']}): "
          f"lr at step 0={lr_at_start:.2e}, lr after {training_cfg['warmup_steps']} warmup steps={lr_after_warmup:.2e}.")
    assert lr_after_warmup > lr_at_start, "Warmup schedule did not increase LR as expected."

    logger = RawLogger()
    logger.log({"type": "check", "message": "first record"})
    logger.log({"type": "check", "message": "second record"})
    with open(logger.path) as f:
        lines = f.readlines()
    assert len(lines) == 2, "RawLogger did not append both records to the same file."
    print(f"[PASS] RawLogger appends without overwriting: {logger.path} has {len(lines)} lines.")

    print("\nAll Step 6 lightweight implementation checks passed.")


if __name__ == "__main__":
    main()

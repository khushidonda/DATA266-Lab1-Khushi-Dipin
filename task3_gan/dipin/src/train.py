"""Train CycleGAN (photo <-> Monet).

Usage (from repo root):
    python task3_gan/dipin/src/train.py                      # full run from config
    python task3_gan/dipin/src/train.py --resume             # continue from checkpoints/latest.pt
    python task3_gan/dipin/src/train.py --smoke              # 2 tiny epochs, outputs under outputs/smoke

Losses (per step):
    D_X : LSGAN  0.5 * [ (D(x)-1)^2 + D(G(y))^2 ]          fakes drawn from a 50-image replay pool
    G   : LSGAN (D_B(G_AB(a))-1)^2 + (D_A(G_BA(b))-1)^2
          + lambda_cyc * ( |G_BA(G_AB(a)) - a| + |G_AB(G_BA(b)) - b| )
          + lambda_id  * ( |G_BA(a) - a| + |G_AB(b) - b| )
"""
import argparse
import itertools
import json
import math
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (DIPIN, GpuAugment, UnpairedSampler, load_split, make_splits,
                  to_model_range)
from models import build_models, count_params

CFG_PATH = DIPIN / "configs" / "task3_config.json"


class ImagePool:
    """Shrivastava et al. (2017) history buffer: with p=0.5 return an older fake."""

    def __init__(self, size):
        self.size, self.images = size, []

    def query(self, imgs):
        if self.size == 0:
            return imgs
        out = []
        for img in imgs.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                k = random.randrange(self.size)
                out.append(self.images[k].clone())
                self.images[k] = img
            else:
                out.append(img)
        return torch.cat(out)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def grad_norm(params):
    norms = [p.grad.detach().float().norm(2) for p in params if p.grad is not None]
    return torch.stack(norms).norm(2).item() if norms else 0.0


def lr_lambda_factory(constant_epochs, decay_epochs):
    def f(epoch):  # epoch index starting at 0; stepped once per epoch
        return 1.0 - max(0, epoch + 1 - constant_epochs) / float(decay_epochs + 1)
    return f


def hardware_info(device):
    info = {"device": str(device), "python": platform.python_version(),
            "torch": torch.__version__, "platform": platform.platform(),
            "cpu": platform.processor()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 2),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return info


@torch.no_grad()
def sample_grid(G_AB, G_BA, fa, fb, path, amp_ctx):
    """Rows: photo | photo->Monet | cycle-rec  and  Monet | Monet->photo | cycle-rec."""
    G_AB.eval(); G_BA.eval()
    with amp_ctx():
        fake_b = G_AB(fa); rec_a = G_BA(fake_b)
        fake_a = G_BA(fb); rec_b = G_AB(fake_a)
    G_AB.train(); G_BA.train()
    grid = torch.cat([fa, fake_b.float(), rec_a.float(), fb, fake_a.float(), rec_b.float()])
    save_image(grid.add(1).div(2).clamp(0, 1), path, nrow=len(fa))
    return ((rec_a.float() - fa).abs().mean().item(), (rec_b.float() - fb).abs().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_PATH))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, help="override total epochs (smoke/benchmark)")
    ap.add_argument("--steps-per-epoch", type=int)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tr = cfg["training"]
    if args.smoke:
        tr.update(constant_epochs=1, decay_epochs=1, steps_per_epoch=20, log_every=5)
    if args.epochs:
        n = args.epochs
        tr["constant_epochs"], tr["decay_epochs"] = math.ceil(n / 2), n // 2
    if args.steps_per_epoch:
        tr["steps_per_epoch"] = args.steps_per_epoch
    total_epochs = tr["constant_epochs"] + tr["decay_epochs"]

    tag = "smoke" if args.smoke else "full"
    out_dir = DIPIN / "outputs" / ("smoke" if args.smoke else "")
    ckpt_dir = DIPIN / "checkpoints" / ("smoke" if args.smoke else "")
    log_dir, sample_dir = out_dir / "logs", out_dir / "samples"
    for d in (ckpt_dir, log_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)

    seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = tr.get("amp_bf16", False) and device.type == "cuda"
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp \
        else (lambda: torch.autocast("cpu", enabled=False))

    # ---- data -------------------------------------------------------------
    splits = make_splits(cfg)
    A_train, _ = load_split(splits, "A", "train", cfg["image_size"])
    B_train, _ = load_split(splits, "B", "train", cfg["image_size"])
    A_test, _ = load_split(splits, "A", "test", cfg["image_size"])
    B_test, _ = load_split(splits, "B", "test", cfg["image_size"])
    print(f"A(photo) train {len(A_train)} test {len(A_test)} | "
          f"B(Monet) train {len(B_train)} test {len(B_test)}", flush=True)
    sampler = UnpairedSampler(A_train, B_train, tr["batch_size"], cfg["seed"])
    aug_cfg = tr["augmentation"]
    augment = GpuAugment(aug_cfg["resize"], aug_cfg["random_crop"], aug_cfg["random_horizontal_flip"])
    fixed_a = to_model_range(A_test[:4]).to(device)
    fixed_b = to_model_range(B_test[:4]).to(device)

    # ---- models / optim ---------------------------------------------------
    G_AB, G_BA, D_A, D_B = [m.to(device) for m in build_models(cfg)]
    params = {"G_AB": count_params(G_AB), "G_BA": count_params(G_BA),
              "D_A": count_params(D_A), "D_B": count_params(D_B)}
    params["total"] = sum(params.values())
    g_params = list(itertools.chain(G_AB.parameters(), G_BA.parameters()))
    d_params = list(itertools.chain(D_A.parameters(), D_B.parameters()))
    betas = (tr["beta1"], tr["beta2"])
    opt_G = torch.optim.Adam(g_params, lr=tr["learning_rate"], betas=betas)
    opt_D = torch.optim.Adam(d_params, lr=tr["learning_rate"], betas=betas)
    lam = lr_lambda_factory(tr["constant_epochs"], tr["decay_epochs"])
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lam)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lam)
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    lam_cyc, lam_id = tr["cycle_consistency_weight"], tr["identity_loss_weight"]
    pool_A, pool_B = ImagePool(tr["replay_buffer_size"]), ImagePool(tr["replay_buffer_size"])

    start_epoch, global_step, nan_total, train_seconds = 0, 0, 0, 0.0
    run_id = f"task3_dipin_{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    latest = ckpt_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        for name, m in [("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B)]:
            m.load_state_dict(ck[name])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        sched_G.load_state_dict(ck["sched_G"]); sched_D.load_state_dict(ck["sched_D"])
        start_epoch, global_step = ck["epoch"], ck["global_step"]
        nan_total, train_seconds, run_id = ck["nan_total"], ck["train_seconds"], ck["run_id"]
        pool_A.images, pool_B.images = ck["pool_A"], ck["pool_B"]
        random.setstate(ck["py_rng"]); torch.set_rng_state(ck["torch_rng"])
        sampler.g.set_state(ck["sampler_rng"])
        print(f"resumed {run_id} at epoch {start_epoch}, step {global_step}", flush=True)
    log_path = log_dir / f"{run_id}.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")

    def log(rec):
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()

    if global_step == 0:
        log({"event": "start", "run_id": run_id, "config": cfg, "params": params,
             "hardware": hardware_info(device), "time": datetime.now().isoformat()})

    keys = ["loss_G", "loss_G_adv", "loss_cyc", "loss_id", "loss_D_A", "loss_D_B",
            "D_A_real", "D_A_fake", "D_B_real", "D_B_fake", "grad_norm_G", "grad_norm_D"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, total_epochs):
        acc = {k: 0.0 for k in keys}; acc_n = 0
        ep_acc = {k: 0.0 for k in keys}; ep_n = 0
        ep_nan = 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ep = t_win = time.time()
        for it in range(tr["steps_per_epoch"]):
            a_u8, b_u8 = sampler.next()
            real_a = augment(to_model_range(a_u8.to(device, non_blocking=True)))
            real_b = augment(to_model_range(b_u8.to(device, non_blocking=True)))

            # ---- generators ----
            for p in d_params:
                p.requires_grad_(False)
            opt_G.zero_grad(set_to_none=True)
            with amp_ctx():
                fake_b = G_AB(real_a); fake_a = G_BA(real_b)
                rec_a = G_BA(fake_b); rec_b = G_AB(fake_a)
                pred_fb = D_B(fake_b).float(); pred_fa = D_A(fake_a).float()
                adv = mse(pred_fb, torch.ones_like(pred_fb)) + mse(pred_fa, torch.ones_like(pred_fa))
                cyc = l1(rec_a.float(), real_a) + l1(rec_b.float(), real_b)
                if lam_id > 0:
                    idt = l1(G_BA(real_a).float(), real_a) + l1(G_AB(real_b).float(), real_b)
                else:
                    idt = torch.zeros((), device=device)
                loss_G = adv + lam_cyc * cyc + lam_id * idt
            loss_G.backward()
            gn_G = grad_norm(g_params)
            ok_G = math.isfinite(loss_G.item()) and math.isfinite(gn_G)
            if ok_G:
                opt_G.step()

            # ---- discriminators ----
            for p in d_params:
                p.requires_grad_(True)
            opt_D.zero_grad(set_to_none=True)
            fa_pool = pool_A.query(fake_a.float()); fb_pool = pool_B.query(fake_b.float())
            with amp_ctx():
                pa_real = D_A(real_a).float(); pa_fake = D_A(fa_pool).float()
                pb_real = D_B(real_b).float(); pb_fake = D_B(fb_pool).float()
                loss_D_A = 0.5 * (mse(pa_real, torch.ones_like(pa_real)) + mse(pa_fake, torch.zeros_like(pa_fake)))
                loss_D_B = 0.5 * (mse(pb_real, torch.ones_like(pb_real)) + mse(pb_fake, torch.zeros_like(pb_fake)))
            (loss_D_A + loss_D_B).backward()
            gn_D = grad_norm(d_params)
            ok_D = math.isfinite(loss_D_A.item() + loss_D_B.item()) and math.isfinite(gn_D)
            if ok_D:
                opt_D.step()
            if not (ok_G and ok_D):  # non-finite step: update skipped, counted
                ep_nan += 1
                continue

            vals = dict(loss_G=loss_G.item(), loss_G_adv=adv.item(), loss_cyc=cyc.item(),
                        loss_id=idt.item(), loss_D_A=loss_D_A.item(), loss_D_B=loss_D_B.item(),
                        D_A_real=pa_real.mean().item(), D_A_fake=pa_fake.mean().item(),
                        D_B_real=pb_real.mean().item(), D_B_fake=pb_fake.mean().item(),
                        grad_norm_G=gn_G, grad_norm_D=gn_D)
            for k, v in vals.items():
                acc[k] += v; ep_acc[k] += v
            acc_n += 1; ep_n += 1
            global_step += 1

            if global_step % tr["log_every"] == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t_win
                rec = {"event": "step", "step": global_step, "epoch": epoch + 1,
                       "lr": opt_G.param_groups[0]["lr"],
                       **{k: v / max(acc_n, 1) for k, v in acc.items()},
                       "images_per_sec": acc_n * tr["batch_size"] / dt,
                       "nan_count_cum": nan_total + ep_nan}
                log(rec)
                acc = {k: 0.0 for k in keys}; acc_n = 0; t_win = time.time()

        if device.type == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t_ep
        train_seconds += ep_time
        nan_total += ep_nan
        sched_G.step(); sched_D.step()
        cyc_a, cyc_b = sample_grid(G_AB, G_BA, fixed_a, fixed_b,
                                   sample_dir / f"epoch_{epoch + 1:03d}.png", amp_ctx)
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        ep_rec = {"event": "epoch", "epoch": epoch + 1, "step": global_step,
                  **{k: v / max(ep_n, 1) for k, v in ep_acc.items()},
                  "epoch_seconds": ep_time, "train_seconds": train_seconds,
                  "images_per_sec": ep_n * tr["batch_size"] / ep_time,
                  "nan_count_epoch": ep_nan, "nan_count_total": nan_total,
                  "fixed_test_cycle_L1_A": cyc_a, "fixed_test_cycle_L1_B": cyc_b,
                  "peak_memory_gb": peak, "lr_next": opt_G.param_groups[0]["lr"]}
        log(ep_rec)
        print(f"[{run_id}] epoch {epoch + 1}/{total_epochs} G {ep_rec['loss_G']:.3f} "
              f"cyc {ep_rec['loss_cyc']:.3f} D_A {ep_rec['loss_D_A']:.3f} D_B {ep_rec['loss_D_B']:.3f} "
              f"| {ep_rec['images_per_sec']:.1f} img/s, {ep_time:.0f}s, nan {nan_total}", flush=True)

        state = {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                 "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                 "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                 "sched_G": sched_G.state_dict(), "sched_D": sched_D.state_dict(),
                 "epoch": epoch + 1, "global_step": global_step, "nan_total": nan_total,
                 "train_seconds": train_seconds, "run_id": run_id,
                 "pool_A": pool_A.images, "pool_B": pool_B.images,
                 "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                 "sampler_rng": sampler.g.get_state()}
        torch.save(state, ckpt_dir / "latest.tmp")
        (ckpt_dir / "latest.tmp").replace(latest)
        if (epoch + 1) % tr["snapshot_every"] == 0 or epoch + 1 == total_epochs:
            torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict()},
                       ckpt_dir / f"generators_epoch{epoch + 1:03d}.pt")

    # ---- final weights (one file per network) -----------------------------
    for name, m in [("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B)]:
        torch.save(m.state_dict(), ckpt_dir / f"{name}.pt")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    summary = {"run_id": run_id, "epochs": total_epochs, "steps": global_step,
               "batch_size": tr["batch_size"], "train_seconds": train_seconds,
               "train_hours": train_seconds / 3600,
               "images_per_sec": global_step * tr["batch_size"] / max(train_seconds, 1e-9),
               "peak_memory_gb_since_last_start": peak, "nan_count_total": nan_total,
               "params": params, "hardware": hardware_info(device), "raw_log": log_path.relative_to(DIPIN).as_posix()}
    log({"event": "end", **summary})
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

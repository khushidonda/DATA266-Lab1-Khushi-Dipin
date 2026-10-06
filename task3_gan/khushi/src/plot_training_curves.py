"""Task 3 (Khushi): epoch-level training curves from the unedited production raw logs (read-only).

Reads the two JSONL segments of run task3_khushi_prod_20260930_154246 and writes, for completed
epochs 1..29 (0-indexed epoch 0..28; epoch_28.pt is the selected checkpoint):
  outputs/plots/training_epoch_metrics.csv
  outputs/plots/generator_discriminator_losses.png
  outputs/plots/cycle_identity_losses.png
  outputs/plots/gradient_norms.png
  outputs/plots/training_summary.json   (totals through the selected epoch + NaN/Inf event count)
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KHUSHI = Path(__file__).resolve().parent.parent
LOGS = Path(__file__).resolve().parents[3] / "reproducibility" / "raw_logs" / "khushi"
SEGMENTS = ["task3_khushi_prod_20260930_154246.jsonl", "task3_khushi_prod_20260930_154246_resumed_20260930_164230.jsonl"]
LAST_EPOCH = 28                      # 0-indexed; completed epoch 29
OUT = KHUSHI / "outputs" / "plots"

records = []
for name in SEGMENTS:
    with open(LOGS / name) as f:
        records += [json.loads(line) for line in f]

ends = {r["epoch"]: r for r in records if r.get("type") == "epoch_end" and r["epoch"] <= LAST_EPOCH}
assert sorted(ends) == list(range(LAST_EPOCH + 1)), "missing or duplicate epochs"
nan_events = [r for r in records if r.get("type") in ("nan_detected", "run_aborted") or r.get("event") in ("nan_detected", "run_aborted")]

FIELDS = [("epoch", None), ("G_total", "mean_loss_G"), ("G_adv_A2B", "mean_loss_G_A2B_adv"), ("G_adv_B2A", "mean_loss_G_B2A_adv"),
          ("cycle_A", "mean_loss_cycle_A"), ("cycle_B", "mean_loss_cycle_B"), ("identity_A", "mean_loss_idt_A"),
          ("identity_B", "mean_loss_idt_B"), ("D_A", "mean_loss_D_A"), ("D_B", "mean_loss_D_B"),
          ("G_grad_norm", "mean_grad_norm_G"), ("D_A_grad_norm", "mean_grad_norm_D_A"), ("D_B_grad_norm", "mean_grad_norm_D_B"),
          ("epoch_seconds", "epoch_seconds"), ("images_per_second", "images_per_sec"),
          ("peak_memory_bytes", "peak_gpu_memory_allocated_bytes")]
OUT.mkdir(parents=True, exist_ok=True)
table = []
for e in range(LAST_EPOCH + 1):
    r = ends[e]
    table.append({k: (e + 1 if src is None else r[src]) for k, src in FIELDS})   # "epoch" = completed-epoch number (1-based)
with open(OUT / "training_epoch_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k, _ in FIELDS]); w.writeheader(); w.writerows(table)

x = [t["epoch"] for t in table]
col = lambda k: [t[k] for t in table]
mark = dict(color="0.55", lw=1, ls="--")


def finish(ax, title, ylabel):
    ax.axvline(29, **mark)
    ax.set_title(title); ax.set_xlabel("completed epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=8)


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(x, col("G_total"), label="G total (adv + 10*cycle + 5*identity)", color="#1f77b4")
finish(ax[0], "Generator loss (epoch mean)", "loss")
ax[1].plot(x, col("D_A"), label="D_A (Monet)", color="#d62728"); ax[1].plot(x, col("D_B"), label="D_B (photo)", color="#2ca02c")
finish(ax[1], "Discriminator loss (epoch mean, LSGAN)", "loss")
fig.tight_layout(); fig.savefig(OUT / "generator_discriminator_losses.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(x, col("cycle_A"), label="cycle A (Monet)", color="#1f77b4"); ax[0].plot(x, col("cycle_B"), label="cycle B (photo)", color="#ff7f0e")
finish(ax[0], "Cycle-consistency loss (L1, unweighted)", "L1 on [-1,1] scale")
ax[1].plot(x, col("identity_A"), label="identity A", color="#1f77b4"); ax[1].plot(x, col("identity_B"), label="identity B", color="#ff7f0e")
finish(ax[1], "Identity loss (L1, unweighted)", "L1 on [-1,1] scale")
fig.tight_layout(); fig.savefig(OUT / "cycle_identity_losses.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, col("G_grad_norm"), label="G (both generators)", color="#1f77b4")
ax.plot(x, col("D_A_grad_norm"), label="D_A", color="#d62728"); ax.plot(x, col("D_B_grad_norm"), label="D_B", color="#2ca02c")
finish(ax, "Gradient norm (epoch mean)", "L2 norm")
fig.tight_layout(); fig.savefig(OUT / "gradient_norms.png", dpi=150); plt.close(fig)

last = table[-1]
summary = {
    "selected_completed_epoch": LAST_EPOCH + 1,
    "global_step_at_selected_epoch": ends[LAST_EPOCH]["global_step"],
    "training_seconds_through_selected_epoch": sum(t["epoch_seconds"] for t in table),
    "selected_epoch_images_per_second": last["images_per_second"],
    "mean_images_per_second_epochs_1_to_29": sum(t["images_per_second"] for t in table) / len(table),
    "peak_memory_bytes_max_over_epochs_1_to_29": max(t["peak_memory_bytes"] for t in table),
    "selected_epoch_means": {k: last[k] for k in ("G_total", "D_A", "D_B", "cycle_A", "cycle_B", "identity_A", "identity_B",
                                                  "G_grad_norm", "D_A_grad_norm", "D_B_grad_norm")},
    "recorded_nan_inf_abort_events": len(nan_events),
    "note": "the logger has no explicit NaN counter; training aborts on a non-finite loss/gradient and writes a nan_detected record, so 0 records = 0 recorded NaN/Inf abort events",
}
with open(OUT / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

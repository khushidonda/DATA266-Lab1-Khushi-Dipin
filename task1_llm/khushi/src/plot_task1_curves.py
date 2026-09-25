"""Task 1 (Phase 8): loss curves from a finished run's raw JSONL log.

    python plot_task1_curves.py --run-id task1_full_20260925_204011
    python plot_task1_curves.py --run-id task1_full_20260925_204011 --dynamics-only

Reads reproducibility/raw_logs/khushi/<run_id>.jsonl (read-only) and writes to
task1_llm/khushi/outputs/<run_id>/curves/ (refuses to overwrite an existing folder;
--dynamics-only re-renders just training_dynamics.png in an existing folder):
  loss_curves.png        training vs validation cross-entropy per epoch, one axis
  loss_curves.csv        the per-epoch numbers behind that plot, plus epoch timing/throughput
  training_dynamics.png  per-step training loss, gradient norm and learning rate (three panels)
  train_steps.csv        the per-step numbers behind that figure

Both curves come straight from the raw log's epoch_end records:
  training CE   = mean of the per-step training losses over the epoch (dropout on,
                  weights still changing during the epoch)
  validation CE = validation loss at the end of the epoch (eval mode)
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from train import RAW_LOG_DIR

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
MOVING_AVERAGE_STEPS = 500  # smoothing window for the per-step panels only

# Light-mode chart tokens from the reference palette. The two series use
# categorical slots 1-2 (validated: lightness, chroma, CVD, normal-vision and
# contrast checks all pass on the #fcfcfb surface).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


def load_log(path, allow_incomplete):
    with open(path) as f:
        lines = f.read().splitlines()
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_incomplete and i == len(lines) - 1:
                break  # last line still being written during a preview
            raise
    by_type = {}
    for r in records:
        by_type.setdefault(r["type"], []).append(r)
    if "run_end" not in by_type and not allow_incomplete:
        raise SystemExit(f"{path} has no run_end record (run not finished); use --allow-incomplete for a preview.")
    return by_type


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def moving_average(y, window):
    c = np.cumsum(np.insert(np.asarray(y, dtype=float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def plot_epochs(epochs, out_png, run_id):
    x = [e["epoch"] + 1 for e in epochs]
    train = [e["train_loss"] for e in epochs]
    val = [e["val_loss"] for e in epochs]

    fig = plt.figure(figsize=(7.2, 4.9), dpi=200)
    ax = fig.add_axes([0.1, 0.17, 0.86, 0.58])
    series = (
        (train, TRAIN_COLOR, f"Training: mean over the epoch, dropout on (final {train[-1]:.4f})"),
        (val, VAL_COLOR, f"Validation: end of the epoch, eval mode (final {val[-1]:.4f})"),
    )
    for y, color, label in series:
        ax.plot(x, y, color=color, lw=1.75, marker="o", ms=6.5, mec=SURFACE, mew=1.5, label=label, zorder=3)
    ax.set_xticks(x)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy (nats per character)")
    ax.grid(axis="y")
    # Legend above the plot area so it can never cover a data point.
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, labelcolor=INK_2)

    fig.text(0.1, 0.935, "Training vs validation cross-entropy", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.1, 0.89, f"Character-level GPT on TinyStories · run {run_id}", color=INK_2)
    fig.text(0.1, 0.03, "Training CE averages all steps of an epoch while the model is still improving, so early on it "
             "sits\nabove the end-of-epoch validation CE. Numbers: loss_curves.csv.", color=MUTED, fontsize=8)
    fig.savefig(out_png)
    plt.close(fig)


def plot_steps(steps, run_start, out_png, run_id):
    s = np.array([r["global_step"] for r in steps])
    loss = np.array([r["train_loss"] for r in steps])
    grad_norm = np.array([r["grad_norm"] for r in steps])
    lr = np.array([r.get("lr_used", r["lr"]) for r in steps])
    window = min(MOVING_AVERAGE_STEPS, max(1, len(s) // 10))
    warmup = run_start["config"]["training"]["warmup_steps"]
    steps_per_epoch = run_start["steps_per_epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8), dpi=200, sharex=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.07, hspace=0.45)
    panels = (
        (axes[0], loss, "Training loss per step (nats per character)", True),
        (axes[1], grad_norm, "Gradient norm per step (global L2, not clipped)", True),
        (axes[2], lr, f"Learning rate per step: {warmup:,}-step linear warmup, then cosine decay to 0", False),
    )
    for ax, y, title, smooth in panels:
        for boundary in range(steps_per_epoch, int(s[-1]) + 1, steps_per_epoch):
            ax.axvline(boundary, color=GRID, lw=0.8, zorder=0)
        if smooth:
            ax.plot(s, y, color=TRAIN_COLOR, lw=0.6, alpha=0.22)
            ax.plot(s[window - 1:], moving_average(y, window), color=TRAIN_COLOR, lw=1.5)
        else:
            ax.plot(s, y, color=TRAIN_COLOR, lw=1.5)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.grid(axis="y")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.0e}"))
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[2].set_xlabel("Optimizer step (vertical hairlines: epoch boundaries)")

    fig.text(0.11, 0.955, "Training dynamics", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.11, 0.925, f"Run {run_id} · faint line: every step; solid line: {window:,}-step moving average",
             color=INK_2)
    fig.savefig(out_png)
    plt.close(fig)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot Task 1 loss curves from a raw JSONL log.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log", type=Path, help="default: reproducibility/raw_logs/khushi/<run_id>.jsonl")
    parser.add_argument("--out-dir", type=Path, help="default: task1_llm/khushi/outputs/<run_id>/curves")
    parser.add_argument("--allow-incomplete", action="store_true", help="preview a run that has not finished")
    parser.add_argument("--dynamics-only", action="store_true",
                        help="re-render only training_dynamics.png in an existing curves folder")
    args = parser.parse_args()

    log_path = args.log or RAW_LOG_DIR / f"{args.run_id}.jsonl"
    out_dir = args.out_dir or OUTPUTS_DIR / args.run_id / "curves"
    if args.dynamics_only:
        if not out_dir.is_dir():
            raise SystemExit(f"{out_dir} does not exist; run without --dynamics-only first.")
        by_type = load_log(log_path, args.allow_incomplete)
        apply_style()
        plot_steps(by_type["train_step"], by_type["run_start"][0], out_dir / "training_dynamics.png", args.run_id)
        print(f"Re-rendered {out_dir / 'training_dynamics.png'} (CSVs and loss_curves.png untouched)")
        return
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; refusing to overwrite it.")
    by_type = load_log(log_path, args.allow_incomplete)
    run_start = by_type["run_start"][0]
    epochs = sorted(by_type.get("epoch_end", []), key=lambda e: e["epoch"])
    steps = by_type["train_step"]

    out_dir.mkdir(parents=True)
    apply_style()
    epoch_rows = [{**e, "epoch_index": e["epoch"], "epoch": e["epoch"] + 1,
                   "train_ce": e["train_loss"], "val_ce": e["val_loss"]} for e in epochs]
    write_csv(out_dir / "loss_curves.csv", epoch_rows,
              ["epoch", "epoch_index", "train_ce", "val_ce", "train_seconds", "val_seconds", "epoch_seconds",
               "train_tokens_per_sec", "val_tokens_per_sec", "grad_norm_mean", "grad_norm_max", "lr",
               "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes", "nan_detected"])
    step_rows = [{**r, "lr_after_step": r["lr"]} for r in steps]
    write_csv(out_dir / "train_steps.csv", step_rows,
              ["global_step", "epoch", "batch_idx", "train_loss", "grad_norm", "lr_used", "lr_after_step", "time"])
    if epochs:
        plot_epochs(epochs, out_dir / "loss_curves.png", args.run_id)
    plot_steps(steps, run_start, out_dir / "training_dynamics.png", args.run_id)
    print(f"Wrote {len(epoch_rows)} epoch rows and {len(step_rows):,} step rows, plus plots, to {out_dir}")


if __name__ == "__main__":
    main()

"""Task 1 (Dipin): evaluation, text generation, plots and failure-case candidates.

    python task1_llm/dipin/src/evaluate.py                 # evaluates checkpoints/model.pt

Metric definitions follow the team protocol (same as Khushi's evaluate_task1.py):
  * train/validation cross-entropy: eval mode (dropout off), token-weighted, every sequence
  * perplexity = exp(CE), bits-per-character = CE / ln 2, gap = val CE - train CE
  * Distinct-n = unique n-grams / all n-grams pooled over all continuations (prompt excluded)
  * repeated 4-gram rate = per continuation 1 - unique 4-grams / all 4-grams, averaged
  * generation: 5 prompts x 10 samples x 200 new characters, batch size 1
Character-level diversity is primary (the model is character-level); word-level is secondary.
"""

import argparse
import csv
import json
import math
import re
import time

import matplotlib.pyplot as plt
import torch

from data import DATA_DIR, REPO_DIR, TASK_DIR, load_prepared
from train import CKPT_DIR, load_checkpoint, score_stream
from utils import get_device, sync

OUT_DIR = TASK_DIR / "outputs"
WORD_RE = re.compile(r"[a-z0-9']+")


def words(text):
    return WORD_RE.findall(text.lower())


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct(seqs, n):
    grams = [g for s in seqs for g in ngrams(s, n)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def repeat_rate(seq, n=4):
    grams = ngrams(seq, n)
    return 1 - len(set(grams)) / len(grams) if grams else float("nan")


def mean_finite(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def generate_text(model, tok, prompt, max_new_tokens, temperature, device, greedy=False, fixed_length=True):
    """fixed_length=True never emits <eos>/<unk>, so every sample has exactly max_new_tokens characters
    (needed for comparable diversity metrics). fixed_length=False lets the story end at <eos>."""
    missing = sorted(set(prompt) - set(tok.char_to_idx))
    if missing:
        raise ValueError(f"prompt {prompt!r} has characters outside the vocabulary: {missing}")
    idx = torch.tensor([tok.encode(prompt)], device=device)
    banned = [tok.eos_id, tok.unk_id] if fixed_length else [tok.unk_id]
    out = model.generate(idx, max_new_tokens, temperature, greedy=greedy, banned_ids=banned,
                         stop_id=None if fixed_length else tok.eos_id)[0].tolist()
    if not fixed_length and out[-1] == tok.eos_id:
        out = out[:-1]
    return tok.decode(out)


def generation_suite(model, tok, cfg, device, train_vocab):
    g = cfg["generation"]
    generate_text(model, tok, g["prompts"][0], 20, g["temperature"], device)  # warm-up, not timed
    torch.manual_seed(cfg["seed"])
    samples, total_tokens, total_seconds = [], 0, 0.0
    for pi, prompt in enumerate(g["prompts"]):
        for si in range(g["samples_per_prompt"]):
            sync(device)
            t0 = time.perf_counter()
            text = generate_text(model, tok, prompt, g["max_new_tokens"], g["temperature"], device)
            sync(device)
            dt = time.perf_counter() - t0
            cont = text[len(prompt):]
            w = words(cont)
            samples.append({"id": len(samples), "prompt": prompt, "sample_index": si, "text": text,
                            "continuation": cont, "seconds": dt,
                            "repeated_4gram_rate_char": repeat_rate(list(cont)),
                            "repeated_4gram_rate_word": repeat_rate(w),
                            "unseen_word_rate": sum(1 for x in w if x not in train_vocab) / max(1, len(w)),
                            "unseen_words": sorted({x for x in w if x not in train_vocab})})
            total_tokens += len(cont)
            total_seconds += dt
    chars = [list(s["continuation"]) for s in samples]
    wrds = [words(s["continuation"]) for s in samples]
    greedy = [{"prompt": p, "text": generate_text(model, tok, p, g["max_new_tokens"], 1.0, device, greedy=True)}
              for p in g["prompts"]]
    return {
        "settings": {**g, "seed": cfg["seed"], "batch_size": 1},
        "generation_tokens_per_sec": total_tokens / total_seconds,
        "char_level": {**{f"distinct_{n}": distinct(chars, n) for n in (1, 2, 3)},
                       "repeated_4gram_rate": mean_finite([s["repeated_4gram_rate_char"] for s in samples])},
        "word_level": {**{f"distinct_{n}": distinct(wrds, n) for n in (1, 2, 3)},
                       "repeated_4gram_rate": mean_finite([s["repeated_4gram_rate_word"] for s in samples])},
        "samples": samples, "greedy_samples": greedy,
    }


def failure_candidates_markdown(gen, k=4):
    """Shortlist of samples that look most suspicious under simple heuristics. The failure type
    and the observation are judged by hand in failure_analysis.md."""
    s = gen["samples"]
    picks = [("Highest word-level repetition", sorted(s, key=lambda r: -(r["repeated_4gram_rate_word"]
                                                                          if not math.isnan(r["repeated_4gram_rate_word"]) else 0))[:k]),
             ("Highest character-level repetition", sorted(s, key=lambda r: -r["repeated_4gram_rate_char"])[:k]),
             ("Most words never seen in training (possible broken spelling / invented words)",
              sorted(s, key=lambda r: -r["unseen_word_rate"])[:k])]
    lines = ["# Failure-case candidates (auto-selected, to be reviewed by hand)", ""]
    for title, rows in picks:
        lines += [f"## {title}", ""]
        for r in rows:
            lines += [f"**sample {r['id']}** | prompt {r['prompt']!r} | char-rep {r['repeated_4gram_rate_char']:.3f} | "
                      f"word-rep {r['repeated_4gram_rate_word']:.3f} | unseen words: {', '.join(r['unseen_words']) or 'none'}",
                      "", "```", r["text"], "```", ""]
    lines += ["## Greedy decoding (one per prompt)", ""]
    for r in gen["greedy_samples"]:
        lines += [f"prompt {r['prompt']!r}", "", "```", r["text"], "```", ""]
    return "\n".join(lines)


def plot_curves(summary, log_path, out_dir):
    hist = summary["history"]
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, [h["train_loss"] for h in hist], "-o", label="train (running mean, dropout on)")
    ax.plot(ep, [h["val_loss"] for h in hist], "-o", label="validation (eval mode)")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (nats/char)"); ax.set_title("Training and validation loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=130); plt.close(fig)

    steps = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    steps = [r for r in steps if r["event"] == "step"]
    if steps:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        x = [r["step"] for r in steps]
        for a, key, title in zip(axes, ["loss", "grad_norm", "lr"],
                                 ["Step training loss", "Gradient norm (before clipping)", "Learning rate"]):
            a.plot(x, [r[key] for r in steps], lw=0.8)
            a.set_title(title); a.set_xlabel("optimizer step"); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "training_stability.png", dpi=130); plt.close(fig)


def evaluate(ckpt_path=None, smoke=False, data=None):
    device = get_device()
    ckpt_path = ckpt_path or (CKPT_DIR / ("smoke/model.pt" if smoke else "model.pt"))
    out_dir = OUT_DIR / "smoke" if smoke else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok, saved = load_checkpoint(ckpt_path, device)
    cfg = saved["config"]
    # Run-level stats live in the final checkpoint; best.pt carries weights only.
    summary = saved.get("train_summary") or torch.load(ckpt_path.parent / "model.pt", map_location="cpu",
                                                         weights_only=False)["train_summary"]
    data = data or load_prepared(out_dir / "data" if smoke else DATA_DIR)
    T, bs, amp = cfg["data"]["sequence_length"], cfg["training"]["batch_size"] * 2, cfg["training"]["amp_bf16"]

    train_score = score_stream(model, torch.from_numpy(data["train"]).to(device), T, bs, device, amp)
    val_score = score_stream(model, torch.from_numpy(data["validation"]).to(device), T, bs, device, amp)
    train_vocab = set(words(tok.decode(data["train"][:20_000_000]).replace("<eos>", " ")))
    gen = generation_suite(model, tok, cfg, device, train_vocab)
    c, w = gen["char_level"], gen["word_level"]
    gpu = summary["hardware"].get("gpu")
    mib = lambda b: "n/a" if b is None else f"{b / 2**20:.1f} MiB"
    ck = f"checkpoint {ckpt_path.relative_to(REPO_DIR).as_posix()} (epoch {saved['epoch']})"

    rows = [
        ("training_cross_entropy_loss", train_score["ce_nats"], f"eval mode (dropout off), full train split ({train_score['tokens']:,} chars), {ck}"),
        ("validation_cross_entropy_loss", val_score["ce_nats"], f"eval mode, full validation split ({val_score['tokens']:,} chars)"),
        ("perplexity", val_score["perplexity"], f"validation; training perplexity = {train_score['perplexity']:.4f}"),
        ("bits_per_character", val_score["bpc"], f"validation; training bpc = {train_score['bpc']:.4f}"),
        ("generalization_gap", val_score["ce_nats"] - train_score["ce_nats"], "validation CE - training CE (nats/char), both in eval mode"),
        ("top1_next_character_accuracy", val_score["top1_accuracy"], f"validation; training = {train_score['top1_accuracy']:.4f}"),
        ("distinct_1", c["distinct_1"], f"character-level (primary); word-level = {w['distinct_1']:.4f}"),
        ("distinct_2", c["distinct_2"], f"character-level (primary); word-level = {w['distinct_2']:.4f}"),
        ("distinct_3", c["distinct_3"], f"character-level (primary); word-level = {w['distinct_3']:.4f}"),
        ("repeated_4gram_rate", c["repeated_4gram_rate"], f"character-level (primary); word-level = {w['repeated_4gram_rate']:.4f}"),
        ("gradient_norm", summary["grad_norm_mean"], f"mean pre-clip norm over {summary['total_steps']} steps; max {summary['grad_norm_max']:.4f} at step {summary['grad_norm_max_step']}; clipped at {summary['grad_clip_max_norm']}"),
        ("nan_count", summary["nan_count"], f"non-finite loss/gradient events; loss spikes (>1.25x running average) = {summary['loss_spikes']}"),
        ("parameter_count", summary["parameter_count"], "unique trainable parameters (input/output embedding tied, counted once)"),
        ("training_tokens_per_sec", summary["training_tokens_per_sec"], "characters/sec over all epochs, training steps only (validation excluded)"),
        ("generation_tokens_per_sec", gen["generation_tokens_per_sec"], f"batch size 1, {len(gen['samples'])} samples x {cfg['generation']['max_new_tokens']} new chars, no KV cache"),
        ("peak_memory_usage", summary["peak_gpu_allocated_bytes"] if summary["peak_gpu_allocated_bytes"] is not None else summary["peak_host_rss_bytes"],
         f"bytes; GPU allocated {mib(summary['peak_gpu_allocated_bytes'])}, GPU reserved {mib(summary['peak_gpu_reserved_bytes'])}, host RSS {mib(summary['peak_host_rss_bytes'])}"),
        ("total_training_time", summary["total_training_seconds"], f"seconds, {summary['epochs']} epochs, {gpu} / {summary['hardware']['cpu']}"),
    ]
    report = out_dir / "metrics_report.csv" if smoke else TASK_DIR / "metrics_report.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", "value", "notes"])
        wr.writerows(rows)

    plot_curves(summary, REPO_DIR / summary["raw_log"], out_dir)
    (out_dir / "failure_candidates.md").write_text(failure_candidates_markdown(gen), encoding="utf-8")
    with open(out_dir / "generated_samples.txt", "w", encoding="utf-8") as f:
        for s in gen["samples"]:
            f.write(f"=== sample {s['id']} | prompt {s['prompt']!r} #{s['sample_index']} | T={cfg['generation']['temperature']} ===\n{s['text']}\n\n")
        for s in gen["greedy_samples"]:
            f.write(f"=== greedy | prompt {s['prompt']!r} ===\n{s['text']}\n\n")
    result = {"run_id": summary["run_id"], "checkpoint": ckpt_path.relative_to(REPO_DIR).as_posix(),
              "train": train_score, "validation": val_score, "generation": gen, "train_summary": summary}
    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {report.relative_to(REPO_DIR).as_posix()} and {out_dir.relative_to(REPO_DIR).as_posix()}/")
    return result, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="default: checkpoints/model.pt (final weights)")
    args = ap.parse_args()
    from pathlib import Path
    evaluate(Path(args.checkpoint).resolve() if args.checkpoint else None)

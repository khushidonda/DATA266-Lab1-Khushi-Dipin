"""Task 1 (Phases 6-7): metrics, generations and failure-case candidates for a finished run.

    python evaluate_task1.py --run-id task1_full_20260925_204011
    python evaluate_task1.py --run-id task1_full_20260925_204011 --rebuild-candidates

Primary checkpoint: the epoch with the lowest validation cross-entropy in the raw
log (override with --checkpoint final|<epoch>). The final checkpoint's identity and
validation result are always reported as well. --rebuild-candidates only rewrites
failure_candidates.md from an existing evaluation's generations.jsonl (no model and
no data; metrics.json and the generations are not touched).

Read-only on the run's raw log, checkpoints, config and data; writes only to
task1_llm/khushi/outputs/<run_id>/eval_epoch_<N>/ and refuses to overwrite it:
  metrics.json           every metric below, with its definition and provenance
  generations.jsonl      every generated sample with its per-sample statistics
  generations.txt        the same samples, for reading
  failure_candidates.md  samples ranked by objective per-sample statistics for manual
                         review; no failure types are assigned (that analysis is Khushi's)

Definitions
  cross-entropy (CE)    mean next-character negative log-likelihood in nats, weighted per
                        token, model in eval mode (no dropout), over ALL train or ALL
                        validation sequences
  perplexity            exp(CE)
  bits per character    CE / ln 2 (one token is one character)
  generalization gap    generalization_gap = validation_eval_ce - training_eval_ce: both in
                        eval mode on the full split, both from the primary checkpoint (nats)
  top-1 accuracy        share of positions whose highest-scoring character is the true next one
  Distinct-n            unique n-grams / all n-grams, pooled over every generated continuation
                        (prompt excluded). PRIMARY: character n-grams, n = 1, 2, 3.
                        SECONDARY diagnostic: word n-grams.
  repeated 4-gram rate  per continuation, 1 - unique 4-grams / all 4-grams (the share of
                        4-grams repeating an earlier one), averaged over continuations.
                        PRIMARY: character 4-grams. SECONDARY diagnostic: word 4-grams.
  generation tokens/s   new characters / wall-clock seconds at batch size 1, train.generate()
  run-level values      training/validation tokens/s, gradient norms, NaN count, peak GPU
                        memory, total time and the logged final-epoch training CE are read
                        from the raw log, not re-measured
"""

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model_from_config
from train import DATA_DIR, RAW_LOG_DIR, generate, load_checkpoint, load_full_config, sha256_file

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
SPLIT_PATH = DATA_DIR / "split_indices.json"

# Fixed generation protocol (evaluation settings, not model hyperparameters).
DEFAULT_PROMPTS = ["Once upon a time", "One day,", "Lily and Ben", "The little bird", "Tom was sad because"]
DEFAULT_SAMPLES_PER_PROMPT = 10
WORD_RE = re.compile(r"[a-z0-9']+")  # secondary word-level diagnostics; applied to lowercased text
TOP_K_CANDIDATES = 3


def load_log(path):
    by_type = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            by_type.setdefault(record["type"], []).append(record)
    if "run_end" not in by_type:
        raise SystemExit(f"{path} has no run_end record: the run has not finished.")
    return by_type


def resolve_epoch(choice, by_type):
    if choice == "best":
        return by_type["run_end"][0]["best_epoch_by_val_loss"]
    if choice == "final":
        return by_type["run_start"][0]["epochs"] - 1
    return int(choice)


def load_verified(saved, device):
    """Builds the model and loads a checkpoint after checking its SHA-256 against the raw log."""
    path = Path(saved["path"])
    sha = sha256_file(path)
    if sha != saved["sha256"]:
        raise SystemExit(f"{path} does not match the SHA-256 recorded in the raw log.")
    model, _ = build_model_from_config()
    model.to(device)
    checkpoint = load_checkpoint(path, model, map_location=device)
    return model, checkpoint, path, sha


@torch.no_grad()
def score(model, inputs, targets, batch_size, device):
    """Token-weighted CE (nats) and top-1 accuracy over every sequence, eval mode."""
    model.eval()
    nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.int64, device=device)
    start = time.perf_counter()
    for i in range(0, inputs.shape[0], batch_size):
        x = inputs[i:i + batch_size].to(device)
        y = targets[i:i + batch_size].to(device)
        logits, _ = model(x)
        nll_sum += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum").double()
        correct += (logits.argmax(dim=-1) == y).sum()
    n = targets.numel()
    ce = nll_sum.item() / n
    return {"ce_nats": ce, "perplexity": math.exp(ce), "bpc": ce / math.log(2),
            "top1_accuracy": correct.item() / n, "tokens": n, "seconds": time.perf_counter() - start}


def load_split(name):
    return (torch.from_numpy(np.load(DATA_DIR / f"{name}_inputs.npy")),
            torch.from_numpy(np.load(DATA_DIR / f"{name}_targets.npy")))


def words(text):
    return WORD_RE.findall(text.lower())


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct(seqs, n):
    grams = [g for seq in seqs for g in ngrams(seq, n)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def repeat_rate(seq, n=4):
    grams = ngrams(seq, n)
    return 1 - len(set(grams)) / len(grams) if grams else float("nan")


def mean_finite(values):
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def training_words():
    """Every word appearing in Khushi's training split (for the unseen-word diagnostic)."""
    from datasets import load_dataset

    with open(SPLIT_PATH) as f:
        train_indices = json.load(f)["train_indices"]
    ds = load_dataset("roneneldan/TinyStories", split="train")
    vocab = set()
    for i in train_indices:
        vocab.update(words(ds[i]["text"]))
    return vocab


def fence(text):
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def candidates_markdown(samples, run_id, epoch):
    max_new = max(s["new_tokens"] for s in samples)
    unseen_caveat = (f"Caveat: every sample stops at exactly {max_new} new characters, so its last word can be cut "
                     "off mid-word, and such fragments count as unseen words. This inflates this secondary "
                     "diagnostic; do not use it as a primary failure-selection criterion.")
    rankings = (
        ("Highest character-level repeated 4-gram rate (primary diversity metric)", "repeated_4gram_rate_char", None),
        ("Highest word-level repeated 4-gram rate (secondary diagnostic)", "repeated_4gram_rate_word", None),
        ("Highest share of words never seen in the training split (secondary diagnostic)", "unseen_word_share",
         unseen_caveat),
        ("Most characters outside printable ASCII", "non_ascii_chars", None),
    )
    lines = [f"# Task 1 failure-case candidates: {run_id}, epoch {epoch} checkpoint", "",
             "Samples ranked by objective per-sample statistics only. No failure types are assigned here: "
             "read the samples (all of them are in generations.txt), then choose and categorize three cases "
             "yourself.", ""]
    for title, key, note in rankings:
        lines += [f"## {title}", ""]
        if note:
            lines += [note, ""]
        valid = [s for s in samples if not math.isnan(s["stats"][key])]
        if len({s["stats"][key] for s in valid}) <= 1:
            # Every sample scores the same, so any "top 3" would just be arbitrary ties.
            if key == "non_ascii_chars" and all(s["stats"][key] == 0 for s in valid):
                lines += ["All generated samples contained only printable ASCII characters; no candidates "
                          "identified for this criterion.", ""]
            else:
                lines += [f"All {len(valid)} samples have the same value for this criterion, so no meaningful "
                          "ranking exists; no candidates listed.", ""]
            continue
        ranked = sorted(valid, key=lambda s: s["stats"][key], reverse=True)[:TOP_K_CANDIDATES]
        for s in ranked:
            st = s["stats"]
            lines.append(f"**Sample {s['id']}** (prompt {s['prompt']!r}): `{key}` = {st[key]:.3f}; "
                         f"char repeated 4-gram rate {st['repeated_4gram_rate_char']:.3f}, "
                         f"word repeated 4-gram rate {st['repeated_4gram_rate_word']:.3f}, "
                         f"unseen-word share {st['unseen_word_share']:.3f}, non-ASCII chars {st['non_ascii_chars']}")
            if st["unseen_words"]:
                lines.append(f"Unseen words: {', '.join(st['unseen_words'])}")
            f = fence(s["text"])
            lines += ["", f + "text", s["text"], f, ""]
    return "\n".join(lines) + "\n"


def rebuild_candidates(out_dir, run_id, epoch):
    """Rewrites failure_candidates.md from a finished evaluation's saved samples and statistics."""
    path = out_dir / "generations.jsonl"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run the full evaluation first.")
    with open(path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    with open(out_dir / "failure_candidates.md", "w", encoding="utf-8") as f:
        f.write(candidates_markdown(samples, run_id, epoch))
    print(f"Rebuilt {out_dir / 'failure_candidates.md'} from {len(samples)} saved samples "
          "(metrics.json and generations untouched)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a finished Task 1 run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best",
                        help="primary checkpoint: best (lowest validation CE, default), final, or an epoch index")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--samples-per-prompt", type=int, default=DEFAULT_SAMPLES_PER_PROMPT)
    parser.add_argument("--gen-seed", type=int, help="sampling seed (default: the config seed)")
    parser.add_argument("--rebuild-candidates", action="store_true",
                        help="only rewrite failure_candidates.md from this evaluation's saved generations.jsonl")
    args = parser.parse_args()

    log_path = RAW_LOG_DIR / f"{args.run_id}.jsonl"
    by_type = load_log(log_path)
    epoch = resolve_epoch(args.checkpoint, by_type)
    out_dir = OUTPUTS_DIR / args.run_id / f"eval_epoch_{epoch}"
    if args.rebuild_candidates:
        rebuild_candidates(out_dir, args.run_id, epoch)
        return
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; refusing to overwrite it.")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    device = torch.device("cuda")
    cfg = load_full_config()
    gen_cfg = cfg["generation"]
    sequence_length = cfg["data"]["sequence_length"]
    batch_size = cfg["training"]["batch_size"]
    gen_seed = cfg["seed"] if args.gen_seed is None else args.gen_seed

    run_start, run_end = by_type["run_start"][0], by_type["run_end"][0]
    epoch_end = {r["epoch"]: r for r in by_type["epoch_end"]}
    saved = {r["epoch"]: r for r in by_type["checkpoint_saved"]}
    final_epoch = run_start["epochs"] - 1
    for e in (epoch, final_epoch):
        if e not in saved:
            raise SystemExit(f"No checkpoint_saved record for epoch {e} in the raw log.")

    model, checkpoint, checkpoint_path, checkpoint_sha = load_verified(saved[epoch], device)
    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        vocab = json.load(f)["vocab"]
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    print(f"Primary checkpoint: {args.run_id} epoch {epoch} ({checkpoint_path.name}, sha256 {checkpoint_sha[:12]}...)")

    # Phase 6: CE, perplexity, BPC and top-1 accuracy on the full splits (primary checkpoint).
    val_inputs, val_targets = load_split("validation")
    validation = score(model, val_inputs, val_targets, batch_size, device)
    validation["logged_val_loss_same_epoch"] = epoch_end[epoch]["val_loss"]
    validation["matches_training_log"] = abs(validation["ce_nats"] - epoch_end[epoch]["val_loss"]) < 1e-4
    train_inputs, train_targets = load_split("train")
    train = score(model, train_inputs, train_targets, batch_size, device)
    del train_inputs, train_targets
    for name, m in (("validation", validation), ("train", train)):
        print(f"  {name} (eval mode): CE {m['ce_nats']:.4f} nats, PPL {m['perplexity']:.3f}, "
              f"BPC {m['bpc']:.4f}, top-1 {m['top1_accuracy']:.4f}")

    # Final checkpoint identity and validation result (reported for completeness).
    if final_epoch == epoch:
        final_block = {"epoch": final_epoch, "path": str(checkpoint_path), "sha256": checkpoint_sha,
                       "same_as_primary": True, "logged_val_loss": epoch_end[final_epoch]["val_loss"],
                       "validation": "same as primary checkpoint"}
    else:
        final_model, _, final_path, final_sha = load_verified(saved[final_epoch], device)
        final_validation = score(final_model, val_inputs, val_targets, batch_size, device)
        del final_model
        final_block = {"epoch": final_epoch, "path": str(final_path), "sha256": final_sha,
                       "same_as_primary": False, "logged_val_loss": epoch_end[final_epoch]["val_loss"],
                       "validation": final_validation}
        print(f"  final checkpoint (epoch {final_epoch}) validation CE {final_validation['ce_nats']:.4f}")

    # Phase 7: generation with the fixed protocol and the configured sampling settings.
    for prompt in args.prompts:
        missing = sorted(set(prompt) - set(char_to_idx))
        if missing:
            raise SystemExit(f"Prompt {prompt!r} has characters outside the vocabulary: {missing}")
    generate(model, [char_to_idx[c] for c in args.prompts[0]], 20, gen_cfg["temperature"], sequence_length, device)
    torch.manual_seed(gen_seed)  # after the warm-up call, so the saved samples depend only on this seed
    samples = []
    for prompt_index, prompt in enumerate(args.prompts):
        prompt_idx = [char_to_idx[c] for c in prompt]
        for sample_index in range(args.samples_per_prompt):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = generate(model, prompt_idx, gen_cfg["max_new_tokens"], gen_cfg["temperature"],
                           sequence_length, device)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            text = "".join(vocab[i] for i in out)
            samples.append({"id": len(samples), "prompt_index": prompt_index, "sample_index": sample_index,
                            "prompt": prompt, "text": text, "continuation": text[len(prompt):],
                            "new_tokens": len(out) - len(prompt_idx), "seconds": seconds})

    train_vocab = training_words()
    for s in samples:
        w = words(s["continuation"])
        unseen = [x for x in w if x not in train_vocab]
        s["stats"] = {
            "repeated_4gram_rate_char": repeat_rate(list(s["continuation"])),
            "repeated_4gram_rate_word": repeat_rate(w),
            "words": len(w),
            "unseen_word_share": len(unseen) / len(w) if w else float("nan"),
            "unseen_words": sorted(set(unseen)),
            "non_ascii_chars": sum(1 for ch in s["continuation"] if ord(ch) > 126),
        }
    char_seqs = [list(s["continuation"]) for s in samples]
    word_seqs = [words(s["continuation"]) for s in samples]
    new_tokens = sum(s["new_tokens"] for s in samples)
    gen_seconds = sum(s["seconds"] for s in samples)
    total_words = sum(len(w) for w in word_seqs)
    generation = {
        "settings": {"method": gen_cfg["method"], "temperature": gen_cfg["temperature"],
                     "max_new_tokens": gen_cfg["max_new_tokens"], "prompts": args.prompts,
                     "samples_per_prompt": args.samples_per_prompt, "seed": gen_seed, "batch_size": 1},
        "num_samples": len(samples),
        "new_tokens_total": new_tokens,
        "seconds_total": gen_seconds,
        "tokens_per_sec": new_tokens / gen_seconds,
        "primary_character_level": {
            **{f"distinct_{n}": distinct(char_seqs, n) for n in (1, 2, 3)},
            "repeated_4gram_rate": mean_finite([s["stats"]["repeated_4gram_rate_char"] for s in samples]),
        },
        "secondary_word_level": {
            **{f"distinct_{n}": distinct(word_seqs, n) for n in (1, 2, 3)},
            "repeated_4gram_rate": mean_finite([s["stats"]["repeated_4gram_rate_word"] for s in samples]),
            "words_total": total_words,
            "unseen_word_share": sum(len([x for x in w if x not in train_vocab]) for w in word_seqs) / total_words
            if total_words else float("nan"),
        },
    }

    # Run-level values straight from the raw log.
    steps = by_type["train_step"]
    epochs = [epoch_end[e] for e in sorted(epoch_end)]
    grad_norms = np.array([r["grad_norm"] for r in steps])
    step_losses = np.array([r["train_loss"] for r in steps])
    training_run = {
        "parameter_count": run_start["num_parameters"],
        "total_steps": run_end["global_step"],
        "training_tokens_per_sec": sum(e["train_tokens"] for e in epochs) / sum(e["train_seconds"] for e in epochs),
        "validation_tokens_per_sec": sum(e["val_tokens"] for e in epochs) / sum(e["val_seconds"] for e in epochs),
        "tokens_per_sec_note": "measured by the training loop; includes the per-step gradient-norm computation "
                               "and per-step raw-log writes",
        "grad_norm": {"min": float(grad_norms.min()), "p50": float(np.percentile(grad_norms, 50)),
                      "mean": float(grad_norms.mean()), "p99": float(np.percentile(grad_norms, 99)),
                      "max": float(grad_norms.max()), "max_step": int(steps[int(grad_norms.argmax())]["global_step"]),
                      "final_epoch_mean": epochs[-1]["grad_norm_mean"], "clipping": "none (reported only)"},
        "nan_count": len(by_type.get("nan_detected", [])) + int((~np.isfinite(step_losses)).sum())
        + int((~np.isfinite(grad_norms)).sum()) + sum(1 for e in epochs if not math.isfinite(e["val_loss"])),
        "peak_gpu_memory_allocated_bytes": run_end["peak_gpu_memory_allocated_bytes"],
        "peak_gpu_memory_reserved_bytes": max(e["peak_gpu_memory_reserved_bytes"] for e in epochs),
        "total_training_time_seconds": run_end["total_seconds"],
        "best_epoch_by_val_loss": run_end["best_epoch_by_val_loss"],
        "final_epoch_logged_train_ce": epochs[-1]["train_loss"],
        "final_epoch_logged_train_ce_note": "training-dynamics statistic: mean of the per-step training losses "
                                            "over the last epoch, dropout on (not used for the gap)",
        "per_epoch_logged": [{"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val_loss"]}
                             for e in epochs],
    }

    metrics = {
        "run_id": args.run_id,
        "primary_checkpoint": {
            "selection": "lowest validation CE in the raw log" if args.checkpoint == "best"
            else f"--checkpoint {args.checkpoint}",
            "epoch": epoch, "path": str(checkpoint_path), "sha256": checkpoint_sha,
            "sha256_matches_raw_log": True, "logged_val_loss": epoch_end[epoch]["val_loss"],
            "is_final": epoch == final_epoch,
            "config_matches_current_config": checkpoint["config"] == cfg,
        },
        "final_checkpoint": final_block,
        "validation": validation,
        "train_eval_mode": train,
        "generalization_gap": {
            "formula": "generalization_gap = validation_eval_ce - training_eval_ce",
            "definition": "both cross-entropies in eval mode (no dropout) over the full split, from the primary "
                          "checkpoint, in nats per character; positive means validation CE is above training CE",
            "validation_eval_ce": validation["ce_nats"],
            "training_eval_ce": train["ce_nats"],
            "value_nats": validation["ce_nats"] - train["ce_nats"],
        },
        "generation": generation,
        "training_run": training_run,
        "definitions": {
            "cross_entropy": "mean next-character NLL in nats, token-weighted, eval mode, all sequences of the split",
            "perplexity": "exp(CE)", "bpc": "CE / ln 2",
            "generalization_gap": "validation_eval_ce - training_eval_ce (both eval mode, full split, same checkpoint)",
            "top1_accuracy": "share of positions where argmax logits == next character",
            "distinct_n": "unique n-grams / all n-grams pooled over all continuations (prompt excluded); "
                          "primary over characters, secondary over words",
            "repeated_4gram_rate": "per continuation 1 - unique 4-grams / all 4-grams, averaged over "
                                   "continuations; primary over characters, secondary over words",
            "word_tokenizer": f"regex {WORD_RE.pattern} on lowercased text (secondary diagnostics only)",
            "unseen_word_share": "share of generated words that never occur in the training split (secondary "
                                 "diagnostic; the fixed-length cutoff can truncate a sample's last word, which "
                                 "inflates this share)",
            "generation_tokens_per_sec": "new characters / wall-clock seconds, batch size 1, CUDA-synchronized",
        },
        "provenance": {
            "raw_log": str(log_path), "raw_log_sha256": sha256_file(log_path),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "eval_code_git_commit": os.environ.get("TASK1_GIT_COMMIT"),
            "eval_script_sha256": sha256_file(Path(__file__).resolve()),
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0)},
            "eval_batch_size": batch_size,
        },
    }

    out_dir.mkdir(parents=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(out_dir / "generations.txt", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"=== sample {s['id']} | prompt {s['prompt_index']} ({s['prompt']!r}) #{s['sample_index']} ===\n"
                    f"{s['text']}\n\n")
    with open(out_dir / "failure_candidates.md", "w", encoding="utf-8") as f:
        f.write(candidates_markdown(samples, args.run_id, epoch))

    char_level = generation["primary_character_level"]
    print(f"  generalization_gap = validation_eval_ce - training_eval_ce = "
          f"{metrics['generalization_gap']['value_nats']:.4f} nats")
    print(f"  generation: {len(samples)} samples, {generation['tokens_per_sec']:.1f} tokens/s; character-level "
          f"Distinct-1/2/3 {char_level['distinct_1']:.4f}/{char_level['distinct_2']:.4f}/"
          f"{char_level['distinct_3']:.4f}, repeated 4-gram rate {char_level['repeated_4gram_rate']:.4f}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

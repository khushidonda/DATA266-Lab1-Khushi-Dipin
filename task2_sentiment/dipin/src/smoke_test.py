"""Task 2 (Dipin) smoke test: the whole pipeline end-to-end on a tiny subset.

    python task2_sentiment/dipin/src/smoke_test.py

Preprocesses 4,000 training reviews, trains all three models for 1 epoch, and runs the
full metric suite. Everything is written under outputs/smoke/ and checkpoints/*_smoke.pt,
so real results are never overwritten. Takes a few minutes on CPU.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import TASK_DIR, load_config, prepare_all, set_seed  # noqa: E402
from evaluate import build_slices, compute_metrics, error_candidates, mcnemar, slice_metrics  # noqa: E402
from train import get_device, load_model, make_loader, predict, train_model  # noqa: E402


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    smoke_dir = TASK_DIR / "outputs" / "smoke"
    data = prepare_all(cfg, max_train=4000, out_dir=smoke_dir / "data")
    vocab_size = len(data["stoi"])
    device = get_device()
    preds = {}
    for name in cfg["models"]:
        summary, ckpt = train_model(name, cfg, data, vocab_size, device, epochs=1, tag="_smoke")
        model, _ = load_model(ckpt, device)
        loader = make_loader(data["ids"]["test"], data["lengths"]["test"], data["labels"]["test"], 1024, False, 0)
        y, prob = predict(model, loader, device)
        metrics, preds[name] = compute_metrics(y, prob, cfg)
        slices = build_slices(data["raw_text"]["test"], data["unks"]["test"], data["lengths"]["test"],
                              data["truncated"]["test"])
        metrics["per_slice"] = slice_metrics(y, preds[name], slices)
        rows, _ = error_candidates(y, prob, preds[name], data["raw_text"]["test"], slices)
        assert len(rows) > 0
        print(name, json.dumps({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}))
    for name in ["experimental_1", "experimental_2"]:
        print("McNemar baseline vs", name, mcnemar(y, preds["baseline"], preds[name]))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

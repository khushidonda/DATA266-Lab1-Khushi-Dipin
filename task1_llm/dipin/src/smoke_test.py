"""Task 1 (Dipin) smoke test: the whole pipeline end-to-end on a tiny subset.

    python task1_llm/dipin/src/smoke_test.py

300 stories, 2 epochs, then the full evaluation (metrics, generation, plots). Everything is
written under outputs/smoke/ and checkpoints/smoke/ (git-ignored), so real results are never
touched. The model is far too undertrained to produce readable text; this only checks the code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import TASK_DIR, load_config, prepare  # noqa: E402
from evaluate import evaluate  # noqa: E402
from train import train  # noqa: E402
from utils import get_device  # noqa: E402


def main():
    cfg = load_config()
    cfg["generation"]["samples_per_prompt"] = 1
    cfg["generation"]["max_new_tokens"] = 60
    data = prepare(cfg, max_stories=300, out_dir=TASK_DIR / "outputs" / "smoke" / "data", verbose=False)
    summary = train(cfg, data, get_device(), epochs=2, smoke=True, resume=False)
    assert summary["nan_count"] == 0 and len(summary["history"]) == 2
    _, rows = evaluate(smoke=True, data=data)
    for name, value, _ in rows:
        print(f"{name:32s} {value}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

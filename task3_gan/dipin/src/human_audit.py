"""Blinded human audit: 30 fixed samples, 2 raters, inter-rater agreement.

prepare : picks 30 fixed held-out samples (seeded) -- 20 photo->Monet, 10 Monet->photo --
          shuffles them under anonymous ids S01..S30, writes input|output panels,
          an HTML viewer and one blank rating sheet per rater. The id->image key is
          kept in audit_key.json (raters must not open it).
score   : reads the two completed sheets and reports mean scores plus exact
          agreement, within-1 agreement and Cohen's kappa (unweighted and
          quadratic-weighted) per criterion.

Rubric (1-5, higher is better)
    style     : how convincingly the output looks like the target domain
    content   : how well scene layout / objects of the input are preserved
    artifacts : 5 = no visible artifacts, 1 = severe artifacts (checkerboard, blotches, colour bleeding)

Usage:
    python task3_gan/dipin/src/human_audit.py prepare
    python task3_gan/dipin/src/human_audit.py score
"""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import OUT, REPO, make_splits
from evaluate import CFG

AUDIT = OUT / "human_audit"
TDIR = OUT / "translations"
CRITERIA = ["style", "content", "artifacts"]
RATERS = ["rater1", "rater2"]


def prepare():
    splits = make_splits(CFG)
    rng = random.Random(CFG["seed"] + 30)
    a = rng.sample(splits["A_test"], 20)
    b = rng.sample(splits["B_test"], 10)
    items = [("photo->Monet", REPO / splits["A_dir"] / n, TDIR / "A2B_photo_to_monet" / (Path(n).stem + ".jpg")) for n in a]
    items += [("Monet->photo", REPO / splits["B_dir"] / n, TDIR / "B2A_monet_to_photo" / (Path(n).stem + ".jpg")) for n in b]
    rng.shuffle(items)
    (AUDIT / "panels").mkdir(parents=True, exist_ok=True)
    key, rows = {}, []
    for i, (direction, src, out) in enumerate(items, 1):
        sid = f"S{i:02d}"
        panel = Image.new("RGB", (522, 256), "white")
        panel.paste(Image.open(src).convert("RGB").resize((256, 256)), (0, 0))
        panel.paste(Image.open(out).convert("RGB"), (266, 0))
        panel.save(AUDIT / "panels" / f"{sid}.jpg", quality=95)
        key[sid] = {"direction": direction, "input": src.name}
        rows.append(f'<div class="s"><h3>{sid}</h3><img src="panels/{sid}.jpg"><p>left: input &nbsp; right: translation</p></div>')
    (AUDIT / "audit_key.json").write_text(json.dumps(key, indent=1))
    html = ("<!doctype html><meta charset='utf-8'><title>CycleGAN audit</title>"
            "<style>body{font-family:sans-serif;max-width:1100px;margin:auto}.s{display:inline-block;margin:8px}"
            "img{width:522px}</style><h1>Blinded CycleGAN audit (30 samples)</h1>"
            "<p>Score each sample 1-5 in your own sheet without discussing with the other rater. "
            "style = looks like the target domain (Monet painting or real photo); content = scene/objects of the "
            "input are preserved; artifacts = 5 means none, 1 means severe.</p>" + "".join(rows))
    (AUDIT / "audit_sheet.html").write_text(html, encoding="utf-8")
    for r in RATERS:
        p = AUDIT / f"ratings_{r}.csv"
        if not p.exists():
            with open(p, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["sample_id"] + CRITERIA + ["notes"])
                for sid in key:
                    w.writerow([sid, "", "", "", ""])
    print(f"prepared {len(key)} samples in {AUDIT}")


def cohen_kappa(x, y, weights=None, labels=range(1, 6)):
    labels = list(labels); k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for a, b in zip(x, y):
        O[idx[a], idx[b]] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    W = (i != j).astype(float) if weights is None else ((i - j) / (k - 1)) ** 2
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() > 0 else 1.0


def score():
    key = json.loads((AUDIT / "audit_key.json").read_text())
    ratings = {}
    for r in RATERS:
        with open(AUDIT / f"ratings_{r}.csv", newline="") as fh:
            ratings[r] = {row["sample_id"]: row for row in csv.DictReader(fh)}
        missing = [s for s in key for c in CRITERIA if not ratings[r].get(s, {}).get(c, "").strip()]
        if missing:
            raise SystemExit(f"{r} sheet incomplete: {len(missing)} empty cells")
    res = {"n_samples": len(key), "raters": RATERS, "criteria": {}}
    for c in CRITERIA:
        x = [int(ratings["rater1"][s][c]) for s in key]
        y = [int(ratings["rater2"][s][c]) for s in key]
        by_dir = {}
        for d in ("photo->Monet", "Monet->photo"):
            ids = [s for s in key if key[s]["direction"] == d]
            by_dir[d] = float(np.mean([(int(ratings["rater1"][s][c]) + int(ratings["rater2"][s][c])) / 2 for s in ids]))
        res["criteria"][c] = {
            "mean_rater1": float(np.mean(x)), "mean_rater2": float(np.mean(y)),
            "mean_both": float(np.mean(x + y)), "mean_by_direction": by_dir,
            "exact_agreement_pct": 100 * float(np.mean(np.array(x) == np.array(y))),
            "within1_agreement_pct": 100 * float(np.mean(np.abs(np.array(x) - np.array(y)) <= 1)),
            "cohen_kappa": cohen_kappa(x, y),
            "cohen_kappa_quadratic": cohen_kappa(x, y, "quadratic"),
        }
    c = res["criteria"]
    res["overall_audit_score"] = float(np.mean([c[k]["mean_both"] for k in CRITERIA]))
    res["overall_exact_agreement_pct"] = float(np.mean([c[k]["exact_agreement_pct"] for k in CRITERIA]))
    res["overall_quadratic_kappa"] = float(np.mean([c[k]["cohen_kappa_quadratic"] for k in CRITERIA]))
    (AUDIT / "audit_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"prepare": prepare, "score": score}[sys.argv[1]]()

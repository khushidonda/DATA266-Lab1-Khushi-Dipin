"""Checkpoint selection on our own held-out data.

Scores every generator snapshot saved during training (checkpoints/generators_epochXXX.pt)
with the same protocol as evaluate.py:
    A2B: 500 held-out test photos -> Monet, vs all 300 real Monet
    B2A: all 300 Monet -> photo,           vs the 500 held-out real photos
Selection rule (fixed before looking at results): lowest mean KID over both
directions (KID is unbiased at these sample sizes); FID mean breaks ties.
The class-competition metric is NOT used for selection.

Usage: python task3_gan/dipin/src/checkpoint_sweep.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CKPT, OUT, load_split, make_splits
from evaluate import CFG, Inception, fid, kid, translate
from models import ResnetGenerator


def gen(state, device):
    g = ResnetGenerator(ngf=CFG["generator"]["ngf"], n_blocks=CFG["generator"]["residual_blocks"])
    g.load_state_dict(state)
    return g.to(device).eval()


def main():
    device = torch.device("cuda")
    splits = make_splits(CFG)
    A_test, _ = load_split(splits, "A", "test")
    B_all = torch.cat([load_split(splits, "B", "test")[0], load_split(splits, "B", "train")[0]])
    inc = Inception(device)
    real_B, real_A = inc(B_all), inc(A_test)

    rows = []
    for p in sorted(CKPT.glob("generators_epoch*.pt")):
        sd = torch.load(p, map_location=device, weights_only=True)
        G_AB, G_BA = gen(sd["G_AB"], device), gen(sd["G_BA"], device)
        fB, fA = inc(translate(G_AB, A_test, device)), inc(translate(G_BA, B_all, device))
        r = {"epoch": int(p.stem[-3:]), "checkpoint": p.name,
             "FID_A2B": fid(real_B, fB), "KID_A2B": kid(real_B, fB)[0],
             "FID_B2A": fid(real_A, fA), "KID_B2A": kid(real_A, fA)[0]}
        r["KID_mean"] = (r["KID_A2B"] + r["KID_B2A"]) / 2
        r["FID_mean"] = (r["FID_A2B"] + r["FID_B2A"]) / 2
        rows.append(r)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)

    df = pd.DataFrame(rows).sort_values("epoch")
    best = df.sort_values(["KID_mean", "FID_mean"]).iloc[0]
    out = OUT / "eval"
    df.to_csv(out / "checkpoint_sweep.csv", index=False)
    sel = {"rule": "lowest mean KID over A2B and B2A on held-out data; FID mean breaks ties",
           "selected_epoch": int(best["epoch"]), "selected_checkpoint": best["checkpoint"],
           **{k: float(best[k]) for k in ["KID_A2B", "KID_B2A", "KID_mean", "FID_A2B", "FID_B2A", "FID_mean"]}}
    (out / "checkpoint_selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()

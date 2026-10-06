"""Task 3 (Khushi): final evaluation of the selected CycleGAN checkpoint on the fixed in-domain set.

Evaluation set (NOT a held-out set -- the model trained on these domains):
  A = all 300 sorted Monet images, B = first 300 sorted photos (exactly what the official
  course evaluator selects with N_EVAL = 300).

Computes (everything deterministic, eval() + torch.no_grad(), no random crop/flip):
  * pred_A2B / pred_B2A, 300 JPEG q95 files each, via the repo's generate.py (what the official score used)
  * Inception-v3 features, defined exactly as in the course evaluator (Resize 299, CenterCrop 299,
    ImageNet mean/std, fc=Identity -> 2048-d)
  * FID recomputed as a consistency check only (official values come from the course notebook)
  * KID, k-NN precision/recall (k=3), density/coverage (k=5)
  * cycle-reconstruction L1 and LPIPS(input, cycle) on raw tensors (before JPEG encoding), in [0, 1] / LPIPS convention
  * content cosine = cos(feature(input), feature(translation))

Usage:
  python evaluate_final.py --data-dir <dir containing monet_jpg/ and photo_jpg/> [--device cpu]
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from dataset import list_images  # noqa: E402
from generate import _load_input_tensor, generate_both_directions  # noqa: E402
from models import build_generator  # noqa: E402
from utils import load_checkpoint  # noqa: E402

KHUSHI = SRC.parent
EXPECTED_SHA = "d3124ab24a6f2201a03673cde5a416d04c78dd44985a3c7d590c48efc7e89d3c"
DEFAULT_CKPT = KHUSHI / "checkpoints" / "task3_khushi_prod_20260930_154246" / "epoch_28.pt"
RUN_ID = "final_eval_epoch029"
N_EVAL = 300
SEED = 42
KID_SUBSETS, KID_SUBSET_SIZE = 100, 100
PR_K, DC_K = 3, 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- Inception (official definition)
INCEPTION_TF = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def get_inception(device):
    m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    m.fc = nn.Identity()
    return m.to(device).eval()


@torch.no_grad()
def activations(model, paths, device, batch_size=32):
    feats = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
        feats.append(model(x).cpu().numpy().astype(np.float64))
    return np.concatenate(feats)


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    d = mu1 - mu2
    return float(d.dot(d) + np.trace(s1 + s2 - 2 * covmean))


def fid(real, gen):
    return frechet_distance(real.mean(0), np.cov(real, rowvar=False), gen.mean(0), np.cov(gen, rowvar=False))


# ---------------------------------------------------------------- KID
def poly_kernel(x, y):
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3          # degree 3, gamma = 1/d, coef0 = 1


def mmd2_unbiased(x, y):
    m, n = len(x), len(y)
    kxx, kyy, kxy = poly_kernel(x, x), poly_kernel(y, y), poly_kernel(x, y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())


def kid(real, gen, rng):
    vals = []
    for _ in range(KID_SUBSETS):
        i = rng.choice(len(real), KID_SUBSET_SIZE, replace=False)
        j = rng.choice(len(gen), KID_SUBSET_SIZE, replace=False)
        vals.append(mmd2_unbiased(real[i], gen[j]))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "full_sample_unbiased": mmd2_unbiased(real, gen)}


# ---------------------------------------------------------------- precision/recall, density/coverage
def pdist(a, b):
    a2, b2 = (a ** 2).sum(1)[:, None], (b ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2 * a @ b.T, 0.0))


def kth_radius(feats, k):
    d = pdist(feats, feats)
    return np.sort(d, axis=1)[:, k]          # index 0 is the point itself


def precision_recall(real, gen, k=PR_K):       # Kynkaanniemi et al. 2019
    rr, rg = kth_radius(real, k), kth_radius(gen, k)
    d = pdist(gen, real)                       # (gen, real)
    precision = float((d <= rr[None, :]).any(1).mean())
    recall = float((d.T <= rg[None, :]).any(1).mean())
    return precision, recall


def density_coverage(real, gen, k=DC_K):       # Naeem et al. 2020
    rr = kth_radius(real, k)
    d = pdist(gen, real)
    density = float((d <= rr[None, :]).sum() / (k * len(gen)))
    coverage = float((d.min(0) <= rr).mean())
    return density, coverage


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing monet_jpg/ and photo_jpg/")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(KHUSHI / "outputs" / RUN_ID))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    device = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.checkpoint)
    ck_sha = sha256(ckpt)
    assert ck_sha == EXPECTED_SHA, f"checkpoint SHA mismatch: {ck_sha}"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = Path(a.data_dir)
    monet = list_images(data / "monet_jpg")
    photo = list_images(data / "photo_jpg")
    assert len(monet) == 300 and len(photo) == 7038
    A_paths, B_paths = monet[:N_EVAL], photo[:N_EVAL]

    # 1. predictions (JPEG q95) via the repo's generate.py -- refuse to mix with existing files
    pred_a2b, pred_b2a = out / "pred_A2B", out / "pred_B2A"
    for d in (pred_a2b, pred_b2a):
        if d.exists() and any(d.iterdir()):
            sys.exit(f"{d} already exists and is not empty; refusing to overwrite/mix")
    res = generate_both_directions(ckpt, data / "monet_jpg", data / "photo_jpg", "", device=device,
                                   max_images_a2b=N_EVAL, max_images_b2a=N_EVAL, output_dir=out)
    # generate_both_directions writes to out/<run_id>/pred_*; run_id "" resolves to out/pred_*
    assert res["pred_A2B_count"] == N_EVAL and res["pred_B2A_count"] == N_EVAL
    gen_a2b = sorted(pred_a2b.glob("*.jpg"))
    gen_b2a = sorted(pred_b2a.glob("*.jpg"))
    assert len(gen_a2b) == len(gen_b2a) == N_EVAL
    assert [p.stem for p in gen_a2b] == [p.stem for p in A_paths] and [p.stem for p in gen_b2a] == [p.stem for p in B_paths]
    for p in gen_a2b + gen_b2a:
        with Image.open(p) as im:
            im.load()
            assert im.mode == "RGB" and im.size == (256, 256)
    print("predictions OK:", len(gen_a2b), len(gen_b2a), flush=True)

    # 2. cycle L1 + LPIPS on raw tensors
    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    g_a2b, g_b2a = build_generator().to(device), build_generator().to(device)
    load_checkpoint(ckpt, models={"G_A2B": g_a2b, "G_B2A": g_b2a}, map_location=device)
    g_a2b.eval(); g_b2a.eval()
    rows = {"A": [], "B": []}
    with torch.no_grad():
        for dom, paths, g_fwd, g_back in (("A", A_paths, g_a2b, g_b2a), ("B", B_paths, g_b2a, g_a2b)):
            for p in paths:
                x = _load_input_tensor(p, 256, device)
                fake = g_fwd(x)
                rec = g_back(fake)
                assert torch.isfinite(fake).all() and torch.isfinite(rec).all()
                l1 = float(((x - rec).abs() / 2.0).mean())            # [-1,1] -> [0,1] pixel scale
                rows[dom].append({
                    "file": p.stem, "cycle_L1": l1,
                    "lpips_cycle": float(lp(x, rec).item()),
                    "lpips_direct_translation": float(lp(x, fake).item()),
                    "direct_L1_translation": float(((x - fake).abs() / 2.0).mean()),
                })
    for dom in rows:
        with open(out / f"per_image_cycle_{dom}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[dom][0])); w.writeheader(); w.writerows(rows[dom])
    print("cycle/LPIPS done", flush=True)

    # 3. Inception features (cached)
    inc = get_inception(device)
    feats = {
        "real_monet": activations(inc, A_paths, device), "real_photo": activations(inc, B_paths, device),
        "gen_a2b": activations(inc, gen_a2b, device), "gen_b2a": activations(inc, gen_b2a, device),
    }
    (out / "cache").mkdir(exist_ok=True)
    np.savez_compressed(out / "cache" / "inception_features.npz", **{k: v.astype(np.float32) for k, v in feats.items()})
    print("features done", {k: v.shape for k, v in feats.items()}, flush=True)

    # 4. metrics
    # A2B: translated Monet vs real photos; B2A: translated photos vs real Monet
    pairs = {"A2B": (feats["real_photo"], feats["gen_a2b"], feats["real_monet"]),
             "B2A": (feats["real_monet"], feats["gen_b2a"], feats["real_photo"])}   # (real target, generated, input)
    m = {"checkpoint_sha256": ck_sha, "n_eval": N_EVAL, "seed": SEED, "device": str(device)}
    rng = np.random.RandomState(SEED)
    for d, (real, gen, inp) in pairs.items():
        m[f"fid_recomputed_{d}"] = fid(real, gen)
        k = kid(real, gen, rng)
        m[f"kid_{d}_subset_mean"], m[f"kid_{d}_subset_std"], m[f"kid_{d}_full_sample"] = k["mean"], k["std"], k["full_sample_unbiased"]
        m[f"precision_{d}"], m[f"recall_{d}"] = precision_recall(real, gen)
        m[f"density_{d}"], m[f"coverage_{d}"] = density_coverage(real, gen)
        cos = [float(np.dot(inp[i], gen[i]) / (np.linalg.norm(inp[i]) * np.linalg.norm(gen[i]))) for i in range(N_EVAL)]
        m[f"content_cosine_{d}"] = float(np.mean(cos))
        m[f"content_cosine_{d}_std"] = float(np.std(cos))
        with open(out / f"per_image_content_cosine_{d}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["file", "content_cosine"])
            src = A_paths if d == "A2B" else B_paths
            for p, c in zip(src, cos):
                w.writerow([p.stem, c])
    for dom, d in (("A", "A2B"), ("B", "B2A")):
        m[f"cycle_L1_{dom}"] = float(np.mean([r["cycle_L1"] for r in rows[dom]]))
        m[f"lpips_cycle_{dom}"] = float(np.mean([r["lpips_cycle"] for r in rows[dom]]))
        m[f"lpips_direct_{dom}"] = float(np.mean([r["lpips_direct_translation"] for r in rows[dom]]))
    m["cycle_L1_overall"] = (m["cycle_L1_A"] + m["cycle_L1_B"]) / 2
    m["lpips_cycle_overall"] = (m["lpips_cycle_A"] + m["lpips_cycle_B"]) / 2
    m["lpips_direct_overall"] = (m["lpips_direct_A"] + m["lpips_direct_B"]) / 2
    m["content_cosine_overall"] = (m["content_cosine_A2B"] + m["content_cosine_B2A"]) / 2
    m["protocol"] = {
        "evaluation_set": "fixed in-domain set: all 300 sorted Monet + first 300 sorted photos (model trained on these domains; not held-out)",
        "kid": f"unbiased MMD^2, polynomial kernel (x.y/d + 1)^3, d=2048, gamma=1/d, coef0=1, degree=3; {KID_SUBSETS} random subsets of {KID_SUBSET_SIZE} (no replacement) from 300 vs 300, RandomState({SEED}); also full-sample 300 vs 300",
        "precision_recall": f"Kynkaanniemi 2019 k-NN manifolds, k={PR_K}, Euclidean on 2048-d Inception features",
        "density_coverage": f"Naeem 2020, k={DC_K}",
        "cycle_L1": "mean |x - G_back(G_fwd(x))| on [0,1] pixel scale, raw tensors before JPEG",
        "lpips": f"lpips {lpips.__version__ if hasattr(lpips, '__version__') else '0.1.4'}, net=alex (v0.1), inputs in [-1,1], spatial mean",
        "content_cosine": "cos(Inception(input file), Inception(JPEG translation)), matched by sorted filename",
        "inception": "torchvision inception_v3 IMAGENET1K_V1, transform_input=False, fc=Identity, Resize299/CenterCrop299/ImageNet norm (course evaluator definition)",
    }
    with open(out / "metrics_final.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "protocol"}, indent=1))


if __name__ == "__main__":
    main()

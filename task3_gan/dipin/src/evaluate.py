"""Evaluate the trained CycleGAN in both directions.

A = photo, B = Monet.  A2B = photo -> Monet, B2A = Monet -> photo.

Inputs translated
    A2B : the 500 held-out test photos (never seen in training)
    B2A : all 300 Monet paintings for the distribution metrics (FID/KID/P-R need
          more than 30 samples), and the 30 held-out Monet for per-image metrics.
Real reference sets
    A2B : all 300 real Monet paintings (same reference the Kaggle MiFID uses)
    B2A : the 500 held-out real test photos

Metrics
    FID, KID (Inception-v3 pool3 2048-d), precision/recall (Kynkaanniemi 2019, k=3),
    density/coverage (Naeem 2020, k=5), Kaggle-style MiFID estimate,
    cycle-reconstruction L1 + PSNR, LPIPS(alex), content cosine similarity
    (Inception features of input vs translation), inference images/sec.
Pretrained networks (Inception, AlexNet-LPIPS) are used only to *measure*
images; they never generate or modify the submitted images.

Usage: python task3_gan/dipin/src/evaluate.py [--ckpt-dir task3_gan/dipin/checkpoints]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CKPT, DIPIN, OUT, load_split, make_splits, to_model_range, to_uint8
from models import ResnetGenerator

CFG = json.loads((DIPIN / "configs" / "task3_config.json").read_text())


# ---------------------------------------------------------------- features --
class Inception:
    def __init__(self, device):
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        self.net = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x_u8, bs=50):
        feats = [self.net(x_u8[i:i + bs].to(self.device))[0].double().cpu()
                 for i in range(0, len(x_u8), bs)]
        return torch.cat(feats).numpy()


def fid(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        off = np.eye(len(s1)) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid(f1, f2, n_subsets=100, subset_size=100, seed=0):
    """Unbiased MMD^2 with cubic polynomial kernel (Binkowski et al. 2018)."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f1), len(f2))
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def _pdist(a, b):
    a, b = torch.from_numpy(a), torch.from_numpy(b)
    return torch.cdist(a, b).numpy()


def prdc(real, fake, k_pr=3, k_dc=5):
    """Precision/recall (k-NN manifolds) and density/coverage."""
    rr = _pdist(real, real); ff = _pdist(fake, fake); rf = _pdist(real, fake)
    r_rad = {k: np.sort(rr, 1)[:, k] for k in (k_pr, k_dc)}  # col 0 is self
    f_rad = np.sort(ff, 1)[:, k_pr]
    precision = float((rf < r_rad[k_pr][:, None]).any(0).mean())
    recall = float((rf.T < f_rad[:, None]).any(0).mean())
    density = float((rf < r_rad[k_dc][:, None]).sum(0).mean() / k_dc)
    coverage = float((rf.min(1) < r_rad[k_dc]).mean())
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


def mifid(real, fake, fid_value, eps=0.1):
    """Kaggle GAN-competition MiFID: FID / memorization distance if it is below eps."""
    rn = real / np.linalg.norm(real, axis=1, keepdims=True)
    fn = fake / np.linalg.norm(fake, axis=1, keepdims=True)
    d = float((1 - np.abs(fn @ rn.T)).min(1).mean())
    return fid_value / (d if d < eps else 1.0), d


# -------------------------------------------------------------- generators --
def load_generator(path, device):
    g = ResnetGenerator(ngf=CFG["generator"]["ngf"], n_blocks=CFG["generator"]["residual_blocks"])
    g.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return g.to(device).eval()


@torch.no_grad()
def translate(G, x_u8, device, bs=16):
    out = []
    for i in range(0, len(x_u8), bs):
        out.append(to_uint8(G(to_model_range(x_u8[i:i + bs].to(device)))).cpu())
    return torch.cat(out)


def timed_translate(G, x_u8, device):
    translate(G, x_u8[:8], device)  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    y = translate(G, x_u8, device)
    torch.cuda.synchronize()
    return y, len(x_u8) / (time.time() - t)


def per_image(x, y, rec, lpips_net, feats_x, feats_y, device):
    """Per-image content metrics. x, y, rec are uint8 tensors."""
    xf, yf, rf = [t.float() / 255 for t in (x, y, rec)]
    cyc_l1 = (rf - xf).abs().mean((1, 2, 3)).numpy()
    mse = ((rf - xf) ** 2).mean((1, 2, 3)).numpy()
    psnr = 10 * np.log10(1.0 / np.maximum(mse, 1e-10))
    trans_l1 = (yf - xf).abs().mean((1, 2, 3)).numpy()
    lp_t, lp_r = [], []
    with torch.no_grad():
        for i in range(0, len(x), 25):
            a = to_model_range(x[i:i + 25]).to(device)
            lp_t.append(lpips_net(a, to_model_range(y[i:i + 25]).to(device)).flatten().cpu())
            lp_r.append(lpips_net(a, to_model_range(rec[i:i + 25]).to(device)).flatten().cpu())
    cos = (feats_x * feats_y).sum(1) / (np.linalg.norm(feats_x, axis=1) * np.linalg.norm(feats_y, axis=1))
    return dict(cycle_L1=cyc_l1, cycle_PSNR=psnr, translation_L1=trans_l1,
                LPIPS_input_vs_translation=torch.cat(lp_t).numpy(),
                LPIPS_input_vs_reconstruction=torch.cat(lp_r).numpy(), content_cosine=cos)


def save_jpgs(x_u8, names, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for img, n in zip(x_u8, names):
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(folder / (Path(n).stem + ".jpg"), quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(CKPT))
    ap.add_argument("--out-dir", default=str(OUT / "eval"))
    args = ap.parse_args()
    ckpt, out = Path(args.ckpt_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(0)

    splits = make_splits(CFG)
    A_test, A_names = load_split(splits, "A", "test")
    B_test, B_test_names = load_split(splits, "B", "test")
    B_train, B_train_names = load_split(splits, "B", "train")
    B_all = torch.cat([B_test, B_train]); B_all_names = B_test_names + B_train_names

    G_AB, G_BA = load_generator(ckpt / "G_AB.pt", device), load_generator(ckpt / "G_BA.pt", device)
    fake_B, ips_AB = timed_translate(G_AB, A_test, device)
    rec_A = translate(G_BA, fake_B, device)
    fake_A_all, ips_BA = timed_translate(G_BA, B_all, device)
    rec_B_all = translate(G_AB, fake_A_all, device)
    n_bt = len(B_test)

    inc = Inception(device)
    F = {k: inc(v) for k, v in dict(A_test=A_test, B_all=B_all, fake_B=fake_B,
                                     fake_A_all=fake_A_all).items()}

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    results, per_img = {}, {}
    for d, fake_f, real_f, src_f, x, y, rec, names, ips in [
        ("A2B", F["fake_B"], F["B_all"], F["A_test"], A_test, fake_B, rec_A, A_names, ips_AB),
        ("B2A", F["fake_A_all"], F["A_test"], F["B_all"], B_all[:n_bt], fake_A_all[:n_bt],
         rec_B_all[:n_bt], B_test_names, ips_BA),
    ]:
        f = fid(real_f, fake_f)
        k_mean, k_std = kid(real_f, fake_f)
        base_f = fid(real_f, src_f)  # untranslated inputs vs target domain
        base_k, _ = kid(real_f, src_f)
        mi, mem_d = mifid(real_f, fake_f, f)
        src_rows = src_f if d == "A2B" else src_f[:n_bt]
        fake_rows = fake_f if d == "A2B" else fake_f[:n_bt]
        pim = per_image(x, y, rec, lp, src_rows, fake_rows, device)
        per_img[d] = (names, pim)
        results[d] = {"FID": f, "KID": k_mean, "KID_std": k_std,
                      "FID_untranslated_baseline": base_f, "KID_untranslated_baseline": base_k,
                      "MiFID_estimate": mi, "memorization_distance": mem_d,
                      **prdc(real_f, fake_f),
                      **{k: float(np.mean(v)) for k, v in pim.items()},
                      **{k + "_std": float(np.std(v)) for k, v in pim.items()},
                      "n_fake": len(fake_f), "n_real_ref": len(real_f), "n_per_image": len(x),
                      "inference_images_per_sec": ips}
        print(d, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in results[d].items()}), flush=True)

    # cycle-consistency sanity check: an untrained generator pair gives far larger L1
    from models import init_weights
    torch.manual_seed(0)
    rnd_ab = init_weights(ResnetGenerator()).to(device).eval()
    rnd_ba = init_weights(ResnetGenerator()).to(device).eval()
    r = translate(rnd_ba, translate(rnd_ab, A_test[:100], device), device)
    results["cycle_check"] = {
        "random_init_generators_cycle_L1_A": float((r.float() - A_test[:100].float()).abs().mean() / 255),
        "trained_cycle_L1_A_first100": float(per_img["A2B"][1]["cycle_L1"][:100].mean()),
    }

    (out / "eval_metrics.json").write_text(json.dumps(results, indent=2))
    for d, (names, pim) in per_img.items():
        with open(out / f"per_image_{d}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image"] + list(pim))
            for i, n in enumerate(names):
                w.writerow([n] + [f"{pim[k][i]:.6f}" for k in pim])

    # translated test images for inspection / failure analysis
    tdir = OUT / "translations"
    save_jpgs(fake_B, A_names, tdir / "A2B_photo_to_monet")
    save_jpgs(rec_A, A_names, tdir / "A2B_cycle_reconstruction")
    save_jpgs(fake_A_all[:n_bt], B_test_names, tdir / "B2A_monet_to_photo")
    save_jpgs(rec_B_all[:n_bt], B_test_names, tdir / "B2A_cycle_reconstruction")
    print("saved", out / "eval_metrics.json")


if __name__ == "__main__":
    main()

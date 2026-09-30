"""Unpaired domain preparation.

Domain A = photos, domain B = Monet paintings (Kaggle "I'm Something of a
Painter Myself"). Images are read once into uint8 tensors held in RAM; the
training loop samples A and B independently (unpaired) and applies the
CycleGAN augmentation (resize 286 -> random crop 256 -> random h-flip) on GPU.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}
HERE = Path(__file__).resolve().parent
DIPIN = HERE.parent
REPO = DIPIN.parent.parent
# Smoke test: TASK3_SMOKE_ROOT points at a tiny synthetic dataset; splits and all
# outputs are then redirected under outputs/smoke so real results are untouched.
SMOKE_ROOT = os.environ.get("TASK3_SMOKE_ROOT")
# Experiment runs: TASK3_RUN=v2 uses configs/task3_config_v2.json and writes to
# outputs/v2 and checkpoints/v2; unset = the original run (v1) at the top level.
RUN = os.environ.get("TASK3_RUN", "")
CFG_PATH = DIPIN / "configs" / (f"task3_config_{RUN}.json" if RUN else "task3_config.json")
OUT = DIPIN / "outputs" / (RUN or "")
CKPT = DIPIN / "checkpoints" / (RUN or "")
if SMOKE_ROOT:
    OUT, CKPT = OUT / "smoke", CKPT / "smoke"


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def find_domain_dirs(data_root):
    """Locate the photo and Monet jpg folders anywhere under data_root."""
    data_root = Path(data_root)
    dirs = [p for p in data_root.rglob("*") if p.is_dir()] + [data_root]
    def pick(key):
        cands = [d for d in dirs if key in d.name.lower() and "tfrec" not in d.name.lower()
                 and any(f.suffix.lower() in IMG_EXT for f in d.iterdir())]
        if not cands:
            raise FileNotFoundError(f"no image folder containing '{key}' under {data_root}")
        return max(cands, key=lambda d: len(list_images(d)))
    return pick("photo"), pick("monet")


def make_splits(cfg, force=False):
    """Deterministic train/test split per domain, saved to data_processed/splits.json."""
    out = OUT / "splits.json" if SMOKE_ROOT else DIPIN / "data_processed" / "splits.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    photo_dir, monet_dir = find_domain_dirs(SMOKE_ROOT or REPO / cfg["data"]["root"])
    rng = random.Random(cfg["seed"])
    rel = (lambda p: str(p)) if SMOKE_ROOT else (lambda p: p.relative_to(REPO).as_posix())
    splits = {"A_dir": rel(photo_dir), "B_dir": rel(monet_dir)}
    for dom, folder, n_test in [("A", photo_dir, cfg["data"]["test_photos"]),
                                ("B", monet_dir, cfg["data"]["test_monet"])]:
        names = [p.name for p in list_images(folder)]
        if SMOKE_ROOT:
            n_test = min(n_test, len(names) // 4)
        rng.shuffle(names)
        splits[f"{dom}_test"] = sorted(names[:n_test])
        splits[f"{dom}_train"] = sorted(names[n_test:])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=1))
    return splits


def load_uint8(paths, size=256):
    """Stack images into a uint8 tensor (N, 3, size, size)."""
    arr = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        arr[i] = np.asarray(im)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def load_split(splits, domain, part, size=256):
    folder = REPO / splits[f"{domain}_dir"]
    names = splits[f"{domain}_{part}"]
    return load_uint8([folder / n for n in names], size), names


def to_model_range(x_uint8):
    """uint8 [0,255] -> float [-1,1]."""
    return x_uint8.float().div(127.5).sub(1.0)


def to_uint8(x):
    """float [-1,1] -> uint8 [0,255]."""
    return x.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)


class GpuAugment:
    """resize(load) -> random crop(crop) -> random horizontal flip, per image."""

    def __init__(self, load_size=286, crop_size=256, flip=True, generator=None):
        self.load, self.crop, self.flip = load_size, crop_size, flip
        self.g = generator

    def __call__(self, x):  # x float (N,3,H,W) on device
        x = F.interpolate(x, size=(self.load, self.load), mode="bicubic",
                          align_corners=False).clamp(-1, 1)
        out = []
        for img in x:
            i, j = torch.randint(0, self.load - self.crop + 1, (2,), generator=self.g).tolist()
            img = img[:, i:i + self.crop, j:j + self.crop]
            if self.flip and torch.rand(1, generator=self.g).item() < 0.5:
                img = img.flip(-1)
            out.append(img)
        return torch.stack(out)


class UnpairedSampler:
    """Independent uniform sampling from each domain -> no fixed A/B pairing."""

    def __init__(self, A, B, batch_size, seed):
        self.A, self.B, self.bs = A, B, batch_size
        self.g = torch.Generator().manual_seed(seed)

    def next(self):
        ia = torch.randint(0, len(self.A), (self.bs,), generator=self.g)
        ib = torch.randint(0, len(self.B), (self.bs,), generator=self.g)
        return self.A[ia], self.B[ib]

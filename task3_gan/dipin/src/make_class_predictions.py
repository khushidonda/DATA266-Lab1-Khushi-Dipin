"""Prediction folders for the class Kaggle competition's evaluation script.

The class script (Part3_Evaluation_Script.ipynb) names the domains the other way
round from this repo:  its A = Monet, B = photo.  So
    pred_A2B  (Monet -> photo)  = our G_BA applied to all 300 Monet paintings
    pred_B2A  (photo -> Monet)  = our G_AB applied to the 500 held-out test photos
Every image is the direct output of our trained generators; nothing is selected,
edited or post-processed. The class script then uses the first 300 files of each folder.

Usage: python task3_gan/dipin/src/make_class_predictions.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CKPT, DIPIN, OUT, load_split, make_splits
from evaluate import CFG, load_generator, translate


def save(imgs, names, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for img, n in zip(imgs, names):
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(folder / (Path(n).stem + ".jpg"), quality=95)


def main():
    device = torch.device("cuda")
    splits = make_splits(CFG)
    out = OUT / "class_kaggle"
    photos, photo_names = load_split(splits, "A", "test")
    m_test, m_test_names = load_split(splits, "B", "test")
    m_train, m_train_names = load_split(splits, "B", "train")
    monet = torch.cat([m_test, m_train]); monet_names = m_test_names + m_train_names

    G_AB = load_generator(CKPT / "G_AB.pt", device)
    G_BA = load_generator(CKPT / "G_BA.pt", device)
    save(translate(G_BA, monet, device), monet_names, out / "pred_A2B")
    save(translate(G_AB, photos, device), photo_names, out / "pred_B2A")

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    info = {"pred_A2B": {"direction": "Monet -> photo", "generator": "checkpoints/G_BA.pt",
                         "generator_sha256": sha(CKPT / "G_BA.pt"), "inputs": "all 300 monet_jpg",
                         "n": len(monet_names)},
            "pred_B2A": {"direction": "photo -> Monet", "generator": "checkpoints/G_AB.pt",
                         "generator_sha256": sha(CKPT / "G_AB.pt"),
                         "inputs": "500 held-out test photos (data_processed/splits.json)",
                         "n": len(photo_names)},
            "post_processing": "none"}
    (out / "predictions_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

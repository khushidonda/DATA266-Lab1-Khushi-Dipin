"""Build the Kaggle submission (images.zip) from the trained photo->Monet generator.

Every photo in the dataset (7,038) is passed once through G_AB. The JPEGs are
written exactly as the network produces them -- no selection, editing or
post-processing -- then zipped. Kaggle accepts 7,000-10,000 images of 256x256.

Usage: python task3_gan/dipin/src/make_submission.py
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CKPT, DIPIN, OUT, REPO, list_images, load_uint8, make_splits
from evaluate import CFG, load_generator, translate


def main():
    device = torch.device("cuda")
    splits = make_splits(CFG)
    photos = list_images(REPO / splits["A_dir"])
    G_AB = load_generator(CKPT / "G_AB.pt", device)
    out = OUT / "kaggle_submission"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(photos), 256):
        chunk = photos[i:i + 256]
        fake = translate(G_AB, load_uint8(chunk), device)
        for p, img in zip(chunk, fake):
            Image.fromarray(img.permute(1, 2, 0).numpy()).save(img_dir / f"{p.stem}.jpg", quality=95)

    files = sorted(img_dir.glob("*.jpg"))
    zpath = out / "images.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    g_hash = hashlib.sha256((CKPT / "G_AB.pt").read_bytes()).hexdigest()
    z_hash = hashlib.sha256(zpath.read_bytes()).hexdigest()
    info = {"n_images": len(files), "image_size": 256, "zip": zpath.relative_to(DIPIN).as_posix(),
            "zip_sha256": z_hash, "generator": "checkpoints/G_AB.pt", "generator_sha256": g_hash,
            "source": "direct G_AB inference on every photo_jpg image, no post-processing"}
    (out / "submission_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

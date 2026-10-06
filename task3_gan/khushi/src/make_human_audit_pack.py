"""Task 3 (Khushi): build the blinded 30-sample human-audit pack from the fixed in-domain evaluation set.

Selection: RandomState(42); 15 A2B (Monet -> photo) and 15 B2A (photo -> Monet), sampled without replacement
from the fixed 300 + 300 files; the 30 samples are then shuffled (same RandomState) and numbered S01..S30 so the
ID order does not reveal the direction. Panels show only "<ID>: input | translation" -- no model, epoch, metric or
checkpoint. The private mapping is for the analyst, not the raters. Rating sheets are created EMPTY.

Usage: python make_human_audit_pack.py --data-dir <dir containing monet_jpg/ and photo_jpg/>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from dataset import list_images  # noqa: E402

KHUSHI = SRC.parent
EVAL = KHUSHI / "outputs" / "final_eval_epoch029"
AUDIT = KHUSHI / "outputs" / "human_audit"

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
a = ap.parse_args()
data = Path(a.data_dir)
monet = list_images(data / "monet_jpg")[:300]
photo = list_images(data / "photo_jpg")[:300]

rng = np.random.RandomState(42)
a2b = [("A2B", monet[i], EVAL / "pred_A2B" / f"{monet[i].stem}.jpg") for i in rng.choice(300, 15, replace=False)]
b2a = [("B2A", photo[i], EVAL / "pred_B2A" / f"{photo[i].stem}.jpg") for i in rng.choice(300, 15, replace=False)]
samples = a2b + b2a
order = rng.permutation(30)
samples = [samples[i] for i in order]

(AUDIT / "panels").mkdir(parents=True, exist_ok=True)
pages, panels = [], []
with open(AUDIT / "audit_mapping_private.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "direction", "input_file", "translation_file"])
    for n, (d, src, gen) in enumerate(samples, 1):
        sid = f"S{n:02d}"
        w.writerow([sid, d, src.name, gen.relative_to(KHUSHI.parent.parent).as_posix()])
        panel = Image.new("RGB", (512, 280), "white")
        panel.paste(Image.open(src).convert("RGB").resize((256, 256)), (0, 24))
        panel.paste(Image.open(gen).convert("RGB").resize((256, 256)), (256, 24))
        dr = ImageDraw.Draw(panel)
        dr.text((6, 6), f"{sid}    INPUT (left)  |  TRANSLATION (right)", fill="black")
        panel.save(AUDIT / "panels" / f"{sid}.png")
        panels.append(panel)

# audit_grid.pdf: 6 panels per page (2 columns x 3 rows)
for p in range(0, 30, 6):
    page = Image.new("RGB", (1024 + 30, 3 * 290 + 10), "white")
    for k, panel in enumerate(panels[p:p + 6]):
        page.paste(panel, (10 + (k % 2) * 522, 10 + (k // 2) * 290))
    pages.append(page)
pages[0].save(AUDIT / "audit_grid.pdf", save_all=True, append_images=pages[1:])

for r in (1, 2):
    with open(AUDIT / f"ratings_rater{r}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "style_quality_1_to_5", "content_preservation_1_to_5", "artifact_severity_1_to_5", "optional_notes"])
        for n in range(1, 31):
            w.writerow([f"S{n:02d}", "", "", "", ""])
print("wrote", len(samples), "panels;", sum(1 for s in samples if s[0] == "A2B"), "A2B /", sum(1 for s in samples if s[0] == "B2A"), "B2A")

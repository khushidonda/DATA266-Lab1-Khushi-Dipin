"""Task 3 (Dipin): version-aware metrics_report.csv built only from committed artifacts.

v1 = baseline run (config task3_config.json, outputs/), v2 = improved final run (task3_config_v2.json, outputs/v2/).
Every row names the model version it describes. In this model's own naming A = photos, B = Monet,
G_AB = photo -> Monet ("photo->Monet") and G_BA = Monet -> photo ("Monet->photo").
Sources: outputs/{,v2/}eval/eval_metrics.json, train_summary.json, raw log jsonl (last epoch record),
class_kaggle/predictions_info.json + submission.csv.
"""
import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent
RUNS = {"v1": ("outputs", "outputs/logs/task3_dipin_full_20260929_144428.jsonl"),
        "v2": ("outputs/v2", "outputs/v2/logs/task3_dipin_full_20260929_174752.jsonl")}
DIRS = {"A2B": "photo->Monet", "B2A": "Monet->photo"}
rows = []
add = lambda v, d, m, val, note: rows.append([v, d, m, "" if val is None else val, note])

for v, (base, log) in RUNS.items():
    ev = json.load(open(D / base / "eval/eval_metrics.json"))
    ts = json.load(open(D / base / "train_summary.json"))
    last = [r for r in map(json.loads, open(D / log)) if r.get("event") == "epoch"][-1]
    proto = {"A2B": "500 held-out test photos translated vs 300 real Monet", "B2A": "300 Monet translated vs 500 held-out test photos (per-image metrics: 30 held-out Monet)"}
    for k, lab in DIRS.items():
        e = ev[k]
        add(v, lab, "FID", e["FID"], f"{proto[k]}; untranslated baseline {e['FID_untranslated_baseline']:.2f}; Inception pool3")
        add(v, lab, "KID", e["KID"], f"std {e['KID_std']:.4f} over 100 subsets of 100; untranslated baseline {e['KID_untranslated_baseline']:.4f}")
        add(v, lab, "generative_precision", e["precision"], "Kynkaanniemi 2019, k=3")
        add(v, lab, "generative_recall", e["recall"], "Kynkaanniemi 2019, k=3")
        add(v, lab, "density", e["density"], "Naeem 2020, k=5")
        add(v, lab, "coverage", e["coverage"], "Naeem 2020, k=5")
        add(v, lab, "cycle_reconstruction_L1", e["cycle_L1"], f"[0,1] pixel scale; PSNR {e['cycle_PSNR']:.2f} dB")
        add(v, lab, "LPIPS_input_vs_translation", e["LPIPS_input_vs_translation"], "AlexNet LPIPS")
        add(v, lab, "LPIPS_input_vs_cycle_reconstruction", e["LPIPS_input_vs_reconstruction"], "AlexNet LPIPS")
        add(v, lab, "content_preservation_cosine_similarity", e["content_cosine"], "Inception pool3 features, input vs translation")
    add(v, "overall", "generator_loss", last["loss_G"], "final-epoch mean, total = adv + lambda_cyc*cyc + lambda_id*id")
    add(v, "overall", "discriminator_loss", (last["loss_D_A"] + last["loss_D_B"]) / 2, f"final-epoch mean; D_A {last['loss_D_A']:.4f}, D_B {last['loss_D_B']:.4f}")
    add(v, "overall", "cycle_consistency_loss", last["loss_cyc"], "final-epoch mean, unweighted L1 sum of both cycles, [-1,1] scale")
    add(v, "overall", "identity_loss", last["loss_id"], "final-epoch mean, unweighted L1 sum of both directions")
    add(v, "overall", "gradient_norm_G", last["grad_norm_G"], f"final-epoch mean; D {last['grad_norm_D']:.3f}")
    add(v, "overall", "nan_count", ts["nan_count_total"], "non-finite loss/grad steps over the run")
    p = ts["params"]
    add(v, "overall", "parameter_count", p["total"], f"G_AB {p['G_AB']}, G_BA {p['G_BA']}, D_A {p['D_A']}, D_B {p['D_B']}")
    add(v, "overall", "training_time_hours", ts["train_hours"], f"{ts['steps']} steps x batch {ts['batch_size']} on NVIDIA GeForce RTX 4090")
    add(v, "overall", "images_per_sec", ts["images_per_sec"], "training; one paired photo+Monet sample per image")
    add(v, "overall", "peak_memory_gb", ts["peak_memory_gb_since_last_start"], "torch.cuda.max_memory_allocated during training")
    ck = "checkpoints/G_AB.pt" if v == "v1" else "checkpoints/v2/G_AB.pt"
    info = json.load(open(D / base / "class_kaggle/predictions_info.json"))
    sub = list(csv.DictReader(open(D / base / "class_kaggle/submission.csv")))[0]
    f_, m_ = float(sub["FID"]), float(sub["MiFID"])
    add(v, "overall", "class_evaluator_FID", f_, f"course evaluator (N_EVAL=300) on predictions from the {ck.rsplit('/', 1)[0]}/ generators; mean of both directions")
    add(v, "overall", "class_evaluator_MiFID", m_, "course evaluator MiFID, mean of both directions")
    add(v, "overall", "class_competition_metric", -(f_ + m_) / 2, "-(FID+MiFID)/2 computed from the two rows above; a calculated value, not a leaderboard record")
for v in RUNS:
    add(v, "overall", "human_audit_score", None, "no ratings recorded for this version (rating sheets in outputs/human_audit/ are blank)")
    add(v, "overall", "inter_rater_agreement", None, "no ratings recorded for this version")
with open(D / "metrics_report.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model_version", "direction", "metric", "value", "notes"]); w.writerows(rows)
print(len(rows), "rows")

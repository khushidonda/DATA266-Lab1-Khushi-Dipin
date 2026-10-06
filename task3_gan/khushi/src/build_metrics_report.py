"""Task 3 (Khushi): assemble full_metrics_report.csv / metrics_report.csv from verified artifacts only.
Inputs (all produced earlier, nothing is recomputed or typed in here):
  outputs/phase0/sweep/epoch_029/metrics.json   official course-evaluator FID/MiFID (checkpoint epoch_28.pt)
  outputs/final_eval_epoch029/metrics_final.json  KID, precision/recall, cycle L1, LPIPS, content cosine
  outputs/plots/training_summary.json + training_epoch_metrics.csv  production raw-log statistics
Human-audit rows come from agreement_results.json (compute_agreement.py on the real rater sheets); Kaggle evidence is in outputs/kaggle/ and the report.
"""
import csv
import json
from pathlib import Path

K = Path(__file__).resolve().parent.parent
off = json.load(open(K / "outputs/phase0/sweep/epoch_029/metrics.json"))
fin = json.load(open(K / "outputs/final_eval_epoch029/metrics_final.json"))
tr = json.load(open(K / "outputs/plots/training_summary.json"))
CK = "epoch_28.pt (completed epoch 29) sha256 " + fin["checkpoint_sha256"]
SET = "fixed in-domain set: 300 Monet + first 300 sorted photos (not held-out)"
OFFP = f"official course evaluator, N_EVAL=300, JPEG q95, Inception-v3; {SET}"
OFFS = "outputs/phase0/sweep/epoch_029/metrics.json; outputs/official_eval/epoch_029/*.ipynb"
FIN = "outputs/final_eval_epoch029/metrics_final.json"
LOG = "reproducibility/raw_logs/khushi/task3_khushi_prod_20260930_154246*.jsonl (epoch_end records, epochs 1-29)"
P = fin["protocol"]
rows = []
add = lambda m, d, v, u, p, s: rows.append([m, d, "" if v is None else v, u, p, s, CK])

add("FID", "A2B (Monet->Photo)", off["fid_A2B"], "score", OFFP, OFFS)
add("FID", "B2A (Photo->Monet)", off["fid_B2A"], "score", OFFP, OFFS)
add("FID", "average", off["sub_fid"], "score", OFFP + "; mean of both directions = submission FID", OFFS)
add("MiFID", "A2B (Monet->Photo)", off["mifid_A2B"], "score", OFFP + "; evaluator MiFID exactly as supplied", OFFS)
add("MiFID", "B2A (Photo->Monet)", off["mifid_B2A"], "score", OFFP + "; evaluator MiFID exactly as supplied", OFFS)
add("MiFID", "average", off["sub_mifid"], "score", OFFP + "; mean of both directions = submission MiFID", OFFS)
add("FID_recomputed_cpu", "A2B (Monet->Photo)", fin["fid_recomputed_A2B"], "score", "consistency check only: evaluator FID code on CPU, regenerated JPEG q95 predictions; official value is the row above", FIN)
add("FID_recomputed_cpu", "B2A (Photo->Monet)", fin["fid_recomputed_B2A"], "score", "consistency check only (see above)", FIN)
for d in ("A2B", "B2A"):
    lab = "A2B (Monet->Photo)" if d == "A2B" else "B2A (Photo->Monet)"
    add("KID", lab, fin[f"kid_{d}_subset_mean"], "MMD^2", P["kid"] + f"; subset std {fin[f'kid_{d}_subset_std']:.6f}", FIN)
    add("KID_subset_std", lab, fin[f"kid_{d}_subset_std"], "MMD^2", "std over the 100 subsets above", FIN)
    add("KID_full_sample", lab, fin[f"kid_{d}_full_sample"], "MMD^2", "unbiased MMD^2 on all 300 vs 300 (same kernel)", FIN)
    add("generative_precision", lab, fin[f"precision_{d}"], "fraction", P["precision_recall"], FIN)
    add("generative_recall", lab, fin[f"recall_{d}"], "fraction", P["precision_recall"], FIN)
    add("density", lab, fin[f"density_{d}"], "score", "supplementary; " + P["density_coverage"], FIN)
    add("coverage", lab, fin[f"coverage_{d}"], "fraction", "supplementary; " + P["density_coverage"], FIN)
    add("content_preservation_cosine_similarity", lab, fin[f"content_cosine_{d}"], "cosine", P["content_cosine"] + f"; per-image std {fin[f'content_cosine_{d}_std']:.4f}", FIN)
add("content_preservation_cosine_similarity", "overall (mean of A2B,B2A)", fin["content_cosine_overall"], "cosine", P["content_cosine"], FIN)
for dom, lab in (("A", "A (Monet cycle A->B->A)"), ("B", "B (Photo cycle B->A->B)")):
    add("cycle_reconstruction_L1", lab, fin[f"cycle_L1_{dom}"], "mean abs error, [0,1] pixels", P["cycle_L1"], FIN)
    add("LPIPS_cycle", lab, fin[f"lpips_cycle_{dom}"], "LPIPS", P["lpips"] + "; input vs cycle reconstruction", FIN)
    add("LPIPS_direct_translation", lab, fin[f"lpips_direct_{dom}"], "LPIPS", "supplementary: input vs direct translation (style change is intended); " + P["lpips"], FIN)
add("cycle_reconstruction_L1", "overall", fin["cycle_L1_overall"], "mean abs error, [0,1] pixels", P["cycle_L1"], FIN)
add("LPIPS_cycle", "overall", fin["lpips_cycle_overall"], "LPIPS", P["lpips"] + "; input vs cycle reconstruction", FIN)
add("LPIPS_direct_translation", "overall", fin["lpips_direct_overall"], "LPIPS", "supplementary; " + P["lpips"], FIN)

m = tr["selected_epoch_means"]
ep = "mean over the 7,038 steps of completed epoch 29 (0-indexed epoch 28)"
add("generator_loss_total", "overall", m["G_total"], "loss", ep + "; adv + 10*cycle + 5*identity", LOG)
add("discriminator_loss_D_A", "Monet discriminator", m["D_A"], "loss", ep + "; LSGAN", LOG)
add("discriminator_loss_D_B", "photo discriminator", m["D_B"], "loss", ep + "; LSGAN", LOG)
add("cycle_consistency_loss_A", "A", m["cycle_A"], "L1, [-1,1] scale, unweighted", ep, LOG)
add("cycle_consistency_loss_B", "B", m["cycle_B"], "L1, [-1,1] scale, unweighted", ep, LOG)
add("identity_loss_A", "A", m["identity_A"], "L1, [-1,1] scale, unweighted", ep, LOG)
add("identity_loss_B", "B", m["identity_B"], "L1, [-1,1] scale, unweighted", ep, LOG)
add("gradient_norm_G", "generators", m["G_grad_norm"], "L2 norm", ep, LOG)
add("gradient_norm_D_A", "Monet discriminator", m["D_A_grad_norm"], "L2 norm", ep, LOG)
add("gradient_norm_D_B", "photo discriminator", m["D_B_grad_norm"], "L2 norm", ep, LOG)
add("nan_inf_abort_events", "overall", tr["recorded_nan_inf_abort_events"], "count", "recorded NaN/Inf abort events (nan_detected/run_aborted records) in both log segments; the logger has no explicit counter", LOG)
add("parameter_count", "total (G_A2B 11,378,179 + G_B2A 11,378,179 + D_A 2,764,737 + D_B 2,764,737)", 28285832, "parameters", "counted from checkpoint state dicts loaded into the current source; matches raw-log run_start", "checkpoint + raw log run_start")
add("training_time", "through completed epoch 29", round(tr["training_seconds_through_selected_epoch"], 3), "seconds", "sum of epoch_seconds for epochs 1-29 (RTX 4090; run was paused/resumed once after epoch 5)", LOG)
add("images_per_sec", "selected epoch 29", tr["selected_epoch_images_per_second"], "images/s", "batch size 1, from epoch_end record", LOG)
add("images_per_sec", "mean epochs 1-29", tr["mean_images_per_second_epochs_1_to_29"], "images/s", "mean of per-epoch values", LOG)
add("peak_memory_usage", "max over epochs 1-29", tr["peak_memory_bytes_max_over_epochs_1_to_29"], "bytes (torch.cuda.max_memory_allocated)", "per-epoch peak, maximum", LOG)
ha = json.load(open(K / "outputs/human_audit/agreement_results.json"))
HAP = "blinded audit, 30 fixed samples (15 A2B + 15 B2A, seed 42), 2 independent raters, 1-5 ordinal"
HAS = "outputs/human_audit/ratings_rater1.csv, ratings_rater2.csv, agreement_results.json, human_audit_summary.csv"
for c, v in ha["criteria"].items():
    name = c.replace("_1_to_5", "")
    note = " (1 = no visible artifacts, 5 = severe artifacts; lower is better)" if name == "artifact_severity" else " (higher is better)"
    add(f"human_audit_{name}", "combined mean of 2 raters", v["combined_mean"], "1-5", f"{HAP}{note}; combined SD {v['combined_sd']:.3f}; rater1 {v['rater1_mean']:.3f} +/- {v['rater1_sd']:.3f}, rater2 {v['rater2_mean']:.3f} +/- {v['rater2_sd']:.3f}", HAS)
for c, v in ha["criteria"].items():
    name = c.replace("_1_to_5", "")
    add(f"inter_rater_quadratic_kappa_{name}", "2 raters", v["quadratic_weighted_kappa"], "kappa", f"quadratic-weighted Cohen's kappa; exact agreement {v['exact_agreement_pct']:.1f}%; mean absolute difference {v['mean_abs_difference']:.3f}", HAS)
    add(f"inter_rater_exact_agreement_{name}", "2 raters", v["exact_agreement_pct"], "percent", "share of the 30 samples with identical scores", HAS)

KP = "team PairProgramming_Team_1 class-competition leaderboard screenshot; rank is the position at the time of the screenshot"
KS = "outputs/kaggle/kaggle_leaderboard_rank29_score_-49.5121.png; outputs/kaggle/README.md"
sub = list(csv.DictReader(open(K / "outputs/submission.csv")))[0]
add("kaggle_score", "team leaderboard row", -49.5121, "score (higher is better)", KP + "; banner reads: previous best -49.8366", KS)
add("kaggle_rank", "team leaderboard row", 29, "rank", KP, KS)
add("kaggle_score_calculated", "from submission.csv", -(float(sub["FID"]) + float(sub["MiFID"])) / 2, "score", "-(FID + MiFID)/2 from outputs/submission.csv; calculated cross-check, equals the screenshot score to 4 decimals", "outputs/submission.csv")

hdr = ["metric", "direction", "value", "unit", "protocol", "source_evidence", "checkpoint"]
for name in ("full_metrics_report.csv", "metrics_report.csv"):
    with open(K / name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print(len(rows), "rows")

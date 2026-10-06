# Task 3 (Khushi) — blinded human audit pack (ratings NOT yet collected)

- 30 fixed samples: 15 Monet->photo (A2B) and 15 photo->Monet (B2A), drawn with `RandomState(42)` without replacement from the fixed in-domain 300 + 300 evaluation set, then shuffled and numbered S01-S30. Not hand-picked.
- `panels/S01.png ... S30.png` and `audit_grid.pdf` show only `ID | input (left) | translation (right)`. No model, epoch, metric, checkpoint or rater information appears.
- **Give raters only `panels/` (or `audit_grid.pdf`) and their own sheet. Do not share `audit_mapping_private.csv`** (it reveals direction and source files).
- Each rater fills their own sheet independently (`ratings_rater1.csv`, `ratings_rater2.csv`), one integer 1-5 per cell:
  - `style_quality_1_to_5`: 1 = poor Monet-/photo-like style for the target domain, 5 = excellent.
  - `content_preservation_1_to_5`: 1 = scene content lost, 5 = content fully preserved.
  - `artifact_severity_1_to_5`: 1 = no visible artifacts, 5 = severe artifacts (note: higher is worse).
  - `optional_notes`: free text.
- After both sheets are complete: `python compute_agreement.py` -> mean/SD per rater and criterion, combined mean, quadratic-weighted Cohen's kappa, exact agreement %, mean absolute difference (also written to `agreement_results.json`). It refuses to run if any cell is empty or outside 1-5.
- The rating sheets in this folder are intentionally empty.

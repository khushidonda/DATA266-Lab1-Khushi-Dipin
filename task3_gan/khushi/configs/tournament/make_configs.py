"""Generate the five tournament configs from the locked baseline config.
Only cycle_consistency_weight, identity_loss_weight, learning_rate and D_learning_rate differ;
everything else is copied verbatim from ../task3_config.json."""
import copy, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = json.loads((HERE.parent / "task3_config.json").read_text())

CANDIDATES = {
    "A": dict(cycle=10.0, identity=2.5, g_lr=2e-4, d_lr=2e-4),
    "B": dict(cycle=10.0, identity=1.0, g_lr=2e-4, d_lr=2e-4),
    "C": dict(cycle=10.0, identity=0.0, g_lr=2e-4, d_lr=2e-4),
    "D": dict(cycle=7.5, identity=1.0, g_lr=2e-4, d_lr=2e-4),
    "E": dict(cycle=10.0, identity=1.0, g_lr=2e-4, d_lr=1e-4),
}

for name, c in CANDIDATES.items():
    cfg = copy.deepcopy(base)
    t = cfg["training"]
    t["cycle_consistency_weight"] = c["cycle"]
    t["identity_loss_weight"] = c["identity"]
    t["identity_loss_enabled"] = c["identity"] > 0
    t["learning_rate"] = c["g_lr"]
    t["D_learning_rate"] = c["d_lr"]
    cfg["tournament_candidate"] = name
    path = HERE / f"task3_cand{name}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; configs are immutable once created")
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(path.name, c)

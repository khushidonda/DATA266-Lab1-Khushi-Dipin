"""One-command smoke test for the whole repository (CPU, a few seconds, no dataset download needed).

    python smoke_test.py

For every task it loads the committed final checkpoint with the member's own model code, performs a strict
state-dict load, runs a forward pass and checks the output is finite:
  Task 1 (Khushi)  character-level GPT  -> next-character prediction + a short greedy continuation
  Task 2 (Dipin)   BiLSTM + attention   -> sentiment logits for random token ids
  Task 3 (Dipin)   CycleGAN generator   -> 256 x 256 RGB translation
  Task 3 (Khushi)  CycleGAN generators  -> 256 x 256 RGB translations (if epoch_28.pt is present; the 340 MB
                                           file is distributed with the Canvas package / Drive, not through Git)
Paths are relative to this file; nothing is written to disk. Exit code 0 means every check passed.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
torch.manual_seed(42)
results = []


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------- Task 1 (Khushi)
t1 = ROOT / "task1_llm" / "khushi"
ckpt = t1 / "checkpoints" / "task1_full_20260925_204011" / "epoch_9.pt"
model_mod = load_module("t1_model", t1 / "src" / "model.py")
model, _ = model_mod.build_model_from_config()
model.eval()
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_state_dict"], strict=True)
tok = json.load(open(t1 / "data_processed" / "tokenizer.json"))
c2i, i2c = tok["char_to_idx"], {i: c for c, i in tok["char_to_idx"].items()}
text = "Once upon a time"
idx = torch.tensor([[c2i[c] for c in text]])
with torch.no_grad():
    for _ in range(40):
        logits, _ = model(idx[:, -128:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
print("        greedy continuation:", repr("".join(i2c[int(i)] for i in idx[0])))
check("Task 1 GPT epoch_9.pt", model.num_parameters() == 574830 and torch.isfinite(logits).all(),
      f"{model.num_parameters():,} parameters, SHA-256 {sha256(ckpt)[:12]}..., epoch {ck['epoch']}")

# ---------------------------------------------------------------- Task 2 (Dipin)
t2 = ROOT / "task2_sentiment" / "dipin"
m2 = load_module("t2_models", t2 / "src" / "models.py")
cfg2 = json.load(open(t2 / "configs" / "task2_config.json"))
vocab_size = len(json.load(open(t2 / "data_processed" / "vocab.json")))
ck2 = torch.load(t2 / "checkpoints" / "experimental_2.pt", map_location="cpu", weights_only=False)
net2 = m2.build_model("experimental_2", cfg2, vocab_size).eval()
net2.load_state_dict(ck2["state_dict"], strict=True)
ids = torch.randint(2, vocab_size, (4, 200))
with torch.no_grad():
    out = net2(ids, torch.tensor([200, 150, 80, 20]))
check("Task 2 BiLSTM+attention experimental_2.pt", out.shape == (4,) and torch.isfinite(out).all(),
      f"{m2.count_parameters(net2):,} parameters, logits {[round(float(v), 3) for v in out]}")

# ---------------------------------------------------------------- Task 3 (Dipin)
t3d = ROOT / "task3_gan" / "dipin"
m3d = load_module("t3d_models", t3d / "src" / "models.py")
cfg3d = json.load(open(t3d / "configs" / "task3_config_v2.json"))
g = m3d.build_generator(cfg3d).eval()
g.load_state_dict(torch.load(t3d / "checkpoints" / "v2" / "G_AB.pt", map_location="cpu", weights_only=True), strict=True)
with torch.no_grad():
    y = g(torch.rand(1, 3, 256, 256) * 2 - 1)
check("Task 3 (Dipin) final CycleGAN generator v2/G_AB.pt", y.shape == (1, 3, 256, 256) and torch.isfinite(y).all(),
      f"{m3d.count_params(g):,} parameters, output {tuple(y.shape)}")

# ---------------------------------------------------------------- Task 3 (Khushi)
t3k = ROOT / "task3_gan" / "khushi"
ck3 = t3k / "checkpoints" / "task3_khushi_prod_20260930_154246" / "epoch_28.pt"
if ck3.exists():
    m3k = load_module("t3k_models", t3k / "src" / "models.py")
    expected = "d3124ab24a6f2201a03673cde5a416d04c78dd44985a3c7d590c48efc7e89d3c"
    sha = sha256(ck3)
    state = torch.load(ck3, map_location="cpu", weights_only=False)
    nets = {n: (m3k.build_generator() if n.startswith("G") else m3k.build_discriminator()).eval() for n in ("G_A2B", "G_B2A", "D_A", "D_B")}
    total = 0
    for n, net in nets.items():
        net.load_state_dict(state["model_state_dicts"][n], strict=True)
        total += m3k.count_parameters(net)
    x = torch.rand(1, 3, 256, 256) * 2 - 1
    with torch.no_grad():
        a, b = nets["G_A2B"](x), nets["G_B2A"](x)
    ok = (sha == expected and total == 28285832 and state["global_step"] == 204102
          and a.shape == b.shape == (1, 3, 256, 256) and torch.isfinite(a).all() and torch.isfinite(b).all())
    check("Task 3 (Khushi) CycleGAN epoch_28.pt (completed epoch 29)", ok,
          f"SHA-256 match={sha == expected}, {total:,} parameters, global step {state['global_step']:,}")
else:
    print("[SKIP] Task 3 (Khushi): epoch_28.pt is not in this checkout (340 MB; see README for how it is distributed)")

print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED")
sys.exit(0 if all(results) else 1)

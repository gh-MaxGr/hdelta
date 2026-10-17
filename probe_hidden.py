"""Does the shared delusion separate in the REPRESENTATIONS before it collapses in the logits?

Output divergence is blind to the case where the reflex and the frozen reference give the
SAME WRONG answer - that is the measured blind spot, and per-query confidence does not cover
it either. This asks whether the information is present earlier: are the hidden states on
agree_wrong inputs distinguishable from agree_right inputs, even though the outputs are not?

  If they separate, there is a detection method output-space monitoring cannot reach.
  If they do not, the delusion is baked in before the residual stream reaches the head, and
  no representational probe will save a self-verifying system.

Scored honestly: a linear probe fitted on HALF the inputs and evaluated on the other half.
Fitting and scoring on the same activations finds a separating hyperplane in a 2048-dim space
almost regardless of whether real signal exists.

    python probe_hidden.py results/gate3_gated
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from hdelta.encoding import Encoder
from hdelta.modelwrap import groups_from_config, install_hdelta, load_model
from hdelta.tasks import make_task
from hdelta.toolcache import CompiledMonitor, generate_with_confidence

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate3_gated")
cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
task = make_task(cfg["task"])
rules = task.rule_names()

model, tok = load_model(cfg["model"]["name"], cfg["model"].get("dtype", "bfloat16"), "cuda")
ctrl, _ = install_hdelta(model, groups_from_config(cfg["plastic"]["groups"]),
                         max_fraction=cfg["plastic"].get("max_fraction", 0.05),
                         postsynaptic=cfg["plastic"].get("postsynaptic", "error"),
                         decay_enabled=True, homeo_enabled=True,
                         update_norm=cfg["plastic"].get("update_norm", "adam"))
snap = torch.load(run / "delta.pt", map_location="cuda")
ctrl.restore({k: v.to("cuda") for k, v in snap.items()})
ctrl.invalidate()
enc = Encoder(tok, "cuda")
mon = CompiledMonitor(task, enc, ctrl=ctrl, deliberate_frozen=True)
n_shots = int(cfg["stream"].get("n_shots", 2))
probe_n = int(cfg["stream"]["probe_n"])
seed = int(cfg.get("seed", 0))
print(f"restored dW from {run/'delta.pt'}", flush=True)


@torch.no_grad()
def last_hidden(prompts, chunk=8):
    """Hidden state at the final prompt position, every layer, WITH dW active."""
    out = []
    for i in range(0, len(prompts), chunk):
        ids, attn = enc.encode_prompts_left(prompts[i:i + chunk])
        hs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
        # (layers, batch, hidden) at the last position
        stack = torch.stack([h[:, -1, :].float() for h in hs], dim=0)
        out.append(stack.cpu())
    return torch.cat(out, dim=1)      # (layers, N, hidden)


labels, feats = [], []
for i, r in enumerate(rules):
    rng = random.Random(10_000 + 17 * i + seed)
    pz = [task.sample(r, rng) for _ in range(probe_n)]
    shots = [task.sample(r, rng) for _ in range(n_shots)]
    prompts = [task.make_pair(p, "direct")[0] for p in pz]
    compiled = [a.text for a in generate_with_confidence(model, enc, prompts, 16)]
    delib = mon._deliberate_pass(model, pz, shots)
    h = last_hidden(prompts)
    feats.append(h)
    for p, c, d in zip(pz, compiled, delib):
        if c == d:
            labels.append("agree_right" if c == p.answer else "agree_wrong")
        else:
            labels.append("disagree")
    print(f"  rule {r}: done", flush=True)

H = torch.cat(feats, dim=1).numpy()          # (layers, N, hidden)
labels = np.array(labels)
n_layers = H.shape[0]
print(f"\n{len(labels)} inputs: " + "  ".join(
    f"{k}={int((labels == k).sum())}" for k in ("agree_right", "agree_wrong", "disagree")))

mask = np.isin(labels, ["agree_right", "agree_wrong"])
y = (labels[mask] == "agree_wrong").astype(int)
if y.sum() < 10 or (1 - y).sum() < 10:
    print("too few of one class to probe")
    raise SystemExit

rs = np.random.RandomState(0)
idx = rs.permutation(len(y))
half = len(idx) // 2
fit, test = idx[:half], idx[half:]
base = max(y[test].mean(), 1 - y[test].mean())
print(f"majority-class baseline on the held-out half: {base:.3f}\n")
print(f"{'layer':>6s}{'held-out acc':>14s}{'vs baseline':>13s}")

best = (0, -1.0)
for L in range(n_layers):
    X = H[L][mask]
    X = (X - X[fit].mean(0)) / (X[fit].std(0) + 1e-6)
    # ridge-regularised least squares, closed form; no sklearn in this venv
    Xf = np.hstack([X[fit], np.ones((len(fit), 1))])
    w = np.linalg.solve(Xf.T @ Xf + 50.0 * np.eye(Xf.shape[1]), Xf.T @ (2 * y[fit] - 1))
    Xt = np.hstack([X[test], np.ones((len(test), 1))])
    acc = float((((Xt @ w) > 0).astype(int) == y[test]).mean())
    if L % max(1, n_layers // 12) == 0 or acc > best[1]:
        print(f"{L:6d}{acc:14.3f}{acc - base:+13.3f}")
    if acc > best[1]:
        best = (L, acc)

print(f"\nbest layer {best[0]}: {best[1]:.3f} against a {base:.3f} baseline "
      f"({best[1] - base:+.3f})")
print("\n>0.05 above baseline on held-out data means the representations carry a signal the")
print("outputs do not - an audit hook. At baseline, the delusion is representational and no")
print("probe on this model's own activations will detect it.")

"""Where in the frozen network is a conjunction linearly available?

The sparse-expansion hypothesis: middle-layer MLPs construct conjunctive features that later
layers read linearly. If so, the plastic slice - currently the TOP 9 layers - is being asked
to build a conjunction at a depth where the network expects one to already exist, and that
would explain why `conjunction` refuses to compile (+0.056) while `map` compiles perfectly
(+1.000).

Before spending a plasticity run on a guessed depth, measure the premise directly on the
FROZEN model: at each layer, how linearly decodable from the residual stream is
  * the answer under `map`        - a function of ONE feature (cluster)
  * the answer under `conjunction`- a function of the (zone, cluster) PAIR
from the same inputs, with no rule in context.

Probe fitted and scored on disjoint halves, ridge-regularised, one-vs-rest over 6 labels.
A conjunction curve that rises in the middle and falls by the top is the signature the
hypothesis predicts, and it names the layers to plasticise instead of guessing them.

    python probe_decodable.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from hdelta.encoding import Encoder
from hdelta.modelwrap import load_model
from hdelta.tasks import LogicTask

N = 480
MODEL = "Qwen/Qwen2.5-3B-Instruct"

task = LogicTask({f: {"family": f, "seed": 11} for f in ("map", "conjunction", "map_wide")})
model, tok = load_model(MODEL, "bfloat16", "cuda")
enc = Encoder(tok, "cuda")

rng = random.Random(4242)
inputs = [task._draw(rng) for _ in range(N)]


@torch.no_grad()
def hiddens(prompts, chunk=8):
    out = []
    for i in range(0, len(prompts), chunk):
        ids, attn = enc.encode_prompts_left(prompts[i:i + chunk])
        hs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1, :].float() for h in hs], 0).cpu())
    return torch.cat(out, 1)


# identical prompts for both targets: only the LABELLING differs, so any difference in
# decodability is about the target function, not the input or the prompt
prompts = [f"{task.header}{task.input_label}: {x}\n{task.answer_label}:" for x in inputs]
H = hiddens(prompts).numpy()
n_layers = H.shape[0]
print(f"{N} inputs, {n_layers} layers\n", flush=True)

targets = {fam: np.array([task.answer_under(fam, x) for x in inputs])
           for fam in ("map", "conjunction")}
for fam, y in targets.items():
    labs, counts = np.unique(y, return_counts=True)
    print(f"  {fam}: {len(labs)} labels, majority {counts.max()/N:.3f}")

idx = np.random.RandomState(0).permutation(N)
fit, test = idx[:N // 2], idx[N // 2:]


def decode(X, y):
    """One-vs-rest ridge; returns held-out accuracy."""
    labs = sorted(set(y))
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    Xf = np.hstack([(X[fit] - mu) / sd, np.ones((len(fit), 1))])
    Xt = np.hstack([(X[test] - mu) / sd, np.ones((len(test), 1))])
    G = Xf.T @ Xf + 100.0 * np.eye(Xf.shape[1])
    scores = np.stack([Xt @ np.linalg.solve(G, Xf.T @ (2 * (y[fit] == L) - 1)) for L in labs], 1)
    return float((np.array(labs)[scores.argmax(1)] == y[test]).mean())


print(f"\n{'layer':>6s}{'map':>9s}{'conjunction':>14s}{'gap':>8s}")
rows = []
for L in range(n_layers):
    a = decode(H[L], targets["map"])
    b = decode(H[L], targets["conjunction"])
    rows.append({"layer": L, "map": a, "conjunction": b})
    if L % 2 == 0 or L == n_layers - 1:
        print(f"{L:6d}{a:9.3f}{b:14.3f}{a - b:+8.3f}")

best_c = max(rows, key=lambda r: r["conjunction"])
best_m = max(rows, key=lambda r: r["map"])
top9 = [r for r in rows if r["layer"] >= n_layers - 9]
print(f"\n  map          peaks at layer {best_m['layer']} ({best_m['map']:.3f})")
print(f"  conjunction  peaks at layer {best_c['layer']} ({best_c['conjunction']:.3f})")
print(f"  conjunction over the CURRENT plastic slice (top 9, layers "
      f"{n_layers-9}-{n_layers-1}): mean {np.mean([r['conjunction'] for r in top9]):.3f}, "
      f"max {max(r['conjunction'] for r in top9):.3f}")
print("\n  If conjunction peaks in the middle and is lower across the top 9, the slice is")
print("  positioned where the conjunction is no longer linearly available - and the layers")
print("  around its peak are the ones to plasticise. If it is flat or peaks at the top, the")
print("  sparse-expansion story does not explain the writability gap.")
Path("results/families").mkdir(parents=True, exist_ok=True)
Path("results/families/decodable.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

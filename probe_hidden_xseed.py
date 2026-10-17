"""Does the shared-delusion signal in the hidden states TRANSFER across independent runs?

The within-run probe reached 0.986 against a 0.877 baseline - but on 73 held-out points with
9 in the minority class, and a linear probe in 2048 dimensions will separate almost anything
at that size. Two upgrades decide whether there is a method here:

  more inputs   192 -> up to 4 x `n_probe` per seed, so the minority class is in the hundreds
  CROSS-SEED    fit on one seed's activations, test on a DIFFERENT seed's dW

The cross-seed split is the real test. Within a run, a probe can latch onto the particular
fast-weight state. If the same hyperplane flags delusion in a `dW` it has never seen, the
signal is a property of the phenomenon rather than of the run.

Honest framing for deployment: fitting the probe needs LABELS, and labels need the tool. So
this is not ground-truth-free the way the divergence monitor claimed to be - it is
ground-truth-CHEAP: pay N tool calls once to calibrate, then detect for free. That is still
worth having, but it is a different claim.

    python probe_hidden_xseed.py
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

RUNS = ["results/gate3_gated", "results/gate3_gated_s1", "results/gate3_gated_s2"]
N_PER_RULE = 160          # was 48; the minority class was the binding constraint
LAYERS_TRIED = None       # filled from the model


def collect(run: str):
    """Hidden states + delusion labels for one run's compiled state."""
    cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    task = make_task(cfg["task"])
    rules = task.rule_names()
    seed = int(cfg.get("seed", 0))

    model, tok = load_model(cfg["model"]["name"], cfg["model"].get("dtype", "bfloat16"), "cuda")
    ctrl, _ = install_hdelta(model, groups_from_config(cfg["plastic"]["groups"]),
                             max_fraction=cfg["plastic"].get("max_fraction", 0.05),
                             postsynaptic=cfg["plastic"].get("postsynaptic", "error"),
                             decay_enabled=True, homeo_enabled=True,
                             update_norm=cfg["plastic"].get("update_norm", "adam"))
    snap = torch.load(Path(run) / "delta.pt", map_location="cuda")
    ctrl.restore({k: v.to("cuda") for k, v in snap.items()})
    ctrl.invalidate()
    enc = Encoder(tok, "cuda")
    mon = CompiledMonitor(task, enc, ctrl=ctrl, deliberate_frozen=True)
    n_shots = int(cfg["stream"].get("n_shots", 2))

    @torch.no_grad()
    def last_hidden(prompts, chunk=8):
        out = []
        for i in range(0, len(prompts), chunk):
            ids, attn = enc.encode_prompts_left(prompts[i:i + chunk])
            hs = model(input_ids=ids, attention_mask=attn,
                       output_hidden_states=True).hidden_states
            out.append(torch.stack([h[:, -1, :].float() for h in hs], 0).cpu())
        return torch.cat(out, 1)

    feats, labels = [], []
    for i, r in enumerate(rules):
        rng = random.Random(55_000 + 17 * i + seed)     # NOT the training probe set
        pz = [task.sample(r, rng) for _ in range(N_PER_RULE)]
        shots = [task.sample(r, rng) for _ in range(n_shots)]
        prompts = [task.make_pair(p, "direct")[0] for p in pz]
        compiled = [a.text for a in generate_with_confidence(model, enc, prompts, 16)]
        delib = mon._deliberate_pass(model, pz, shots)
        feats.append(last_hidden(prompts))
        for p, c, d in zip(pz, compiled, delib):
            labels.append(("agree_wrong" if c != p.answer else "agree_right")
                          if c == d else "disagree")
        print(f"    rule {r} done", flush=True)

    H = torch.cat(feats, 1).numpy()
    del model, ctrl
    torch.cuda.empty_cache()
    return H, np.array(labels)


def fit_probe(Xf, yf, alpha=50.0):
    A = np.hstack([Xf, np.ones((len(Xf), 1))])
    return np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (2 * yf - 1))


def score(w, Xt, yt):
    A = np.hstack([Xt, np.ones((len(Xt), 1))])
    pred = ((A @ w) > 0).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return float((pred == yt).mean()), prec, rec, tp, fp, fn


def main():
    data = {}
    for run in RUNS:
        if not (Path(run) / "delta.pt").exists():
            print(f"skip {run} (no delta.pt)")
            continue
        print(f"collecting {run} ...", flush=True)
        data[run] = collect(run)
        H, lab = data[run]
        print(f"  {len(lab)} inputs: " + "  ".join(
            f"{k}={int((lab == k).sum())}" for k in ("agree_right", "agree_wrong", "disagree")),
            flush=True)

    runs = list(data)
    n_layers = data[runs[0]][0].shape[0]

    def xy(run, L):
        H, lab = data[run]
        m = np.isin(lab, ["agree_right", "agree_wrong"])
        return H[L][m], (lab[m] == "agree_wrong").astype(int)

    print(f"\n=== WITHIN-run (fit and test on disjoint halves of the same run) ===")
    print(f"{'run':>24s}{'layer':>7s}{'acc':>7s}{'base':>7s}{'prec':>7s}{'rec':>7s}")
    within = {}
    for run in runs:
        best = (None, -1)
        for L in range(n_layers):
            X, y = xy(run, L)
            idx = np.random.RandomState(0).permutation(len(y))
            h = len(idx) // 2
            f, t = idx[:h], idx[h:]
            mu, sd = X[f].mean(0), X[f].std(0) + 1e-6
            w = fit_probe((X[f] - mu) / sd, y[f])
            acc, prec, rec, *_ = score(w, (X[t] - mu) / sd, y[t])
            if acc > best[1]:
                best = ((L, acc, prec, rec, max(y[t].mean(), 1 - y[t].mean())), acc)
        L, acc, prec, rec, base = best[0]
        within[run] = best[0]
        print(f"{Path(run).name:>24s}{L:7d}{acc:7.3f}{base:7.3f}{prec:7.3f}{rec:7.3f}")

    if len(runs) < 2:
        print("\nneed >=2 runs with delta.pt for the cross-seed test")
        return
    print(f"\n=== CROSS-SEED (fit on one run's dW, test on another's) ===")
    print(f"{'fit -> test':>34s}{'layer':>7s}{'acc':>7s}{'base':>7s}{'prec':>7s}{'rec':>7s}")
    for a in runs:
        for b in runs:
            if a == b:
                continue
            L = within[a][0]                      # the layer chosen on the FIT run only
            Xa, ya = xy(a, L)
            Xb, yb = xy(b, L)
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
            w = fit_probe((Xa - mu) / sd, ya)
            acc, prec, rec, tp, fp, fn = score(w, (Xb - mu) / sd, yb)
            base = max(yb.mean(), 1 - yb.mean())
            print(f"{Path(a).name[-2:]:>16s} -> {Path(b).name[-2:]:<14s}{L:7d}{acc:7.3f}"
                  f"{base:7.3f}{prec:7.3f}{rec:7.3f}   tp={tp} fp={fp} fn={fn}")

    print("\nCross-seed accuracy near the baseline with low recall = the within-run probe was")
    print("reading that run's particular dW, not the delusion. Recall well above zero with")
    print("precision above the class prior = a transferable audit hook.")
    Path("results/families").mkdir(parents=True, exist_ok=True)
    Path("results/hidden_xseed.json").write_text(json.dumps(
        {Path(k).name: {"within_layer": int(v[0]), "within_acc": float(v[1])}
         for k, v in within.items()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Is Part E's delusion probe reading the MODEL'S STATE, or only the INPUT'S STRUCTURE?

Part E fitted a linear probe on the final-prompt hidden state to flag shared delusion (reflex
and deliberate reference agree on the SAME WRONG answer), fitted on one seed, tested on
another: 0.957-0.978 against baselines of 0.890-0.949. It had no control for this: on
workspace every blind-spot error sat on an EXCEPTION input, and whether an input is an
exception is written in the prompt (`_test.py`). A probe can score well by recognising test
files, and that would transfer across seeds just as well, since every seed has the same
exceptions. Conjunction's version of the probe, split so it could not recognise inputs, fell
to chance (AUC 0.28-0.76).

Two labels, both scored CROSS-SEED (fit on one run, test on another), as Part E did:

    L1 shared delusion   agree_wrong vs agree_right, among inputs where reflex == reference
                         (Part E's label; reference = frozen deliberate path)
    L2 confident-wrong   instinct wrong vs right, among inputs with p_min >= 0.9 - the answers
                         a confidence gate would pass, which is where the pipeline is blind

Against each, three predictors:
    structure   "positive iff the input is an exception" - no fitting, no model
    p_min       the instinct's own confidence (L1 only; L2 is already confident)
    probe       ridge on one layer's hidden state, layer chosen on the fit run alone

And the decisive control: the same probe fitted and scored WITHIN exception inputs only,
where structure is constant and carries no information. 100 extra exception inputs per rule
are drawn so that subset has power; the pooled rows use the natural draw only.

Decision criterion, stated before the run: the probe earns a place in the pipeline only if,
within exceptions, its cross-seed AUC is >= 0.7 on both labels where both classes have >= 10
members. Otherwise it is an exception detector, and keyed memory already does that job
without labels.

Changes from Part E, both deliberate: hidden states are taken one prompt at a time (a batched
forward without position ids makes a left-padded state depend on its batch-mates), and the
reference gets 160 new tokens instead of the monitor's default 64 (workspace traces need up to
66, so some Part E references were cut off and fell into 'disagree').

    .venv/Scripts/python.exe probe_delusion_ctrl.py
    .venv/Scripts/python.exe probe_delusion_ctrl.py --device cpu --nat 2 --exc 1 --delib-tokens 16   # smoke
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from hdelta.encoding import Encoder
from hdelta.metrics import generate_text
from hdelta.toolcache import generate_with_confidence
from idiom import load_compiled
from predict_failure import auc

RUNS = ["results/gate3_gated", "results/gate3_gated_s1", "results/gate3_gated_s2"]
CONF = 0.9
MIN_CLASS = 10


def collect(run, device, n_nat, n_exc, delib_tokens):
    cfg, task, model, tok, ctrl, norm = load_compiled(Path(run), device)
    enc = Encoder(tok, device)
    seed = int(cfg.get("seed", 0))
    n_shots = int(cfg["stream"].get("n_shots", 2))
    rows, H = [], []
    for i, r in enumerate(task.rule_names()):
        rng = random.Random(55_000 + 17 * i + seed)          # Part E's draw
        pz = [task.sample(r, rng) for _ in range(n_nat)]
        shots = [task.sample(r, rng) for _ in range(n_shots)]
        xrng = random.Random(77_000 + 17 * i + seed)
        extra = []
        while len(extra) < n_exc:
            p = task.sample(r, xrng)
            if task.is_exception(r, p.input_str):
                extra.append(p)
        allp = [(p, True) for p in pz] + [(p, False) for p in extra]
        prompts = [task.make_pair(p, "direct")[0] for p, _ in allp]
        inst = [generate_with_confidence(model, enc, [q], 16, chunk=1)[0] for q in prompts]
        dprompts = [task.make_pair(p, "free", shots=shots, rule_text=True)[0] for p, _ in allp]
        ctrl.set_apply_delta(False)                           # frozen reference, as Part E
        try:
            dtexts = generate_text(model, enc, dprompts, delib_tokens)
        finally:
            ctrl.set_apply_delta(True)
        with torch.no_grad():
            for q in prompts:                                  # one at a time: no padding
                ids, attn = enc.encode_prompts_left([q])
                hs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
                H.append(torch.stack([h[0, -1, :] for h in hs], 0).to(torch.float16).cpu())
        for (p, nat), a, t in zip(allp, inst, dtexts):
            d = task.parse_generation(t)[0]
            rows.append({"rule": r, "input": p.input_str, "natural": nat,
                         "exc": task.is_exception(r, p.input_str), "truth": p.answer,
                         "inst": a.text, "p_min": a.p_min, "delib": d,
                         "inst_wrong": a.text != p.answer, "agree": a.text == d})
        print(f"    {run} rule {r}: {len(allp)} inputs", flush=True)
    del model, ctrl
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows, torch.stack(H, 1).numpy()          # H: (layers, N, hidden)


def label_mask(rows, label, subset):
    """(mask over rows, y) for a label restricted to a subset."""
    keep, y = [], []
    for r in rows:
        in_sub = {"natural": r["natural"], "exc": r["exc"], "ord": (not r["exc"]) and r["natural"]}[subset]
        if not in_sub:
            keep.append(False); y.append(0); continue
        if label == "L1":
            ok = r["agree"]
            y.append(int(r["inst_wrong"]))
        else:
            ok = r["p_min"] >= CONF
            y.append(int(r["inst_wrong"]))
        keep.append(ok)
    m = np.array(keep)
    return m, np.array(y)[m]


def ridge(Xf, yf, alpha=50.0):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A = np.hstack([(Xf - mu) / sd, np.ones((len(Xf), 1))])
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (2.0 * yf - 1))
    return lambda X: np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w


def pick_layer(H, y):
    """Best layer by within-run AUC on a seeded half-split of the FIT run only."""
    idx = np.random.RandomState(0).permutation(len(y))
    f, t = idx[: len(idx) // 2], idx[len(idx) // 2:]
    if y[f].sum() < 2 or (1 - y[f]).sum() < 2 or y[t].sum() < 1 or (1 - y[t]).sum() < 1:
        return None
    best = max(range(H.shape[0]), key=lambda L: auc(ridge(H[L][f].astype(np.float32), y[f])(
        H[L][t].astype(np.float32)), y[t]))
    return best


def prf(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return (float((pred == y).mean()), tp / (tp + fp) if tp + fp else float("nan"),
            tp / (tp + fn) if tp + fn else float("nan"))


def analyse(data, min_class=MIN_CLASS):
    runs = list(data)
    out, layer_of = [], {}
    for label in ("L1", "L2"):
        for subset in ("natural", "exc", "ord"):
            for a in runs:
                for b in runs:
                    if a == b:
                        continue
                    ra, Ha = data[a]
                    rb, Hb = data[b]
                    ma, ya = label_mask(ra, label, subset)
                    mb, yb = label_mask(rb, label, subset)
                    row = {"label": label, "subset": subset, "fit": Path(a).name, "test": Path(b).name,
                           "n_test": int(len(yb)), "pos_test": int(yb.sum()),
                           "base": float(max(yb.mean(), 1 - yb.mean())) if len(yb) else float("nan")}
                    powered = min(ya.sum(), len(ya) - ya.sum()) >= MIN_CLASS and \
                        min(yb.sum(), len(yb) - yb.sum()) >= MIN_CLASS
                    row["powered"] = bool(powered)
                    exc_b = np.array([r["exc"] for r, k in zip(rb, mb) if k], dtype=int)
                    row["structure"] = prf(exc_b, yb) if len(yb) else None
                    pm = np.array([r["p_min"] for r, k in zip(rb, mb) if k])
                    row["p_min_auc"] = auc(1 - pm, yb) if len(yb) else float("nan")
                    if powered:
                        if (a, label, subset) not in layer_of:     # chosen on the fit run only
                            layer_of[(a, label, subset)] = pick_layer(Ha[:, ma], ya)
                        L = layer_of[(a, label, subset)]
                        if L is not None:
                            s = ridge(Ha[L][ma].astype(np.float32), ya)(Hb[L][mb].astype(np.float32))
                            row.update(layer=int(L), probe_auc=auc(s, yb), probe=prf((s > 0).astype(int), yb))
                    out.append(row)
    return out


def show(data, res):
    for run, (rows, _) in data.items():
        nat = [r for r in rows if r["natural"]]
        exc = [r for r in rows if r["exc"]]
        f = lambda rs, k: sum(r[k] for r in rs)
        print(f"\n{Path(run).name}: natural {len(nat)} ({f(nat, 'exc')} exceptions) + extra exceptions; "
              f"instinct wrong on {f([r for r in nat if not r['exc']], 'inst_wrong')} ordinary and "
              f"{f(exc, 'inst_wrong')}/{len(exc)} exception inputs; "
              f"shared delusion {sum(r['agree'] and r['inst_wrong'] for r in nat)} natural; "
              f"confident-wrong {sum(r['p_min'] >= CONF and r['inst_wrong'] for r in nat)} natural "
              f"({sum(r['p_min'] >= CONF and r['inst_wrong'] and r['exc'] for r in nat)} on exceptions)")
    print(f"\n{'label':<6}{'subset':<9}{'fit->test':<22}{'n/pos':>9}{'base':>7}"
          f"{'struct acc/prec/rec':>22}{'p_min AUC':>10}{'probe AUC':>10}{'probe acc':>10}{'layer':>6}")
    for r in res:
        st_ = r["structure"]
        s_txt = "-" if st_ is None else f"{st_[0]:.3f}/{st_[1]:.2f}/{st_[2]:.2f}"
        p_auc = f"{r['probe_auc']:.3f}" if "probe_auc" in r else ("underpow." if not r["powered"] else "-")
        p_acc = f"{r['probe'][0]:.3f}" if "probe" in r else "-"
        print(f"{r['label']:<6}{r['subset']:<9}{r['fit'][-2:] + '->' + r['test'][-2:]:<22}"
              f"{str(r['n_test']) + '/' + str(r['pos_test']):>9}{r['base']:>7.3f}{s_txt:>22}"
              f"{r['p_min_auc']:>10.3f}{p_auc:>10}{p_acc:>10}{r.get('layer', '-'):>6}")
    print("\nDecision: the probe earns a place only if its WITHIN-EXCEPTION ('exc') cross-seed AUC is")
    print(">= 0.7 on both labels wherever powered. Structure = 'predict wrong iff exception input'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nat", type=int, default=160)
    ap.add_argument("--exc", type=int, default=100)
    ap.add_argument("--delib-tokens", type=int, default=160)
    ap.add_argument("--min-class", type=int, default=MIN_CLASS)
    ap.add_argument("--out", default="results/probe_delusion_ctrl.json")
    a = ap.parse_args()
    data = {}
    cache = Path(a.out).with_suffix("")          # collected states survive an analysis crash
    for run in (a.runs or RUNS):
        f_rows, f_h = Path(f"{cache}_{Path(run).name}_rows.json"), Path(f"{cache}_{Path(run).name}_H.npy")
        if f_rows.exists() and f_h.exists():
            print(f"loading collected {run}", flush=True)
            data[run] = (json.load(open(f_rows, encoding="utf-8")), np.load(f_h))
            continue
        print(f"collecting {run} ...", flush=True)
        data[run] = collect(run, a.device, a.nat, a.exc, a.delib_tokens)
        json.dump(data[run][0], open(f_rows, "w", encoding="utf-8"))
        np.save(f_h, data[run][1])
    res = analyse(data, a.min_class)
    json.dump({"rows": {k: v[0] for k, v in data.items()}, "results": res},
              open(a.out, "w", encoding="utf-8"), indent=1)
    show(data, res)

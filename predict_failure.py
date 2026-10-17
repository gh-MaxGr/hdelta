"""Can the instinct tell, BEFORE it answers, that it is about to be wrong - and does using that
make decisions better?

The instinct is the compiled reflex: bare task in, answer out, no reasoning. The question has
three parts, measured on the conjunction runs so the pain-free baseline and the pain arm are
scored identically:

  1. PREDICT   Is the instinct's own failure readable from it, pre-answer?
               - confidence: the probability it assigns to the answer it gives. No fitting.
               - hidden-state probe: ridge on the final-layer state at the last prompt token,
                 fitted on half the CIRCUMSTANCES and scored on the other half.
  2. HOW MUCH  How much of the learned delta is needed before failure becomes readable? The
               delta is scaled by alpha in [0, 1] - 0 is the untrained model, 1 the trained
               instinct - and predictability is measured at each step. Not the same as
               snapshots taken during training, but it asks how much stored change it takes.
  3. DECIDE    Does acting on the warning improve decisions? The instinct answers unless its
               confidence is low; then the deliberate path (rule stated, reasoning written out)
               answers instead. Accuracy is reported against the fraction deferred.

Two things about this task shape the method, and both matter:

  - Conjunction has only 192 distinct inputs and the answer depends on zone+cluster alone -
    24 real CIRCUMSTANCES, with the item field irrelevant. A probe split by input would fit on
    one item of a circumstance and be tested on another item of the same one, and "predict
    failure" by remembering which circumstances fail. So the probe is split by circumstance.
  - Late in training the instinct is nearly perfect here, so there are few failures to predict
    and any AUC is weakly powered. The failure count is printed beside every AUC for that
    reason, and the alpha sweep exists partly to put the instinct where it still fails.

All six labels tokenize as [' #', 'q', digit, '\\n'] and differ only in the digit, so the
model's full belief over the six answers is exactly a softmax over six digit logits at one
position - one forward pass per input, no sampling.

    .venv/Scripts/python.exe predict_failure.py                 # all 7 conjunction runs
    .venv/Scripts/python.exe predict_failure.py --device cpu --limit 6 RUN   # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

import numpy as np
import torch

from hdelta.encoding import Encoder
from hdelta.metrics import generate, generate_text
from idiom import load_compiled

RULE = "conjunction"
BASELINE = ["results/ladder_long/conjunction", "results/locus/conjunction_attn_long",
            "results/variance/conjunction_s1", "results/variance/conjunction_s2"]
PAIN = ["results/pain/conjunction_s0", "results/pain/conjunction_s1",
        "results/pain/conjunction_s2"]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFER = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
LABEL_RE = re.compile(r"#q[1-6]")


# ---------------------------------------------------------------- the input domain
def domain(task):
    rng = random.Random(0)
    return sorted({task.sample_input(RULE, rng) for _ in range(5000)})


def circumstance(x):
    """zone+cluster: the part of the input the answer depends on (the item never does)."""
    return tuple(x.split("/")[:2])


# ---------------------------------------------------------------- reading the model
class Reader:
    def __init__(self, model, enc, device):
        self.model, self.enc, self.dev = model, enc, device
        tok = enc.tok
        rows = [tok(f" #q{i}\n", add_special_tokens=False)["input_ids"] for i in range(1, 7)]
        self.prefix = rows[0][:-2]
        assert all(r[:-2] == self.prefix and len(r) == len(rows[0]) for r in rows), \
            "labels no longer share a prefix - the one-position readout is invalid"
        self.digits = [r[-2] for r in rows]

    def _batch(self, seqs):
        pad, T = self.enc.pad, max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, T - len(s):] = torch.tensor(s)
            attn[i, T - len(s):] = 1
        pos = (attn.cumsum(-1) - 1).clamp_min(0)
        return ids.to(self.dev), attn.to(self.dev), pos.to(self.dev)

    def _seq(self, prompt):
        pre = [self.enc.bos] if self.enc.bos is not None else []
        return pre + self.enc.ids(prompt)

    @torch.no_grad()
    def belief(self, prompts, chunk=32):
        """(N, 6) probability over the six labels, read at the digit position."""
        out = []
        for i in range(0, len(prompts), chunk):
            ids, attn, pos = self._batch([self._seq(p) + self.prefix for p in prompts[i:i + chunk]])
            lg = self.model(input_ids=ids, attention_mask=attn, position_ids=pos,
                            logits_to_keep=1).logits[:, -1, :]
            out.append(torch.softmax(lg[:, self.digits].float(), -1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def hidden(self, prompts, chunk=16):
        """(N, hidden) final-layer state at the last prompt token, BEFORE any answer."""
        out = []
        for i in range(0, len(prompts), chunk):
            ids, attn, pos = self._batch([self._seq(p) for p in prompts[i:i + chunk]])
            hs = self.model(input_ids=ids, attention_mask=attn, position_ids=pos,
                            output_hidden_states=True).hidden_states
            out.append(hs[-1][:, -1, :].float().cpu())
        return torch.cat(out).numpy()


def parse_label(text):
    m = LABEL_RE.search(text or "")
    return m.group(0) if m else ""


# ---------------------------------------------------------------- statistics, numpy only
def auc(score, y):
    """P(a random failure outscores a random success); ties count half. nan if one class."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def ridge_scores(Xf, yf, Xt, lam=10.0):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A = np.hstack([(Xf - mu) / sd, np.ones((len(Xf), 1))])
    B = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yf)
    return B @ w


def probe_auc(H, fail, circs, seed=0):
    """Split BY CIRCUMSTANCE, fit on one half, score the other, then swap. nan if underpowered."""
    keys = sorted(set(circs))
    random.Random(seed).shuffle(keys)
    halves = [set(keys[: len(keys) // 2]), set(keys[len(keys) // 2:])]
    scores = np.full(len(fail), np.nan)
    for a, b in ((0, 1), (1, 0)):
        fi = np.array([c in halves[a] for c in circs])
        ti = np.array([c in halves[b] for c in circs])
        if fail[fi].sum() < 2 or (1 - fail[fi]).sum() < 2:
            return float("nan")
        scores[ti] = ridge_scores(H[fi], fail[fi].astype(float), H[ti])
    return auc(scores, fail)


# ---------------------------------------------------------------- one run
def analyse(run, device="cuda", limit=None, max_new_delib=128):
    run = Path(run)
    cfg, task, model, tok, ctrl, norm = load_compiled(run, device)
    enc = Encoder(tok, device)
    rd = Reader(model, enc, device)
    xs = domain(task)
    if limit:
        xs = xs[:: max(1, len(xs) // limit)][:limit]
    truth = [task.answer_under(RULE, x) for x in xs]
    circs = [circumstance(x) for x in xs]
    from hdelta.tasks.base import Puzzle
    pz = [Puzzle(x, [], t, RULE, {}) for x, t in zip(xs, truth)]
    direct = [task.make_pair(p, "direct")[0] for p in pz]
    idx = {l: i for i, l in enumerate(f"#q{i}" for i in range(1, 7))}
    base = ctrl.snapshot()

    def instinct_at(alpha):
        ctrl.restore({k: alpha * v for k, v in base.items()})
        ans = [parse_label(g) for g in generate(model, enc, direct, 8)]
        bel = rd.belief(direct)
        conf = np.array([bel[i, idx[a]] if a in idx else 0.0 for i, a in enumerate(ans)])
        fail = np.array([int(a != t) for a, t in zip(ans, truth)])
        return ans, bel, conf, fail

    out = {"run": str(run), "seed": cfg.get("seed"),
           "pain_lambda": cfg["stream"].get("pain_lambda", 0.0), "n": len(xs),
           "dW_norm": norm, "alpha": {}}

    # 2. how much delta before failure is readable
    for a in ALPHAS:
        ans, bel, conf, fail = instinct_at(a)
        out["alpha"][str(a)] = {"acc": float(1 - fail.mean()), "n_fail": int(fail.sum()),
                                "conf_auc": auc(1 - conf, fail)}

    # 1. predict, at the trained weights
    ans, bel, conf, fail = instinct_at(1.0)
    H = rd.hidden(direct)
    argmax = [f"#q{int(b.argmax()) + 1}" for b in bel]
    out.update(acc=float(1 - fail.mean()), n_fail=int(fail.sum()),
               conf_auc=auc(1 - conf, fail), probe_auc=probe_auc(H, fail, circs),
               gen_vs_argmax_agree=float(np.mean([a == m for a, m in zip(ans, argmax)])),
               mean_conf_right=float(conf[fail == 0].mean()) if (fail == 0).any() else None,
               mean_conf_wrong=float(conf[fail == 1].mean()) if (fail == 1).any() else None)

    # 3. decide: defer the least-confident fraction to deliberation
    shots_rng = random.Random(7)
    shots = [task.sample(RULE, shots_rng) for _ in range(2)]
    dprompts = [task.make_pair(p, "free", shots=shots, rule_text=True)[0] for p in pz]
    dtexts = generate_text(model, enc, dprompts, max_new_delib)
    delib = [parse_label(task.parse_generation(t)[0]) for t in dtexts]
    dcorrect = np.array([int(d == t) for d, t in zip(delib, truth)])
    order = np.argsort(conf)                        # least confident first
    gate = {}
    for f in DEFER:
        k = int(round(f * len(xs)))
        use_delib = np.zeros(len(xs), bool)
        use_delib[order[:k]] = True
        correct = np.where(use_delib, dcorrect, 1 - fail)
        gate[str(f)] = float(correct.mean())
    oracle = np.where(fail == 1, dcorrect, 1).mean()
    out.update(delib_acc=float(dcorrect.mean()), gate=gate, oracle_gate=float(oracle),
               delib_trace_example=dtexts[0][:120])
    ctrl.restore(base)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def show(r):
    print(f"\n=== {r['run']}  pain_lambda={r['pain_lambda']}  seed={r['seed']}  n={r['n']} ===")
    print(f"  instinct acc {r['acc']:.3f}  ({r['n_fail']} failures)   deliberate acc {r['delib_acc']:.3f}"
          f"   generated==argmax {r['gen_vs_argmax_agree']:.2f}")
    print(f"  PREDICT  confidence AUC {r['conf_auc']:.3f}   hidden-probe AUC {r['probe_auc']:.3f}"
          f"   (0.5 = no signal; weak with few failures)")
    if r["mean_conf_wrong"] is not None:
        print(f"           mean confidence when right {r['mean_conf_right']:.3f} / when wrong {r['mean_conf_wrong']:.3f}")
    print("  HOW MUCH  alpha:  " + "  ".join(
        f"{a}->acc {v['acc']:.2f} auc {v['conf_auc']:.2f} ({v['n_fail']}f)" for a, v in r["alpha"].items()))
    print("  DECIDE   defer least-confident fraction to deliberation:  " + "  ".join(
        f"{f}:{v:.3f}" for f, v in r["gate"].items()) + f"   oracle {r['oracle_gate']:.3f}")


def summary(rows):
    print("\n" + "=" * 78)
    for name, lam in (("BASELINE (no pain)", 0.0), ("PAIN (lambda 1.0)", 1.0)):
        g = [r for r in rows if float(r["pain_lambda"]) == lam]
        if not g:
            continue
        m = lambda k: st.mean(r[k] for r in g if r[k] == r[k])
        print(f"{name:<20} n={len(g)}  instinct {m('acc'):.3f}  deliberate {m('delib_acc'):.3f}  "
              f"conf-AUC {m('conf_auc'):.3f}  best gate "
              f"{st.mean(max(r['gate'].values()) for r in g):.3f}")
    print("\nA gate only helps if the instinct is WRONG where it is UNSURE (AUC well above 0.5)")
    print("AND the deliberate path is right there. Compare 'best gate' with 'instinct'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/predict_failure.json")
    a = ap.parse_args()
    rows = []
    for run in (a.runs or BASELINE + PAIN):
        if not (Path(run) / "delta.pt").exists():
            print(f"\n=== {run}: no delta.pt, skipped ===")
            continue
        r = analyse(run, a.device, a.limit)
        show(r)
        rows.append(r)
        json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)
    summary(rows)

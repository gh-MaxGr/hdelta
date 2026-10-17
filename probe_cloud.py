"""Teacher gate for a large model on rented hardware. Read-only, no plasticity, no training.

The question this answers, for a few dollars of A100 time, is: **which logical structures can
this model execute when the rule is in front of it?** A structure the model cannot demonstrate
correctly cannot be compiled into fast weights - the training loop never sees a correct example
to consolidate. Running this before committing to a training run turns a multi-thousand-dollar
gamble into a ten-minute measurement.

Measured at 0.5B / 1.5B / 3B, the seven original families are near ceiling by 3B (teacher
0.806-1.000), so a larger model would score all-PASS and tell you nothing. The HARD tier
(conj3, comp3, prec5) exists so the gate can still find a ceiling:

    conj3   answer depends on all THREE fields jointly, full 24-entry table in the prompt
    comp3   three chained lookups: cluster -> group -> tier -> bucket
    prec5   five ordered clauses, last match wins

Usage:
    python probe_cloud.py --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct
    python probe_cloud.py --models meta-llama/Llama-3.1-8B-Instruct --tier all --draws 5

What to upload: this file plus the `hdelta/` package and `.venv` requirements
(torch, transformers, accelerate, pyyaml). Nothing else is needed - the task generates itself.

Scoring notes, all of which cost us a wrong conclusion at some point:
  * generated output, never teacher-forced - teacher-forcing hid real failures three times
  * lift over each family's own answer prior, because the priors differ (0.17 to 0.67)
  * several shot draws, because the deliberate path swings widely between draws
  * zero-shot is reported too: if it is above the prior, the answer leaks from the input
    surface and that family's number means nothing
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import time
from pathlib import Path

import torch

from hdelta.encoding import Encoder
from hdelta.metrics import generate, generate_text
from hdelta.modelwrap import load_model
from hdelta.tasks import LogicTask

EASY = ("map", "map_wide", "composition", "conjunction", "disjunction", "negation", "precedence")
HARD = ("conj3", "comp3", "prec5")


def vram():
    if not torch.cuda.is_available():
        return "no cuda"
    free, total = torch.cuda.mem_get_info()
    return f"{(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB"


def probe_model(name, families, n_eval, draws, k_shots, dtype):
    task = LogicTask({f: {"family": f, "seed": 11} for f in families})
    t0 = time.time()
    try:
        model, tok = load_model(name, dtype, "cuda")
    except Exception as e:                      # OOM or missing weights: report, do not crash
        print(f"  !! could not load {name}: {type(e).__name__}: {e}")
        return None
    enc = Encoder(tok, "cuda")
    print(f"  loaded in {time.time() - t0:.0f}s, vram {vram()}", flush=True)

    rows = {}
    for fam in families:
        bal = task.balance(fam)
        base = bal["majority_share"]
        rng0 = random.Random(900 + sum(map(ord, fam)))
        pz = [task.sample(fam, rng0) for _ in range(n_eval)]

        zero = generate(model, enc, [task.make_pair(p, "direct")[0] for p in pz],
                        task.max_answer_tokens())
        z_acc = sum(g == p.answer for g, p in zip(zero, pz)) / len(pz)

        taught = []
        for d in range(draws):
            rng = random.Random(7000 + 13 * d)
            shots = [task.sample(fam, rng) for _ in range(k_shots)]
            prompts = [task.make_pair(p, "free", shots=shots, rule_text=True)[0] for p in pz]
            got = [task.parse_generation(t)[0]
                   for t in generate_text(model, enc, prompts, 96)]
            taught.append(sum(g == p.answer for g, p in zip(got, pz)) / len(pz))

        tm = st.mean(taught)
        lift = (tm - base) / (1 - base) if base < 1 else float("nan")
        leak = (z_acc - base) >= 0.15
        rows[fam] = {"baseline": base, "n_answers": bal["n_answers"], "zero": z_acc,
                     "taught_mean": tm, "taught_sd": st.pstdev(taught), "taught": taught,
                     "lift": lift, "rule_chars": len(task.describe(fam)),
                     "pass": bool(lift >= 0.5 and not leak), "leak": bool(leak)}
        print(f"    {fam:>12s}  base {base:.3f}  zero {z_acc:.3f}  taught {tm:.3f}"
              f" +- {st.pstdev(taught):.3f}  lift {lift:+.3f}"
              f"  {'PASS' if rows[fam]['pass'] else 'fail'}", flush=True)

    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tier", choices=["easy", "hard", "all"], default="all")
    ap.add_argument("--n-eval", type=int, default=48)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--k-shots", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/cloud")
    a = ap.parse_args()

    fams = {"easy": EASY, "hard": HARD, "all": EASY + HARD}[a.tier]
    print(f"families ({a.tier}): {', '.join(fams)}")
    print(f"{a.draws} shot draws x {a.n_eval} inputs, generated output only\n")

    out = {}
    for name in a.models:
        print(f"=== {name} ===", flush=True)
        r = probe_model(name, fams, a.n_eval, a.draws, a.k_shots, a.dtype)
        if r:
            out[name] = r
        print()

    if not out:
        print("nothing probed successfully")
        return

    print(f"{'family':>12s}" + "".join(f"{m.split('/')[-1][:14]:>16s}" for m in out))
    for fam in fams:
        cells = []
        for m in out:
            r = out[m].get(fam)
            cells.append("-" if not r else
                         f"{r['lift']:+.2f} {'P' if r['pass'] else 'f'}"
                         + ("!" if r["leak"] else ""))
        print(f"{fam:>12s}" + "".join(f"{c:>16s}" for c in cells))
    print("\n  P = passes gate 1 (lift >= 0.5, no surface leak); ! = zero-shot above the prior,")
    print("  meaning the answer is recoverable from the input and that number is meaningless.")
    print("  A model that passes EVERYTHING has not been probed hard enough - add structures.")

    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for m, r in out.items():
        f = d / f"gate_{m.split('/')[-1].replace('.', '_')}.json"
        f.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"  -> {f}")


if __name__ == "__main__":
    main()

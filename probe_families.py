"""Gate 1 for the writability ladder: can the TEACHER execute each logical family?

A family the frozen model cannot execute with the rule in front of it tells us nothing about
writability - plasticity cannot compile what the teacher never demonstrates correctly. Part D
learned this the expensive way: the exception branch was blamed on the fast weights for two
runs before anyone measured that the teacher was at 0.142 on it.

Reports, per family, on generated output (never teacher-forced):
  zero      no rule, no examples          - must be near the majority-class baseline
  taught    rule stated + k examples      - the teacher ceiling
  lift      (taught - baseline) / (1 - baseline), so families with different answer priors
            are comparable

Several shot draws per family, because the deliberate path swings widely across draws.

    python probe_families.py
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
from pathlib import Path

import torch
import yaml

from hdelta.encoding import Encoder
from hdelta.metrics import generate, generate_text
from hdelta.modelwrap import load_model
from hdelta.tasks import FAMILIES, LogicTask

N_EVAL = 48
N_DRAWS = 3
K_SHOTS = 2
MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

task = LogicTask({f: {"family": f, "seed": 11} for f in FAMILIES})
model, tok = load_model(MODEL, "bfloat16", "cuda")
enc = Encoder(tok, "cuda")
print(f"model {MODEL}\n", flush=True)

rows = {}
for fam in FAMILIES:
    bal = task.balance(fam)
    base = bal["majority_share"]
    rng0 = random.Random(900 + hash(fam) % 1000)
    pz = [task.sample(fam, rng0) for _ in range(N_EVAL)]

    # zero-shot: nothing in context. Should sit at the baseline; well above it would mean the
    # answer is guessable from the input surface, which would invalidate the family.
    zero = generate(model, enc, [task.make_pair(p, "direct")[0] for p in pz],
                    task.max_answer_tokens())
    z_acc = sum(g == p.answer for g, p in zip(zero, pz)) / len(pz)

    taught = []
    for d in range(N_DRAWS):
        rng = random.Random(7000 + 13 * d)
        shots = [task.sample(fam, rng) for _ in range(K_SHOTS)]
        prompts = [task.make_pair(p, "free", shots=shots, rule_text=True)[0] for p in pz]
        texts = generate_text(model, enc, prompts, 64)
        got = [task.parse_generation(t)[0] for t in texts]
        taught.append(sum(g == p.answer for g, p in zip(got, pz)) / len(pz))

    t_mean = st.mean(taught)
    lift = (t_mean - base) / (1 - base) if base < 1 else float("nan")
    rows[fam] = {"baseline": base, "n_answers": bal["n_answers"], "zero": z_acc,
                 "taught_mean": t_mean, "taught_sd": st.pstdev(taught), "taught": taught,
                 "lift": lift, "rule_chars": len(task.describe(fam))}
    print(f"  {fam:>12s}  baseline {base:.3f}  zero {z_acc:.3f}  "
          f"taught {t_mean:.3f} +- {st.pstdev(taught):.3f}  lift {lift:+.3f}", flush=True)

print(f"\n{'family':>13s}{'base':>7s}{'zero':>7s}{'taught':>9s}{'sd':>7s}{'lift':>8s}{'gate1':>7s}")
for f, r in rows.items():
    # gate 1: the teacher must be clearly above the prior, and zero-shot must not already be
    ok = r["lift"] >= 0.5 and (r["zero"] - r["baseline"]) < 0.15
    print(f"{f:>13s}{r['baseline']:7.3f}{r['zero']:7.3f}{r['taught_mean']:9.3f}"
          f"{r['taught_sd']:7.3f}{r['lift']:+8.3f}{'PASS' if ok else 'fail':>7s}")

os.makedirs("results/families", exist_ok=True)
tag = MODEL.split("/")[-1].replace(".", "_")
Path(f"results/families/teacher_{tag}.json").write_text(json.dumps(rows, indent=2),
                                                        encoding="utf-8")
print(f"-> results/families/teacher_{tag}.json")
print("\nOnly families that PASS are worth spending plasticity on: a family the teacher "
      "cannot\nexecute measures teacher capability, not writability.")

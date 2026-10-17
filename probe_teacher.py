"""What is the TEACHER's ceiling on the override exception?  (frozen model, no plasticity)

Part D blamed the fast weights for never learning the `**/*_test.ext` override. But the
frozen deliberate path - rule stated, worked examples given - is itself wrong on at least
50.6 % +- 7.2 % of override-eligible inputs. The training signal was already degraded before
plasticity saw it. This measures the teacher alone, under three prompt formulations:

  baseline   rule + k shots drawn at random (what every run so far used)
  override   rule + k shots, at least one guaranteed to exercise the override
  first_match the table reversed and the instruction flipped to "first pattern that matches"

The third is NOT just a reordering: reversing the table while keeping "last match wins"
inverts the rule, so the instruction and the worked examples' step order flip with it. The
two formulations are semantically identical; only the reading order differs.

Reported per input class, because a formulation that fixes overrides by breaking ordinary
inputs is not a fix. Several shot draws per variant, because the deliberate path swings
0.52-0.94 across draws - a single draw cannot separate these.

    python probe_teacher.py
"""
from __future__ import annotations

import json
import os
import random
import statistics as st
from pathlib import Path
from typing import List, Tuple

import yaml

from hdelta.encoding import Encoder
from hdelta.metrics import generate_text
from hdelta.modelwrap import load_model
from hdelta.tasks import make_task
from hdelta.tasks.workspace import WorkspaceResolutionTask

N_INPUTS = 24         # per rule, per class
N_DRAWS = 3           # shot draws per variant
K_SHOTS = 2


class FirstMatchTask(WorkspaceResolutionTask):
    """Same rule, read specific-to-general: table reversed, first match wins."""

    def __init__(self, base: WorkspaceResolutionTask):
        self.__dict__.update(base.__dict__)
        self.tables = {r: list(reversed(t)) for r, t in base.tables.items()}

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        hits = [(p, t) for p, t in self.tables[rule]
                if any(p == pp for pp, _ in self.tools[rule].trace(input_str))]
        if not hits:
            return ["no pattern matches"], "//:unowned"
        steps = [f"{p} matches -> {t}" for p, t in hits]
        steps.append(f"first match wins -> {hits[0][1]}")
        return steps, hits[0][1]

    def describe(self, rule: str) -> str:
        table = "; ".join(f"{p} -> {t}" for p, t in self.tables[rule])
        return (f"The build table is: {table}. Check the file path against every "
                f"pattern in order; the target is the one from the first pattern "
                f"that matches.")


def main():
    cfg = yaml.safe_load(open("configs/workspace_gated.yaml", encoding="utf-8"))
    base = make_task(cfg["task"])
    first = FirstMatchTask(base)
    rules = base.rule_names()

    model, tok = load_model(cfg["model"]["name"], cfg["model"].get("dtype", "bfloat16"), "cuda")
    enc = Encoder(tok, "cuda")

    def is_override(task, rule, x):
        return len(task.tools[rule].trace(x)) > 2

    # fixed evaluation inputs per rule, split by class, shared across all variants
    evals = {}
    for i, r in enumerate(rules):
        rng = random.Random(777 + i)
        ov, ord_ = [], []
        while len(ov) < N_INPUTS or len(ord_) < N_INPUTS:
            p = base.sample(r, rng)
            (ov if is_override(base, r, p.input_str) else ord_).append(p)
        evals[r] = {"override": ov[:N_INPUTS], "ordinary": ord_[:N_INPUTS]}

    def run(task, rule, puzzles, shots):
        prompts = [task.make_pair(p, "free", shots=shots, rule_text=True)[0] for p in puzzles]
        texts = generate_text(model, enc, prompts, 64)
        got = [task.parse_generation(t)[0] for t in texts]
        # score against the CANONICAL answer; the variants must agree on the truth
        return sum(g == base.answer_under(rule, p.input_str)
                   for g, p in zip(got, puzzles)) / len(puzzles)

    results = {v: {"override": [], "ordinary": []} for v in
               ("baseline", "override_shot", "first_match")}
    for d in range(N_DRAWS):
        for i, r in enumerate(rules):
            rng = random.Random(5000 + 31 * i + d)
            # baseline: shots drawn at random, as every run so far did
            shots_b = [base.sample(r, rng) for _ in range(K_SHOTS)]
            # guaranteed override: force the first shot to exercise the exception
            ov_shot = None
            while ov_shot is None:
                p = base.sample(r, rng)
                if is_override(base, r, p.input_str):
                    ov_shot = p
            shots_o = [ov_shot] + shots_b[1:]
            for cls in ("override", "ordinary"):
                pz = evals[r][cls]
                results["baseline"][cls].append(run(base, r, pz, shots_b))
                results["override_shot"][cls].append(run(base, r, pz, shots_o))
                # same inputs; only the shots' step order and the rule text differ, and the
                # "free" prompt carries no steps for the evaluated puzzle itself
                rng_f = random.Random(9000 + 31 * i + d)
                shots_f = [first.sample(r, rng_f) for _ in range(K_SHOTS)]
                results["first_match"][cls].append(run(first, r, pz, shots_f))
        print(f"  draw {d+1}/{N_DRAWS} done", flush=True)

    print(f"\n=== teacher accuracy, frozen model, {N_DRAWS} draws x {len(rules)} rules ===")
    print(f"{'variant':>16s}{'override-eligible':>22s}{'ordinary':>18s}")
    for v, d in results.items():
        o, n = d["override"], d["ordinary"]
        print(f"{v:>16s}   {st.mean(o):.3f} +- {st.pstdev(o):.3f}      "
              f"{st.mean(n):.3f} +- {st.pstdev(n):.3f}")
    print("\n(Part D's runs used the 'baseline' formulation. A variant only helps if it "
          "lifts override WITHOUT costing ordinary.)")
    os.makedirs("results/teacher", exist_ok=True)
    Path("results/teacher/ceiling.json").write_text(json.dumps(results, indent=2),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()

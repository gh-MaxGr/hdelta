"""Which repo skill is worth compiling? Measure, do not guess.

A skill is worth compiling only where figuring it out is EXPENSIVE and checking it is
CHEAP. The last attempt compiled three shell-command templates: cheap to check and cheap
to derive, so compiling them bought nothing a string constant would not. This picks the
next candidate on evidence instead.

Five candidates, all drawn from this repo, all with an exact free verifier. Four of them
require derivation rather than recall - that is the point, because recall is what RAG
already does well and what a compiled dictionary would duplicate.

    paramcount  how many plastic params does this config install?   arithmetic over dims
    vram        will this run fit on the 16 GB card?                derivation + threshold
    chainbug    do these two drivers gate on the SAME marker?       reasoning over a pair
    genbudget   how many generations will this run perform?         arithmetic over config
    metrickey   which metric key answers this question?             recall (the control)

For each we measure the only two numbers that matter:

    COLD  no rules, no examples - can it already do this?   High -> nothing to compile.
    TOLD  procedure stated in context - can it do it at all? Low -> plasticity cannot help.

The gap TOLD - COLD is the thing a compiled skill could capture. `metrickey` is included
as a control: it is pure recall, so if it shows the biggest gap that is a warning the
battery is measuring memorisation, not skill.

Read-only. No plasticity, no training, one model load.

    .venv/Scripts/python.exe probe_candidates.py --n 24
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time

import torch

from hdelta.encoding import Encoder
from hdelta.metrics import generate_text
from hdelta.modelwrap import load_model

# Qwen2.5-3B-Instruct, the model every run in this repo uses.
HID, NH, NKV, HD, NLAYER = 2048, 16, 2, 128, 36
PROJ = {"q_proj": HID * NH * HD, "k_proj": HID * NKV * HD,
        "v_proj": HID * NKV * HD, "o_proj": NH * HD * HID,
        "down_proj": HID * 11008}
MET = {
    "the compiled zero-shot accuracy for rule R": "R/zero_direct_acc",
    "the share of writes that were wrong": "verifier/contamination",
    "how often both paths agreed on a wrong answer for rule R": "R/mon_agree_wrong",
    "the deliberate path's error floor for rule R": "R/mon_floor",
    "perplexity on the reference novel": "lm_ppl",
    "fast-weight energy for group G": "fw_norm_ratio_group/G",
    "how many items the verifier rejected": "verifier/rejects",
    "accuracy on recently seen stream puzzles": "seen/zero_direct_acc",
}


def make(kind, rng):
    """(question, exact answer). Every answer is checkable by pure computation."""
    if kind == "paramcount":
        projs = rng.choice([["q_proj", "k_proj", "v_proj", "o_proj"], ["down_proj"],
                            ["q_proj", "o_proj"], ["q_proj", "k_proj", "v_proj"]])
        n = rng.choice([2, 3, 4, 6, 9, 12])
        tot = n * sum(PROJ[p] for p in projs)
        q = (f"A plastic slice covers {', '.join(projs)} on the top {n} layers of "
             f"Qwen2.5-3B-Instruct (hidden 2048, 16 heads, 2 kv heads, head_dim 128, "
             f"MLP intermediate 11008). How many plastic parameters, in millions to 2dp?")
        return q, f"{tot / 1e6:.2f}"
    if kind == "vram":
        n = rng.choice([2, 4, 6, 9, 12, 18])
        projs = rng.choice([["q_proj", "k_proj", "v_proj", "o_proj"], ["down_proj"]])
        plastic = n * sum(PROJ[p] for p in projs)
        # frozen bf16 + fp32 dW + two fp32 Adam moments
        gib = (3.086e9 * 2 + plastic * 4 * 3) / 2 ** 30
        q = (f"Qwen2.5-3B-Instruct (3.086B params) in bfloat16, with a plastic slice over "
             f"{', '.join(projs)} on {n} layers kept in fp32 plus two fp32 Adam moments. "
             f"Weights only, ignoring activations: does it fit in 16 GiB? Answer yes or no.")
        return q, "yes" if gib < 16 else "no"
    if kind == "chainbug":
        a, b = rng.sample(["variance", "mlpvar", "cloudcheck", "poison", "ladder", "locus"], 2)
        same = rng.random() < 0.5
        second = a if same else b
        q = (f"Driver one waits on results/{a}.DONE then runs a GPU job. Driver two waits "
             f"on results/{second}.DONE then runs a different GPU job. Both are already "
             f"running. Can two GPU jobs start from a single marker being written? "
             f"Answer yes or no.")
        return q, "yes" if same else "no"
    if kind == "genbudget":
        puzzles = rng.choice([600, 1080, 1200, 1800])
        every = rng.choice([25, 50, 100])
        pn = rng.choice([32, 48, 64])
        rules = rng.choice([1, 2, 3])
        n = (puzzles // every + 1) * pn * rules
        q = (f"A run streams {puzzles} puzzles and probes every {every}, plus one probe at "
             f"t=0. Each probe generates one direct answer for each of {pn} held-out items "
             f"per rule, across {rules} rule(s). How many direct generations in total?")
        return q, str(n)
    ask = rng.choice(list(MET))
    return (f"In this testbed's probes.jsonl, which metric key holds {ask}? "
            f"Give the key only."), MET[ask]


PROCEDURE = {
    "paramcount": ("Each projection is a dense matrix. q_proj and o_proj are "
                   "hidden x (heads*head_dim). k_proj and v_proj are hidden x "
                   "(kv_heads*head_dim). down_proj is hidden x intermediate. Sum the listed "
                   "projections for one layer, multiply by the layer count, divide by 1e6."),
    "vram": ("Frozen weights cost 2 bytes per parameter in bf16. The plastic slice costs an "
             "extra 12 bytes per plastic parameter: a 4-byte fp32 delta plus two 4-byte fp32 "
             "Adam moments. Add them, convert to GiB by dividing by 2^30, compare with 16."),
    "chainbug": ("Two drivers collide only if they wait on the SAME marker file. Compare the "
                 "two marker names: identical means one write starts both jobs, different "
                 "means they are correctly serialised."),
    "genbudget": ("The number of probes is puzzles // probe_every, plus one for the baseline "
                  "at t=0. Each probe does probe_n generations per rule, so multiply probes "
                  "by probe_n by the number of rules."),
    "metrickey": ("Per-rule metrics are namespaced '<rule>/<metric>'. Verifier state is "
                  "'verifier/<field>'. Per-group energy is 'fw_norm_ratio_group/<group>'. "
                  "Whole-model metrics such as lm_ppl are flat with no prefix."),
}
KINDS = ("paramcount", "vram", "chainbug", "genbudget", "metrickey")


def norm(s):
    """Last non-empty line. These tasks need arithmetic, so the model reasons first and
    puts the answer at the END; taking the first line scored the preamble."""
    lines = [x.strip() for x in s.strip().split("\n") if x.strip()]
    t = lines[-1] if lines else ""
    t = re.sub(r"^(so,? |therefore,? |thus,? )?(the )?(answer|key|result|total)\s*(is|:)\s*",
               "", t, flags=re.I)
    return t.strip().strip("`'\"*.").strip()


def correct(kind, got, want):
    """Search the WHOLE output, preferring the end.

    A 48-token cap truncated the chain of thought on three candidates and scored them
    0.00, which reads as "cannot do it even when told" and is really "ran out of room".
    Matching anywhere is deliberately lenient: for an exact 2dp float or an exact integer
    the collision risk is low, and being lenient here biases AGAINST the thing I want to
    find (a big told-cold gap), so it cannot manufacture a candidate.
    """
    g, w = got.lower(), want.lower()
    if kind in ("vram", "chainbug"):
        hits = re.findall(r"\b(yes|no)\b", g)
        return bool(hits) and hits[-1] == w
    if kind == "paramcount":
        return any(abs(float(m) - float(w)) < 0.02
                   for m in re.findall(r"\d+\.\d{1,3}", g))
    if kind == "genbudget":
        return any(m.replace(",", "") == w for m in re.findall(r"\d[\d,]*", g))
    return w in g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-new", type=int, default=320)   # arithmetic needs room to reason
    ap.add_argument("--out", default="results/candidates.json")
    a = ap.parse_args()

    t0 = time.time()
    model, tok = load_model(a.model, "bfloat16", "cuda")
    enc = Encoder(tok, "cuda")
    print(f"[{time.time()-t0:5.0f}s] {a.model} loaded\n", flush=True)

    rows = {}
    for kind in KINDS:
        rng = random.Random(7 + hash(kind) % 1000)
        items = [make(kind, rng) for _ in range(a.n)]
        res = {}
        for mode in ("cold", "told"):
            pre = "" if mode == "cold" else f"Procedure: {PROCEDURE[kind]}\n\n"
            prompts = [f"{pre}Question: {q}\nAnswer:" for q, _ in items]
            outs = generate_text(model, enc, prompts, a.max_new)
            res[mode] = sum(correct(kind, o, w) for o, (_, w) in zip(outs, items)) / len(items)
            res[f"{mode}_sample"] = norm(outs[0])
        rows[kind] = res
        print(f"  {kind:<11} cold {res['cold']:.2f}   told {res['told']:.2f}   "
              f"gap {res['told']-res['cold']:+.2f}", flush=True)
        print(f"              q: {items[0][0][:96]}")
        print(f"              want {items[0][1]!r}  cold -> {res['cold_sample'][:60]!r}\n",
              flush=True)

    json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nWorth compiling: TOLD high (the model can do it at all) and COLD low (it")
    print("cannot do it unprompted). metrickey is the recall control - if it tops the")
    print("table, the battery is measuring memorisation rather than skill.")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

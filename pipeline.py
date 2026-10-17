"""The whole design, end to end: instinct -> (unsure) keyed failure memory -> (no note) deliberation.

Each part was measured alone (predict_failure.py, failure_memory.py). This runs them as ONE
system over a deployment stream and asks whether the combination beats each part.

The stream is one pass over the 192 conjunction inputs in a seeded order. For each input:

    1. INSTINCT answers (bare prompt, generated). Its confidence = probability of that answer.
    2. confidence >= tau                -> output the instinct's answer
    3. else, a KEYED note applies       -> instinct answers again with the note(s) in the prompt
    4. else                             -> DELIBERATE path (rule stated, 2 worked examples)

After every output the verifier checks it, as a game engine or test suite would. Whenever the
instinct's own answer was wrong, a note is written - input, what the instinct said, the truth,
and the rule's reason - so memory starts EMPTY and fills from the system's own verified
failures. That is the only way memory can exist for the no-pain runs (they have no failure
log), and it is how a deployed system would get it. Memory content is therefore the same in
every arm at every step, since it depends only on the instinct and the truth; the arms differ
only in what they DO with a deferred input.

Everything else is exactly what was measured: note wording, reason-key retrieval (a note fires
only if the input matches every field its reason names; newest first, at most 2), K=32, the
deliberate prompt and its shots (Random(7)), unbatched cached instinct answers.

Arms, at each confidence threshold tau:
    instinct      the instinct's answer, always
    gate>delib    deferred inputs go straight to deliberation          (predict_failure's gate)
    gate>memory   deferred inputs use a note if one applies, else the instinct's answer
    full          deferred inputs use a note if one applies, else deliberation

Predictions, stated before the run:
    P1  full >= gate>delib in accuracy, every run, every tau (memory never harms).
    P2  no-pain runs: both reach ~1.0 once ~30 % is deferred; full makes FEWER deliberate calls.
    P3  pain s2 (deliberation damaged, 0.79): full > gate>delib.

    .venv/Scripts/python.exe pipeline.py                    # all 7 conjunction runs
    .venv/Scripts/python.exe pipeline.py --limit 24 RUN     # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

import numpy as np
import torch

from failure_memory import K, Answers, reason_key, recall_key, with_notes
from hdelta.encoding import Encoder
from hdelta.metrics import generate_text, trim_to_answer
from hdelta.tasks.base import Puzzle
from idiom import load_compiled
from predict_failure import BASELINE, PAIN, RULE, Reader, domain, parse_label

TAUS = (0.5, 0.7, 0.9, 0.95, 0.99)
ORDERS = 5
ARMS = ("instinct", "gate>delib", "gate>memory", "full")


def run_one(run, device="cuda", limit=None, max_new_delib=128):
    run = Path(run)
    cfg, task, model, tok, ctrl, norm = load_compiled(run, device)
    enc = Encoder(tok, device)
    rd = Reader(model, enc, device)
    xs = domain(task)
    if limit:
        xs = xs[:: max(1, len(xs) // limit)][:limit]
    truth = {x: task.answer_under(RULE, x) for x in xs}
    direct = {x: task.make_pair(Puzzle(x, [], truth[x], RULE, {}), "direct")[0] for x in xs}
    answer = Answers(model, enc)

    # 1. instinct: unbatched answer, confidence read at the digit position
    inst = dict(zip(xs, answer([direct[x] for x in xs])))
    idx = {f"#q{i}": i - 1 for i in range(1, 7)}
    bel = rd.belief([direct[x] for x in xs])
    conf = {x: float(bel[i, idx[inst[x]]]) if inst[x] in idx else 0.0 for i, x in enumerate(xs)}

    # 4. deliberation: computed once for every input, used identically by every arm
    shots_rng = random.Random(7)
    shots = [task.sample(RULE, shots_rng) for _ in range(2)]
    dprompts = [task.make_pair(Puzzle(x, [], truth[x], RULE, {}), "free", shots=shots,
                               rule_text=True)[0] for x in xs]
    dtexts = generate_text(model, enc, dprompts, max_new_delib)
    delib = {x: parse_label(task.parse_generation(t)[0]) for x, t in zip(xs, dtexts)}
    # tokens up to the answer line: the model rambles on after answering, so raw length hits the
    # cap on almost every trace while the answer sits ~15 tokens in (checked by eye)
    dlen = [len(enc.ids(trim_to_answer(task, t))) for t in dtexts]

    out = {"run": str(run), "seed": cfg.get("seed"),
           "pain_lambda": cfg["stream"].get("pain_lambda", 0.0), "n": len(xs),
           "instinct_acc": float(np.mean([inst[x] == truth[x] for x in xs])),
           "delib_acc": float(np.mean([delib[x] == truth[x] for x in xs])),
           "delib_unparseable": sum(1 for x in xs if not delib[x]),
           "delib_tokens_to_answer_max": max(dlen),
           "delib_answer_past_cap": sum(l >= max_new_delib for l in dlen),
           "tau": {}}

    # 2-3. the stream. Memory depends only on instinct + truth, so walk it once per order and
    # record which notes each input would see; the arms are then read off per tau.
    walks = []
    for o in range(ORDERS):
        order = list(xs)
        random.Random(100 + o).shuffle(order)
        memory, keys, seen = [], [], []
        for x in order:
            hits = recall_key(task, x, memory, keys)
            m = answer([with_notes(task, direct[x], hits, True)])[0] if hits else None
            seen.append((x, bool(hits), m))
            if inst[x] != truth[x]:                         # verifier caught the instinct
                rec = {"input": x, "move": inst[x], "truth": truth[x]}
                memory.insert(0, rec)
                keys.insert(0, reason_key(task, rec))
                memory, keys = memory[:K], keys[:K]
        walks.append(seen)

    for tau in TAUS:
        per = {a: [] for a in ARMS}
        cost = {"deferred": [], "delib_calls": [], "memory_calls": [], "delib_calls_2nd_half": [],
                "mem_wrong_delib_right": [], "mem_right_delib_wrong": []}
        for seen in walks:
            ok = {a: 0 for a in ARMS}
            dc = mc = dc2 = harm = helped = 0
            for t, (x, has, m) in enumerate(seen):
                a, tr = inst[x], truth[x]
                sure = conf[x] >= tau
                ok["instinct"] += a == tr
                ok["gate>delib"] += (a if sure else delib[x]) == tr
                ok["gate>memory"] += (a if sure else (m if has else a)) == tr
                full = a if sure else (m if has else delib[x])
                ok["full"] += full == tr
                if not sure:
                    if has:
                        mc += 1
                        harm += (m != tr) and (delib[x] == tr)
                        helped += (m == tr) and (delib[x] != tr)
                    else:
                        dc += 1
                        dc2 += t >= len(seen) // 2
            for a_ in ARMS:
                per[a_].append(ok[a_] / len(seen))
            cost["deferred"].append(float(np.mean([conf[x] < tau for x, _, _ in seen])))
            cost["delib_calls"].append(dc)
            cost["memory_calls"].append(mc)
            cost["delib_calls_2nd_half"].append(dc2)
            cost["mem_wrong_delib_right"].append(harm)
            cost["mem_right_delib_wrong"].append(helped)
        out["tau"][str(tau)] = {
            "acc": {a: {"mean": st.mean(v), "min": min(v), "max": max(v)} for a, v in per.items()},
            **{k: st.mean(v) for k, v in cost.items()},
            "gate_delib_calls": st.mean(c + m for c, m in zip(cost["delib_calls"],
                                                             cost["memory_calls"]))}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def show(r):
    print(f"\n=== {r['run']}  pain_lambda={r['pain_lambda']}  seed={r['seed']}  n={r['n']} ===")
    print(f"  instinct {r['instinct_acc']:.3f}   deliberate {r['delib_acc']:.3f} "
          f"({r['delib_unparseable']} unparseable, answer by token {r['delib_tokens_to_answer_max']} at "
          f"worst, {r['delib_answer_past_cap']} past the cap)")
    print(f"  {'tau':>5} {'defer':>6}" + "".join(f"{a:>13}" for a in ARMS)
          + f"{'delib calls':>13}{'(no memory)':>12}{'mem calls':>10}{'harm':>6}{'help':>6}")
    for tau, c in r["tau"].items():
        print(f"  {tau:>5} {c['deferred']:>6.0%}" + "".join(f"{c['acc'][a]['mean']:>13.3f}" for a in ARMS)
              + f"{c['delib_calls']:>13.1f}{c['gate_delib_calls']:>12.1f}{c['memory_calls']:>10.1f}"
              + f"{c['mem_wrong_delib_right']:>6.1f}{c['mem_right_delib_wrong']:>6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="results/pipeline.json")
    a = ap.parse_args()
    rows = []
    for run in (a.runs or BASELINE + PAIN):
        if not (Path(run) / "delta.pt").exists():
            print(f"\n=== {run}: no delta.pt, skipped ===")
            continue
        r = run_one(run, a.device, a.limit)
        show(r)
        rows.append(r)
        json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nharm = deferred inputs memory got WRONG that deliberation would have got right (P1).")
    print("'delib calls' is the full system; '(no memory)' is what gate>delib spends at the same tau.")

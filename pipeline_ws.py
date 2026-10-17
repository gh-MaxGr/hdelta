"""The pipeline on WORKSPACE: the task where the confidence gate is known to fail.

pipeline.py showed instinct -> keyed memory -> deliberation working on conjunction, the easy
case. Workspace is the hard one, and three earlier findings meet on it:

  - override exceptions (`**/*_test.py` and the like, ~12 % of inputs) are ~50x harder to
    write into fast weights than the ordinary package mapping;
  - the instinct is CONFIDENTLY wrong on them (the self-verification blind spot); Part D's
    gate collapsed between 0.80 and 0.90 accuracy;
  - the frozen deliberate path is poor on overrides (0.14, 0.42 with an override in the shots).

So the confidence gate may never send the exception errors anywhere, and deliberation may not
fix them if it did. The design's answer is keyed memory: a note written from a verified
failure, fired by the conditions its reason names. That lookup costs microseconds, so it
can run on EVERY input, not only unsure ones, which makes the memory a router for exceptions
learned from the verifier's reasons.

What a note's KEY is made of - the one variable between the memory arms (note text is the
same in all of them):

    pos        every pattern the reason says MATCHED, in the same workspace. What a real
               build tool reports ("belongs to T via pattern P").
    neg        pos + the patterns checked AFTER the winner that did NOT match - the reason
               also says why nothing overrode it. Model-free check: ordinary notes are 0.84
               precise under pos, because they also fire on same-package test files.
    decisive   the winning pattern + the later non-matches. Under last-match-wins the earlier
               matches were overridden and decided nothing; dropping them lets one override
               failure cover every package.

The workspace is always part of the key: the verifier consulted one table, and the same
pattern means different targets in different tables (`libs/data-index/**` is in A and C,
`**/*_test.py` in all four). A key holds only the few patterns its reason names. It never
holds the table, so checking it costs a few glob matches, not a tool call.

The stream is 200 distinct inputs per workspace at the NATURAL input distribution, the four
workspaces mixed in 3 seeded orders. Memory starts empty; a note is written whenever the
instinct's answer was wrong (the same rule as pipeline.py). Confidence = p_min, the weakest
generated token, pre-registered because it was Part D's best statistic at high accuracy
targets; p_geo and p_disc AUCs are reported beside it. Deliberation uses each run's own
slow-path settings (rule stated, its shots, exception-covering shot if the run used one).

Arms at each tau:
    instinct                 bare answer, always
    gate>delib               p_min < tau -> deliberate
    gated/<key>              p_min < tau -> note if one fires, else deliberate
    always/<key>             note if one fires (any confidence), else gate>delib

Predictions, stated before the run:
    P1  blind spot: on mech_h1_decay most EXCEPTION failures are confident - at tau 0.9 more
        than half have p_min >= 0.9, so gate>delib leaves them wrong.
    P2  weak fallback: deliberation on exceptions is well below 1.0.
    P3  always/neg and always/decisive lift exception accuracy well above gate>delib, with
        zero harm on ordinary inputs; decisive covers more than neg.
    P4  on the gate3_gated runs (many ordinary failures) always/pos HARMS exception inputs -
        ordinary notes firing on test files - and neg/decisive do not.

Pre-check, run AFTER the predictions above were written (scratch ws_copy.py, mech_h1_decay,
12 exception inputs of workspace A): with the learned delta ON the model does NOT copy a
correct note - 0/12 with a note about another file under the same override, 0/12 with a note
about THE SAME FILE. The compiled package reflex overrides the context (the note only lowers
p_min, to ~0.55). With the delta OFF the frozen model copies 4-5/12. On conjunction notes were
copied blindly. So P3 as written is expected to FAIL in the `read` mode, and a second answer
mode is added:

    read       the instinct answers with the notes in its prompt (the design so far)
    direct     the newest firing note's verified answer IS the output - no model call. Safe
               exactly to the extent the key is precise, which is measured.

    P5  always/<neg|decisive>/direct lifts exception accuracy far above gate>delib, with zero
        harm; always/pos/direct harms exception inputs on the gate3_gated runs (as P4).

    .venv/Scripts/python.exe pipeline_ws.py
    .venv/Scripts/python.exe pipeline_ws.py --per-rule 8 --orders 1 results/mech_h1_decay   # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from hdelta.encoding import Encoder
from hdelta.metrics import generate_text, trim_to_answer
from hdelta.tasks.base import Puzzle
from hdelta.tasks.workspace import _glob_to_re
from hdelta.toolcache import generate_with_confidence
from idiom import load_compiled
from predict_failure import auc

RUNS = ["results/mech_h1_decay", "results/gate3_gated", "results/gate3_gated_s1",
        "results/gate3_gated_s2"]
TAUS = (0.5, 0.7, 0.9, 0.95, 0.99)
KEYS = ("pos", "neg", "decisive")
SHOW = 2                     # notes shown at most, newest first (as in failure_memory.py)
STAT = "p_min"
_rx = lru_cache(maxsize=None)(_glob_to_re)     # fires() is called millions of times


class Instinct:
    """Unbatched greedy answers with per-token confidence, cached by exact prompt text
    (batched generation leaks batch-mates into the answer - see failure_memory.Answers)."""

    def __init__(self, model, enc, max_new):
        self.model, self.enc, self.max_new, self.cache = model, enc, max_new, {}

    def __call__(self, prompt):
        if prompt not in self.cache:
            self.cache[prompt] = generate_with_confidence(self.model, self.enc, [prompt],
                                                          self.max_new, chunk=1)[0]
        return self.cache[prompt]


def reason(task, rule, x):
    """The verifier's account: every pattern that matched, in table order, and the winner."""
    table = task.tables[rule]
    hits = [i for i, (p, _) in enumerate(table) if _rx(p).match(x)]
    steps, answer = task.solve(rule, x)
    win = hits[-1] if hits else None
    later_misses = [i for i in range(win + 1, len(table))] if win is not None else []
    return {"steps": steps, "answer": answer, "matched": hits, "winner": win,
            "later_misses": later_misses}


def make_key(kind, rule, why, table):
    """{"rule", "must": [globs that must match], "must_not": [globs that must not]}."""
    pats = lambda idx: [table[i][0] for i in idx]
    if kind == "pos":
        return {"rule": rule, "must": pats(why["matched"]), "must_not": []}
    if kind == "neg":
        return {"rule": rule, "must": pats(why["matched"]), "must_not": pats(why["later_misses"])}
    if kind == "decisive":
        must = [] if why["winner"] is None else pats([why["winner"]])
        return {"rule": rule, "must": must, "must_not": pats(why["later_misses"])}
    raise ValueError(kind)


def fires(key, rule, x):
    if key["rule"] != rule or not key["must"]:
        return False          # a lesson with no stated scope never fires
    return (all(_rx(p).match(x) for p in key["must"])
            and not any(_rx(p).match(x) for p in key["must_not"]))


def note_line(task, rec):
    return (f"- On [{task.tag(rec['rule'])}] {rec['input']} you answered {rec['move'] or '(nothing)'};"
            f" it was {rec['truth']}, because {'; '.join(rec['steps'])}.")


def with_notes(task, prompt, recs):
    if not recs:
        return prompt
    block = "Mistakes you made before:\n" + "\n".join(note_line(task, r) for r in recs) + "\n\n"
    return prompt.replace(task.header, task.header + block, 1)


def delib_shots(task, cfg, rule):
    """The run's own slow-path shots, fixed per workspace (mirrors experiment.expose_chunk)."""
    sc = cfg["stream"]
    rng = random.Random(7)
    shots = [task.sample(rule, rng) for _ in range(int(sc.get("n_shots", 2)))]
    if sc.get("shots_cover_exception") and shots:
        for _ in range(64):
            cand = task.sample(rule, rng)
            if task.is_exception(rule, cand.input_str):
                shots[0] = cand
                break
    return shots


def run_one(run, device="cuda", per_rule=200, orders=3, max_new_delib=160):
    run = Path(run)
    cfg, task, model, tok, ctrl, norm = load_compiled(run, device)
    enc = Encoder(tok, device)
    sc = cfg["stream"]

    # the stream: distinct inputs at the NATURAL distribution (bypasses exception_fraction)
    rng = random.Random(0)
    items = []
    for rule in task.rule_names():
        seen = set()
        while len(seen) < per_rule:
            seen.add(task._draw_input(rule, rng))
        items += [(rule, x) for x in sorted(seen)]
    why = {(r, x): reason(task, r, x) for r, x in items}
    truth = {k: v["answer"] for k, v in why.items()}
    exc = {(r, x): task.is_exception(r, x) for r, x in items}
    direct = {(r, x): task.make_pair(Puzzle(x, [], truth[(r, x)], r, {}), "direct")[0]
              for r, x in items}
    instinct = Instinct(model, enc, task.max_answer_tokens())

    # 1. instinct + confidence
    ans = {k: instinct(direct[k]) for k in items}
    inst = {k: a.text for k, a in ans.items()}
    conf = {k: getattr(a, STAT) for k, a in ans.items()}
    wrong = {k: inst[k] != truth[k] for k in items}

    # 4. deliberation, once per input, identical for every arm
    shots = {r: delib_shots(task, cfg, r) for r in task.rule_names()}
    dprompts = [task.make_pair(Puzzle(x, [], truth[(r, x)], r, {}), "free", shots=shots[r],
                               rule_text=bool(sc.get("rule_text", True)))[0] for r, x in items]
    dtexts = generate_text(model, enc, dprompts, max_new_delib)
    delib = {k: task.parse_generation(t)[0] for k, t in zip(items, dtexts)}
    dlen = [len(enc.ids(trim_to_answer(task, t))) for t in dtexts]

    def acc(pred, keys=None):
        keys = items if keys is None else keys
        return float(np.mean([pred[k] == truth[k] for k in keys])) if keys else float("nan")

    E = [k for k in items if exc[k]]
    O = [k for k in items if not exc[k]]
    failE = np.array([wrong[k] for k in E])
    out = {"run": str(run), "seed": cfg.get("seed"), "n": len(items), "n_exc": len(E),
           "instinct": {"all": acc(inst), "exc": acc(inst, E), "ord": acc(inst, O)},
           "delib": {"all": acc(delib), "exc": acc(delib, E), "ord": acc(delib, O),
                     "unparseable": sum(1 for k in items if not delib[k]),
                     "tokens_to_answer_max": max(dlen),
                     "answer_past_cap": sum(l >= max_new_delib for l in dlen)},
           "auc": {s: {"all": auc([1 - getattr(ans[k], s) for k in items],
                                  [wrong[k] for k in items]),
                       "exc": auc([1 - getattr(ans[k], s) for k in E], failE)}
                   for s in ("p_min", "p_geo", "p_disc")},
           "exc_failures": int(failE.sum()),
           "exc_failures_confident": {str(t): int(sum(wrong[k] and conf[k] >= t for k in E))
                                      for t in TAUS},
           "ord_failures": int(sum(wrong[k] for k in O)),
           "tau": {}, "retrieval": {}, "notes_written": {}}

    # 2-3. the stream: memory depends only on instinct + truth, identical across arms
    walks = []
    for o in range(orders):
        order = list(items)
        random.Random(100 + o).shuffle(order)
        memory = []                         # newest first
        seen_rows = []
        for k in order:
            r, x = k
            row = {"k": k}
            for kind in KEYS:
                hits = [m for m in memory if fires(m["key"][kind], r, x)][:SHOW]
                row[kind] = instinct(with_notes(task, direct[k], hits)).text if hits else None
                row[kind + "_direct"] = hits[0]["truth"] if hits else None
                row[kind + "_prec"] = [truth[(m["rule"], m["input"])] == truth[k] for m in hits]
            seen_rows.append(row)
            if wrong[k]:                    # the verifier caught the instinct
                w = why[k]
                memory.insert(0, {"rule": r, "input": x, "move": inst[k], "truth": truth[k],
                                  "steps": w["steps"], "confident_when_written": conf[k],
                                  "key": {kind: make_key(kind, r, w, task.tables[r])
                                          for kind in KEYS}})
        walks.append(seen_rows)
        out["notes_written"][str(o)] = len(memory)

    for kind in KEYS:
        shown = [p for rows in walks for row in rows for p in row[kind + "_prec"]]
        out["retrieval"][kind] = {
            "precision": float(np.mean(shown)) if shown else float("nan"),
            "notes_shown": len(shown) / orders,
            "inputs_fired_exc": st.mean(sum(1 for row in rows if row[kind] is not None and exc[row["k"]])
                                        for rows in walks),
            "inputs_fired_ord": st.mean(sum(1 for row in rows if row[kind] is not None and not exc[row["k"]])
                                        for rows in walks)}

    arms = ["instinct", "gate>delib"] + [f"{m}/{kind}/{mode}" for mode in ("read", "direct")
                                         for m in ("gated", "always") for kind in KEYS]
    for tau in TAUS:
        per = {a: {"all": [], "exc": [], "ord": [], "delib_calls": [], "mem_calls": [],
                   "harm": [], "help": []} for a in arms}
        for rows in walks:
            pred = {a: {} for a in arms}
            cost = {a: [0, 0] for a in arms}
            for row in rows:
                k = row["k"]
                sure = conf[k] >= tau
                base = inst[k] if sure else delib[k]
                pred["instinct"][k] = inst[k]
                pred["gate>delib"][k] = base
                cost["gate>delib"][0] += not sure
                for kind in KEYS:
                    for mode in ("read", "direct"):
                        m = row[kind] if mode == "read" else row[kind + "_direct"]
                        g, al = f"gated/{kind}/{mode}", f"always/{kind}/{mode}"
                        if sure:
                            pred[g][k] = inst[k]
                        elif m is not None:
                            pred[g][k] = m
                            cost[g][1] += 1
                        else:
                            pred[g][k] = delib[k]
                            cost[g][0] += 1
                        if m is not None:
                            pred[al][k] = m
                            cost[al][1] += 1
                        else:
                            pred[al][k] = base
                            cost[al][0] += not sure
            ref = pred["gate>delib"]
            for a in arms:
                p = pred[a]
                per[a]["all"].append(acc(p))
                per[a]["exc"].append(acc(p, E))
                per[a]["ord"].append(acc(p, O))
                per[a]["delib_calls"].append(cost[a][0])
                per[a]["mem_calls"].append(cost[a][1])
                per[a]["harm"].append(sum(p[k] != truth[k] and ref[k] == truth[k] for k in items))
                per[a]["help"].append(sum(p[k] == truth[k] and ref[k] != truth[k] for k in items))
        out["tau"][str(tau)] = {
            "deferred": float(np.mean([conf[k] < tau for k in items])),
            "arms": {a: {m: st.mean(v) for m, v in d.items()} for a, d in per.items()}}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def show(r):
    i, d = r["instinct"], r["delib"]
    print(f"\n=== {r['run']}  seed={r['seed']}  n={r['n']} ({r['n_exc']} exceptions) ===")
    print(f"  instinct  all {i['all']:.3f}  exc {i['exc']:.3f}  ord {i['ord']:.3f}   "
          f"failures: {r['exc_failures']} exc, {r['ord_failures']} ord")
    print(f"  deliberate all {d['all']:.3f}  exc {d['exc']:.3f}  ord {d['ord']:.3f}   "
          f"({d['unparseable']} unparseable, answer by token {d['tokens_to_answer_max']} at worst, "
          f"{d['answer_past_cap']} past the cap)")
    print("  failure AUC  " + "  ".join(f"{s} all {v['all']:.3f} exc {v['exc']:.3f}"
                                         for s, v in r["auc"].items()))
    print("  exception failures that are CONFIDENT (p_min >= tau): " + "  ".join(
        f"{t}:{n}/{r['exc_failures']}" for t, n in r["exc_failures_confident"].items()))
    for kind, v in r["retrieval"].items():
        print(f"  retrieval {kind:<9} precision {v['precision']:.3f}  fired on {v['inputs_fired_exc']:.0f} exc"
              f" / {v['inputs_fired_ord']:.0f} ord inputs per order")
    for tau in ("0.9", "0.99"):
        c = r["tau"][tau]
        print(f"  tau {tau}  (deferred {c['deferred']:.0%})")
        print(f"    {'arm':<24}{'all':>7}{'exc':>7}{'ord':>7}{'delib':>8}{'mem':>6}{'harm':>6}{'help':>6}")
        for a, v in c["arms"].items():
            if a.startswith("gated/") and a != "gated/decisive/read":
                continue            # the rest are in the JSON; gated ~ gate>delib when errors are confident
            print(f"    {a:<24}{v['all']:>7.3f}{v['exc']:>7.3f}{v['ord']:>7.3f}"
                  f"{v['delib_calls']:>8.1f}{v['mem_calls']:>6.1f}{v['harm']:>6.1f}{v['help']:>6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per-rule", type=int, default=200)
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--out", default="results/pipeline_ws.json")
    a = ap.parse_args()
    rows = []
    for run in (a.runs or RUNS):
        if not (Path(run) / "delta.pt").exists():
            print(f"\n=== {run}: no delta.pt, skipped ===")
            continue
        r = run_one(run, a.device, a.per_rule, a.orders)
        show(r)
        rows.append(r)
        json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nharm/help are counted against gate>delib at the same tau. exc = override inputs.")

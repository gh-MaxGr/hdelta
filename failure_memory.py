"""Remembering why it was hurt: a small, readable memory of past failures.

Pain as built changes weights and nothing else - the model tries again with no record of what
went wrong. This adds the lightest possible record the model can actually read:

  LIGHTWEIGHT  at most K one-line notes, recalled only when the circumstances genuinely match
               (>= 2 of the 3 input fields), and in the gated variant only when the instinct is
               itself unsure. No training, no new weights.
  ACCESSIBLE   plain text placed in the prompt ahead of the task.
  RELIABLE     every note is built from verified ground truth - the true answer and the rule's
               own reasoning step - never from what the model believed.
  RATIONALE    each note says WHY: "On south/gamma/spar you answered #q4; it was #q3, because
               zone south with cluster gamma -> #q3". Without the reason, a remembered injury is
               just a prohibition: punishment with no explanation.

Whether the reason earns its place is measured, not assumed. Conjunction's answer depends on
zone+cluster only - the item never matters - so the reason is the one thing that tells the
model which fields the lesson covers. That gives two sharp tests:

  SAME CIRCUMSTANCE, OTHER ITEM   a lesson about south/gamma/spar should transfer to
                                  south/gamma/keel. With the reason it says so; bare, it doesn't.
  MISLEADING                      a note about south/delta/spar retrieved for south/gamma/spar
                                  (two fields shared, but a different circumstance). With the
                                  reason the model can see it does not apply; bare, it may copy.

Memory is built only from failures on a seeded HALF of the 24 circumstances, so every run also
has inputs with no relevant memory - where the memory must not hurt.

Conditions (instinct answers in all of them, generated, empty context apart from the notes):
    none        no memory
    reason      recalled notes, with the why
    bare        the SAME recalled notes, without the why
    irrelevant  the two least-similar notes, with the why - is it relevance or just context?
    gated       reason notes, only when the instinct's own confidence is below 0.9

    .venv/Scripts/python.exe failure_memory.py                  # the three pain runs
    .venv/Scripts/python.exe failure_memory.py --device cpu --limit 8 --synthetic 24 RUN
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from hdelta.encoding import Encoder
from hdelta.metrics import generate
from hdelta.tasks.base import Puzzle
from idiom import load_compiled
from predict_failure import RULE, Reader, circumstance, domain, parse_label

PAIN = ["results/pain/conjunction_s0", "results/pain/conjunction_s1",
        "results/pain/conjunction_s2"]
K = 32
GATE_CONF = 0.9
# Run 2 (2026-09-23) changes ONE thing - which note is shown. Notes, wording, the half-split
# and the weights are identical to run 1. `surface` IS run 1's `reason` condition, kept as the
# head-to-head arm; the `key_*` conditions retrieve on the fields the note's reason names.
CONDITIONS = ("none", "surface", "key", "key_bare", "key_gated", "irrelevant")
STRATA = ("exact", "same circumstance", "misleading", "no recall")


def similarity(a, b):
    return sum(x == y for x, y in zip(a.split("/"), b.split("/"))) / 3.0


def note(task, rec, with_reason):
    s = f"- On {rec['input']} you answered {rec['move']}; it was {rec['truth']}"
    if with_reason:
        s += f", because {task.solve(RULE, rec['input'])[0][0]}"
    return s + "."


def load_memory(run, task, half, synthetic=0, seed=0):
    """Past failures, newest first, deduplicated by input, restricted to `half`, capped at K."""
    recs = []
    if synthetic:
        rng = random.Random(seed)
        xs = domain(task)
        for _ in range(synthetic):
            x = rng.choice(xs)
            t = task.answer_under(RULE, x)
            recs.append({"input": x, "truth": t,
                         "move": rng.choice([f"#q{i}" for i in range(1, 7) if f"#q{i}" != t])})
    else:
        f = Path(run) / "experience.jsonl"
        recs = [r for r in (json.loads(l) for l in f.open(encoding="utf-8"))
                if r.get("kind") == "pain"]
    out, seen = [], set()
    for r in reversed(recs):
        if r["input"] in seen or circumstance(r["input"]) not in half:
            continue
        seen.add(r["input"])
        out.append(r)
    return out[:K]


def recall(x, memory):
    """SURFACE retrieval: >= 2 of 3 fields equal. Measured: sent a misleading note - similar
    input, different circumstance - to ~47 % of inputs, and the model copies notes blindly."""
    ranked = sorted(memory, key=lambda r: -similarity(x, r["input"]))
    hits = [r for r in ranked[:2] if similarity(x, r["input"]) >= 2 / 3]
    far = sorted(memory, key=lambda r: similarity(x, r["input"]))[:2]
    return hits, far


def reason_key(task, rec):
    """The input fields the note's own REASON names, as {position: value}.

    "zone south with cluster gamma -> #q3" on input south/gamma/spar names `south` and `gamma`
    and never mentions `spar` - so the lesson is scoped to (south, gamma) and says nothing about
    the item. Read off the explanation, not hard-coded: nothing here knows conjunction's
    structure. A reason that names no input field yields an empty key, and such a note never
    fires, because a lesson with no stated scope cannot be applied safely.
    """
    words = {w.strip(".,;:()") for w in task.solve(RULE, rec["input"])[0][0].split()}
    return {i: v for i, v in enumerate(rec["input"].split("/")) if v in words}


def recall_key(task, x, memory, keys):
    """REASON-KEY retrieval: a note applies iff x matches every field its reason names.
    Newest first (memory is stored newest first), at most two."""
    fields = x.split("/")
    hits = [r for r, k in zip(memory, keys)
            if k and all(i < len(fields) and fields[i] == v for i, v in k.items())]
    return hits[:2]


def stratum(x, hits):
    if not hits:
        return "no recall"
    if any(r["input"] == x for r in hits):
        return "exact"
    if any(circumstance(r["input"]) == circumstance(x) for r in hits):
        return "same circumstance"
    return "misleading"


def with_notes(task, prompt, recs, with_reason):
    if not recs:
        return prompt
    block = ("Mistakes you made before:\n"
             + "\n".join(note(task, r, with_reason) for r in recs) + "\n\n")
    return prompt.replace(task.header, task.header + block, 1)


class Answers:
    """Unbatched greedy answers, cached by exact prompt text.

    Batched generation left-pads each prompt to the longest in its chunk, so the SAME prompt
    gets different padding depending on its neighbours - and near a tie between labels that
    flips the answer. Caught in the smoke test: on inputs with nothing recalled, the `reason`
    prompt is byte-identical to the `none` prompt, yet scored 0.000 against 0.500 at
    alpha=0.5, purely because its batch-mates carried notes and were longer. That noise would
    sit inside the comparison that matters, since `reason` and `bare` are always batched with
    different neighbours. One prompt at a time has no padding, and the cache makes identical
    prompts give identical answers by construction.
    """

    def __init__(self, model, enc):
        self.model, self.enc, self.cache = model, enc, {}

    def reset(self):
        self.cache = {}

    def __call__(self, prompts):
        todo = [p for p in dict.fromkeys(prompts) if p not in self.cache]
        for p in todo:
            self.cache[p] = parse_label(generate(self.model, self.enc, [p], 8, chunk=1)[0])
        return [self.cache[p] for p in prompts]


def run_one(run, device="cuda", limit=None, synthetic=0, alphas=(1.0, 0.5)):
    run = Path(run)
    cfg, task, model, tok, ctrl, norm = load_compiled(run, device)
    enc = Encoder(tok, device)
    rd = Reader(model, enc, device)
    xs = domain(task)
    keys = sorted({circumstance(x) for x in xs})
    random.Random(1234).shuffle(keys)
    half = set(keys[: len(keys) // 2])
    memory = load_memory(run, task, half, synthetic)
    if limit:
        xs = xs[:: max(1, len(xs) // limit)][:limit]
    truth = [task.answer_under(RULE, x) for x in xs]
    direct = [task.make_pair(Puzzle(x, [], t, RULE, {}), "direct")[0] for x, t in zip(xs, truth)]
    rec = [recall(x, memory) for x in xs]
    # strata are defined by SURFACE retrieval so every row lines up with run 1: `misleading`
    # is exactly the set of inputs that surface overlap handed a wrong-circumstance note
    strata = [stratum(x, h) for x, (h, _) in zip(xs, rec)]
    keys = [reason_key(task, r) for r in memory]
    krec = [recall_key(task, x, memory, keys) for x in xs]

    def precision(hits_per_input):
        shown = [(x, r) for x, hs in zip(xs, hits_per_input) for r in hs]
        good = sum(circumstance(r["input"]) == circumstance(x) for x, r in shown)
        return (good / len(shown)) if shown else float("nan"), len(shown)

    available = [any(circumstance(r["input"]) == circumstance(x) for r in memory) for x in xs]
    def coverage(hits_per_input):
        hit = [any(circumstance(r["input"]) == circumstance(x) for r in hs)
               for x, hs in zip(xs, hits_per_input)]
        n = sum(available)
        return (sum(h for h, a in zip(hit, available) if a) / n) if n else float("nan")

    sp, sn = precision([h for h, _ in rec])
    kp, kn = precision(krec)
    retrieval = {"inputs_with_a_relevant_note": int(sum(available)),
                 "surface": {"precision": sp, "notes_shown": sn,
                             "coverage": coverage([h for h, _ in rec])},
                 "key": {"precision": kp, "notes_shown": kn, "coverage": coverage(krec)},
                 "empty_keys": sum(1 for k in keys if not k),
                 "key_example": {"note": memory[0]["input"] if memory else None,
                                 "key": keys[0] if memory else None}}
    idx = {f"#q{i}": i - 1 for i in range(1, 7)}
    base = ctrl.snapshot()
    answer = Answers(model, enc)
    res = {"run": str(run), "seed": cfg.get("seed"), "n": len(xs), "memory_size": len(memory),
           "memory_circumstances": len({circumstance(r["input"]) for r in memory}),
           "strata": {s: strata.count(s) for s in STRATA}, "retrieval": retrieval,
           "alpha": {}}

    for a in alphas:
        ctrl.restore({k: a * v for k, v in base.items()})
        answer.reset()                    # new weights: nothing cached is valid any more
        ans0 = answer(direct)
        bel = rd.belief(direct)
        conf = np.array([bel[i, idx[x]] if x in idx else 0.0 for i, x in enumerate(ans0)])
        prompts = {
            "none": direct,
            "surface": [with_notes(task, p, h, True) for p, (h, _) in zip(direct, rec)],
            "key": [with_notes(task, p, h, True) for p, h in zip(direct, krec)],
            "key_bare": [with_notes(task, p, h, False) for p, h in zip(direct, krec)],
            "key_gated": [with_notes(task, p, h, True) if c < GATE_CONF else p
                          for p, h, c in zip(direct, krec, conf)],
            "irrelevant": [with_notes(task, p, f, True) for p, (_, f) in zip(direct, rec)],
        }
        cell = {}
        for cond in CONDITIONS:
            ans = answer(prompts[cond])
            ok = np.array([int(x == t) for x, t in zip(ans, truth)])
            cell[cond] = {s: (float(ok[[i for i, z in enumerate(strata) if z == s]].mean())
                              if s in strata else None) for s in STRATA}
            cell[cond]["all"] = float(ok.mean())
        extra = [len(enc.ids(p)) - len(enc.ids(d)) for p, d in zip(prompts["key"], direct)]
        cell["_consulted_gated"] = float(np.mean(conf < GATE_CONF))
        cell["_mean_extra_tokens_when_recalled"] = float(np.mean([e for e in extra if e > 0])
                                                         ) if any(extra) else 0.0
        res["alpha"][str(a)] = cell
    ctrl.restore(base)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


def show(r):
    print(f"\n=== {r['run']}  seed={r['seed']}  n={r['n']}   memory: {r['memory_size']} notes "
          f"over {r['memory_circumstances']} circumstances ===")
    print("  strata (by SURFACE retrieval, same as run 1): "
          + "  ".join(f"{s}={c}" for s, c in r["strata"].items()))
    rt = r["retrieval"]
    print(f"  retrieval  {rt['inputs_with_a_relevant_note']} inputs have a relevant note in memory; "
          f"{rt['empty_keys']} notes had an unscopable (empty) key")
    for m in ("surface", "key"):
        print(f"    {m:<8} precision {rt[m]['precision']:.3f}  "
              f"(right-circumstance share of {rt[m]['notes_shown']} notes shown)   "
              f"coverage {rt[m]['coverage']:.3f}  (relevant notes actually found)")
    print(f"    key example: note {rt['key_example']['note']} -> key {rt['key_example']['key']}")
    for a, cell in r["alpha"].items():
        print(f"  alpha={a}   (gated consulted memory on {cell['_consulted_gated']:.0%} of inputs; "
              f"a recalled block costs ~{cell['_mean_extra_tokens_when_recalled']:.0f} tokens)")
        print(f"    {'condition':<11}" + "".join(f"{s:>19}" for s in STRATA) + f"{'ALL':>8}")
        for cond in CONDITIONS:
            row = cell[cond]
            print(f"    {cond:<11}" + "".join(
                f"{('-' if row[s] is None else f'{row[s]:.3f}'):>19}" for s in STRATA)
                + f"{row['all']:>8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="smoke only: fabricate N failure records instead of reading a pain log")
    ap.add_argument("--out", default="results/failure_memory_key.json")
    a = ap.parse_args()
    rows = []
    for run in (a.runs or PAIN):
        if not (Path(run) / "delta.pt").exists():
            print(f"\n=== {run}: no delta.pt, skipped ===")
            continue
        if not a.synthetic and not (Path(run) / "experience.jsonl").exists():
            print(f"\n=== {run}: no experience.jsonl, skipped ===")
            continue
        r = run_one(run, a.device, a.limit, a.synthetic)
        show(r)
        rows.append(r)
        json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)
    print("\nRun 2 changes only WHICH note is shown. Read it as: on the MISLEADING row, does 'key'")
    print("recover what 'surface' destroyed? (Those inputs now get no note, or a correct one.) On")
    print("SAME CIRCUMSTANCE, does 'key' hold or improve? Does 'key' ever fall BELOW 'none'? With")
    print("retrieval precise, does 'key_bare' still trail 'key' - does the reason matter once the")
    print("model is never shown a wrong note? 'no recall' must equal 'none' in every key condition.")

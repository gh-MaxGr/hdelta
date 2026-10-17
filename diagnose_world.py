"""Why is the deliberate path below a constant policy?

The world run's baseline reported deliberate 0.25 / 0.15 / 0.52 against constant floors of
0.50 / 0.70 / 0.58. Two completely different causes produce that same number:

  FORMAT   the model writes prose, or a malformed counter, and the verifier rejects it
           outright. Fixable with a tighter prompt or constrained decoding.
  PLAY     the model emits legal moves that are simply bad. Not fixable by prompting, and
           it means the deliberate path cannot teach the reflex anything worth having.

Reporting "the gate failed" without separating these would be guessing. This prints what
the model actually emits, with each failure bucketed.

    .venv/Scripts/python.exe diagnose_world.py --n 24
"""
from __future__ import annotations

import argparse
import random
from collections import Counter

from hdelta.encoding import Encoder
from hdelta.metrics import generate_text
from hdelta.modelwrap import load_model
from hdelta.tasks import make_task
from hdelta.tasks.world import CONSTANT_FLOOR, Situation, _VALID_MOVE, legal_moves, score


def bucket(task, rule, move, sit_text):
    """Where did this move land?"""
    if not move.strip():
        return "empty"
    if not _VALID_MOVE.match(move.strip().upper()):
        return "unparseable"
    sit = Situation.parse(sit_text)
    m = move.strip().upper()
    if m.startswith("COUNTER"):
        import re
        mm = re.match(r"COUNTER (\d+) (\w+) FOR (\d+) (\w+)", m)
        g, n = mm.group(2).lower(), int(mm.group(1))
        if sit.stock.get(g, 0) < n:
            return "counter: cannot cover"
        if mm.group(4).lower() != sit.give_g:
            return "counter: asks for a good they are not offering"
    ok, got, why = task.evaluate(rule, move, sit_text)
    return "VERIFIED" if ok else "legal but below bar"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()

    cfg = {"name": "world", "tagged": True, "baseline_n": 8,
           "rules": {r: {"family": r} for r in CONSTANT_FLOOR},
           "tag_map": {r: r for r in CONSTANT_FLOOR}}
    task = make_task(cfg)
    model, tok = load_model("Qwen/Qwen2.5-3B-Instruct", "bfloat16", "cuda")
    enc = Encoder(tok, "cuda")
    print("loaded\n", flush=True)

    for rule in CONSTANT_FLOOR:
        rng = random.Random(10_000)
        pz = [task.sample(rule, rng) for _ in range(a.n)]
        shots = [task.sample(rule, rng) for _ in range(2)]
        prompts = [task.make_pair(p, "free", shots=shots, rule_text=True)[0] for p in pz]
        texts = generate_text(model, enc, prompts, a.max_new)
        moves = [task.parse_generation(t)[0] for t in texts]
        buckets = Counter(bucket(task, rule, mv, p.input_str) for mv, p in zip(moves, pz))
        ref = sum(task.evaluate(rule, p.answer, p.input_str)[0] for p in pz) / len(pz)
        print(f"=== {rule}  (floor {CONSTANT_FLOOR[rule]:.2f}, reference {ref:.2f}) ===")
        for k, v in buckets.most_common():
            print(f"  {v:>3}/{a.n}  {k}")
        for t, mv, p in list(zip(texts, moves, pz))[:a.show]:
            print(f"    situation : {p.input_str}")
            print(f"    reference : {p.answer}")
            print(f"    parsed    : {mv!r}")
            print(f"    raw       : {t.strip()[:240]!r}\n")
        print(flush=True)


if __name__ == "__main__":
    main()

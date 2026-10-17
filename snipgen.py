"""Random Python snippets labelled by EXECUTING them: does this snippet raise?

The quadrant no earlier task tested: a pattern that is not a table AND that a pretrained LLM
already has a real prior for (Python semantics). If fast weights have an application niche,
it is correcting a competent model's systematic mistakes - here a tiny learner on tokens has no
such prior to start from, and a reason cache has little to key on (every snippet is unique).

Snippets are 2-5 statements over ints, floats, strings, lists, dicts and None: arithmetic,
indexing, slicing, method calls, len/int/str conversions, comparisons, augmented assignment.
Each one is executed in a fresh subprocess (5 s timeout, no imports, no I/O builtins), so the
label is exact. Output: results/snippets.jsonl with {code, raises, exc}.

    .venv/Scripts/python.exe snipgen.py --n 12000
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

NAMES = ["a", "b", "c", "x", "y", "items", "name", "total", "d", "n"]


def literal(rng):
    k = rng.randrange(7)
    if k == 0:
        return str(rng.randint(-3, 12))
    if k == 1:
        return repr(round(rng.uniform(-2, 5), 1))
    if k == 2:
        return repr(rng.choice(["", "ab", "hello", "42", "x y", "7"]))
    if k == 3:
        return repr([rng.randint(0, 9) for _ in range(rng.randint(0, 3))])
    if k == 4:
        keys = rng.sample(["a", "b", "k"], rng.randint(0, 2))
        return "{" + ", ".join(f"{k!r}: {rng.randint(0, 5)}" for k in keys) + "}"
    if k == 5:
        return "None"
    return repr(rng.choice([True, False]))


def expr(rng, env):
    """An expression over the names bound so far (or literals)."""
    v = lambda: rng.choice(env) if env and rng.random() < 0.75 else literal(rng)
    k = rng.randrange(12)
    if k <= 3:
        return f"{v()} {rng.choice(['+', '-', '*', '/', '//', '%'])} {v()}"
    if k == 4:
        return f"{v()}[{rng.choice([str(rng.randint(-2, 3)), repr('a'), repr('k')])}]"
    if k == 5:
        return f"{v()}[{rng.randint(0, 2)}:{rng.randint(1, 4)}]"
    if k == 6:
        return f"len({v()})"
    if k == 7:
        return f"{rng.choice(['int', 'str', 'float', 'list'])}({v()})"
    if k == 8:
        m = rng.choice([".upper()", ".split()", ".append(1)", ".get('a')", ".keys()", ".pop()", ".strip()", ".index(1)"])
        return f"{v()}{m}"
    if k == 9:
        return f"{v()} {rng.choice(['<', '>', '==', 'in'])} {v()}"
    if k == 10:
        return f"sum({v()})"
    return v()


SAFE = {"len": len, "int": int, "str": str, "float": float, "list": list, "sum": sum}


def runs(code):
    """In-process check used only while BUILDING a snippet (the label itself comes from a fresh
    subprocess). Restricted builtins; the code is our own grammar, never outside input."""
    try:
        with warnings.catch_warnings():               # 3.14 warns at compile time on `3[0]`
            warnings.simplefilter("ignore")
            exec(compile(code, "<s>", "exec"), {"__builtins__": SAFE})
        return True
    except Exception:
        return False


def statement(rng, env):
    name = rng.choice(NAMES)
    if env and rng.random() < 0.2:
        return f"{rng.choice(env)} {rng.choice(['+=', '-=', '*='])} {expr(rng, env)}", None
    return f"{name} = {expr(rng, env) if env else literal(rng)}", name


def snippet(rng):
    """Every statement but the LAST is kept only if the prefix still runs, so whether a snippet
    raises is decided by its final line in the context of the earlier ones - not by how many
    risky operations it happens to contain (the first version's raise rate tracked length)."""
    env, lines = [], []
    for _ in range(rng.randint(2, 4)):
        for _ in range(60):
            st, name = statement(rng, env)
            if len(st) < 60 and runs("\n".join(lines + [st])):
                lines.append(st)
                if name and name not in env:
                    env.append(name)
                break
    st, _ = statement(rng, env)
    return "\n".join(lines + [st])


RUNNER = r"""
import sys, json, warnings
warnings.simplefilter('ignore')
src = sys.stdin.read()
safe = {k: __builtins__.__dict__[k] if hasattr(__builtins__, '__dict__') else __builtins__[k]
        for k in ('len', 'int', 'str', 'float', 'list', 'sum', 'True', 'False', 'None')}
try:
    exec(compile(src, '<snippet>', 'exec'), {'__builtins__': safe})
    print(json.dumps({'raises': False, 'exc': None}))
except Exception as e:
    print(json.dumps({'raises': True, 'exc': type(e).__name__}))
"""


def label(code):
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER], input=code, capture_output=True, text=True, timeout=5)
        r = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        r = {"raises": True, "exc": "Timeout"}
    return {"code": code, **r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(0)
    codes = list(dict.fromkeys(snippet(rng) for _ in range(a.n * 2)))[: a.n]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(label, codes, chunksize=32))
    Path("results").mkdir(exist_ok=True)
    with open("results/snippets.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    print(f"{n} distinct snippets, raise rate {sum(r['raises'] for r in rows) / n:.3f}")
    from collections import Counter
    print(Counter(r["exc"] for r in rows).most_common(8))

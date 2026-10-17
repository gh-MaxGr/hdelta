"""Pre-compute, for every mutant on disk, the source of the function it lives in AFTER the
mutation - the context an LLM would need to use what it knows about code. Offline: the mutant
is rebuilt with mutgen.mutant(); no test is re-run. Writes results/mutants/context/<module>.json
mapping site -> text (at most `WINDOW` lines around the changed line).

    .venv/Scripts/python.exe mut_context.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from mutgen import MODULES, STDLIB, mutant

WINDOW = 30
OUT = Path("results/mutants/context")


def enclosing(tree, path):
    """The FunctionDef/ClassDef node at dotted `path`, or None for module level."""
    if path == "<module>":
        return None
    node = tree
    for name in path.split("."):
        nxt = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == name:
                nxt = child
                break
        if nxt is None:
            return None
        node = nxt
    return node


def context_for(src_mut, func, changed_line):
    lines = src_mut.splitlines()
    node = enclosing(ast.parse(src_mut), func)
    lo, hi = (node.lineno, node.end_lineno) if node is not None else (changed_line - 5, changed_line + 5)
    lo, hi = max(1, lo), min(len(lines), hi)
    if hi - lo + 1 > WINDOW:                        # keep the window centred on the change
        lo = max(lo, changed_line - WINDOW // 2)
        hi = min(hi, lo + WINDOW - 1)
    return "\n".join(lines[lo - 1:hi])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path("results/mutants").glob("*.jsonl")):
        module = f.stem
        if module not in MODULES:
            continue
        src = (STDLIB / f"{module}.py").read_text(encoding="utf-8")
        recs = [json.loads(l) for l in f.open(encoding="utf-8")]
        ctx = {}
        for r in recs:
            src_mut, _ = mutant(src, r["site"])
            # the changed line in the UNPARSED text: first '+' line of the diff hunk
            hunk = r["diff"].splitlines()
            start = int(hunk[0].split("+")[1].split(",")[0]) if hunk and hunk[0].startswith("@@") else 1
            off = next((i for i, l in enumerate(l for l in hunk[1:] if not l.startswith("-")) if l.startswith("+")), 0)
            ctx[r["site"]] = context_for(src_mut, r["func"], start + off)
        json.dump(ctx, open(OUT / f"{module}.json", "w", encoding="utf-8"))
        print(f"{module}: {len(ctx)} contexts, mean {sum(len(c.splitlines()) for c in ctx.values()) / max(1, len(ctx)):.1f} lines",
              flush=True)


if __name__ == "__main__":
    main()

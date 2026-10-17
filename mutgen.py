"""Mutation-testing data from the local CPython standard library - no downloads.

For each module, every mutable site in its syntax tree gets ONE mutation (a flipped operator,
a comparison, a constant off by one, `return x` -> `return None`, a dropped `not`). The mutant
replaces the module in a sandbox directory placed first on sys.path, and the module's own
stdlib test file runs against it in a fresh process. Label: KILLED (any test fails or errors,
the import breaks, or it times out) or SURVIVED.

This is the "will the test suite catch this change?" question from the agent-loop use case,
and it is built to defeat a reason cache: every mutant is unique, the verifier's reasons are
tracebacks (prose), and generalising to held-out FUNCTIONS needs some understanding of the
code, which is the one thing an LLM brings that an MLP on cells or a lookup table does not.

Bytecode caching is OFF (PYTHONDONTWRITEBYTECODE): two mutants written within the same second
with the same size could otherwise reuse a stale .pyc and be scored as the wrong code.

    .venv/Scripts/python.exe mutgen.py                  # all modules, 10 workers
    .venv/Scripts/python.exe mutgen.py --limit 20 colorsys   # smoke
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

STDLIB = Path(sys.base_prefix) / "Lib"
MODULES = {"fractions": "test_fractions", "colorsys": "test_colorsys", "shlex": "test_shlex",
           "calendar": "test_calendar", "statistics": "test_statistics",
           "configparser": "test_configparser", "ipaddress": "test_ipaddress", "graphlib": "test_graphlib",
           "optparse": "test_optparse", "plistlib": "test_plistlib", "getopt": "test_getopt",
           "netrc": "test_netrc", "sched": "test_sched", "filecmp": "test_filecmp", "quopri": "test_quopri",
           "pickletools": "test_pickletools", "base64": "test_base64", "tabnanny": "test_tabnanny"}
TIMEOUT = {"statistics": 90}
OUT = Path("results/mutants")

BINOP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
         ast.FloorDiv: ast.Div, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult,
         ast.LShift: ast.RShift, ast.RShift: ast.LShift, ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd}
CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
       ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}


class Mutator(ast.NodeTransformer):
    """Walk in a fixed order; count eligible sites; mutate only site number `target`.
    With target=None it just counts and records each site's description."""

    def __init__(self, target=None):
        self.target, self.i, self.sites, self.stack = target, 0, [], []
        self.done = None

    def _site(self, kind, node):
        k = self.i
        self.i += 1
        self.sites.append({"op": kind, "func": ".".join(self.stack) or "<module>",
                           "line": getattr(node, "lineno", 0)})
        return k == self.target

    def _scoped(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit_Expr(self, node):                 # skip docstrings and bare constants
        if isinstance(node.value, ast.Constant):
            return node
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if type(node.op) in BINOP and self._site(f"binop {type(node.op).__name__}", node):
            node.op = BINOP[type(node.op)]()
            self.done = self.sites[-1]
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if type(node.op) in BINOP and self._site(f"augassign {type(node.op).__name__}", node):
            node.op = BINOP[type(node.op)]()
            self.done = self.sites[-1]
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in CMP and \
                self._site(f"compare {type(node.ops[0]).__name__}", node):
            node.ops = [CMP[type(node.ops[0])]()]
            self.done = self.sites[-1]
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self._site(f"boolop {type(node.op).__name__}", node):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.done = self.sites[-1]
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._site("drop not", node):
            self.done = self.sites[-1]
            return node.operand
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool) \
                and self._site("constant+1", node):
            self.done = self.sites[-1]
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        if node.value is not None and not (isinstance(node.value, ast.Constant) and node.value.value is None) \
                and self._site("return None", node):
            self.done = self.sites[-1]
            return ast.copy_location(ast.Return(value=ast.Constant(None)), node)
        return node


def sites(src):
    m = Mutator()
    m.visit(ast.parse(src))
    return m.sites


def mutant(src, k):
    m = Mutator(target=k)
    tree = m.visit(ast.parse(src))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), m.done


RUNNER = r"""
import sys, json, importlib, unittest
sys.dont_write_bytecode = True
sandbox, module, test = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, sandbox)
try:
    mod = importlib.import_module(module)
    assert mod.__file__.startswith(sandbox), mod.__file__
    suite = unittest.defaultTestLoader.loadTestsFromName("test." + test)
    res = unittest.TestResult()
    suite.run(res)
    out = {"run": res.testsRun, "fail": [t.id() for t, _ in res.failures] + [t.id() for t, _ in res.errors]}
except BaseException as e:
    out = {"run": 0, "fail": ["<import or load error>"], "exc": repr(e)[:200]}
print("@@RESULT@@" + json.dumps(out))
"""


def run_one(module, test, k, src_mut, timeout):
    with tempfile.TemporaryDirectory() as sb:
        Path(sb, module + ".py").write_text(src_mut, encoding="utf-8")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER, sb, module, test], capture_output=True,
                               text=True, timeout=timeout, env=env)
            line = next((l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
            res = json.loads(line[len("@@RESULT@@"):]) if line else {"run": 0, "fail": ["<crash>"]}
            timed_out = False
        except subprocess.TimeoutExpired:
            res, timed_out = {"run": 0, "fail": ["<timeout>"]}, True
    return k, res, timed_out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for module in (a.modules or list(MODULES)):
        test = MODULES[module]
        src = (STDLIB / f"{module}.py").read_text(encoding="utf-8")
        base = ast.unparse(ast.parse(src))
        # the UNMUTATED module must pass its own tests from the sandbox, or every label is noise
        _, res0, to0, sec0 = run_one(module, test, -1, base, TIMEOUT.get(module, 30))
        assert not res0["fail"] and not to0 and res0["run"] > 0, (module, res0)
        todo = list(range(len(sites(src))))[: a.limit]
        out_f = OUT / f"{module}.jsonl"
        done = set()
        if out_f.exists():
            done = {json.loads(l)["site"] for l in out_f.open(encoding="utf-8")}
        print(f"{module}: {len(todo)} sites, clean run {res0['run']} tests in {sec0:.1f}s, "
              f"{len(done)} already done", flush=True)
        jobs, info = {}, {}
        base_lines = base.splitlines()
        with ProcessPoolExecutor(a.workers) as ex, out_f.open("a", encoding="utf-8") as fo:
            for k in todo:
                if k in done:
                    continue
                src_mut, site = mutant(src, k)
                diff = "\n".join(l for l in difflib.unified_diff(base_lines, src_mut.splitlines(),
                                                                 lineterm="", n=2)
                                 if not l.startswith(("---", "+++")))
                info[k] = (site, diff)
                jobs[ex.submit(run_one, module, test, k, src_mut, TIMEOUT.get(module, 30))] = k
            n = 0
            for f in as_completed(jobs):
                k, res, timed_out, sec = f.result()
                site, diff = info[k]
                rec = {"module": module, "test": test, "site": k, "op": site["op"], "func": site["func"],
                       "line": site["line"], "diff": diff, "killed": bool(res["fail"]),
                       "n_fail": len(res["fail"]), "failing": res["fail"][:20], "tests_run": res["run"],
                       "timeout": timed_out, "seconds": round(sec, 2)}
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                n += 1
                if n % 100 == 0:
                    print(f"  {module}: {n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()

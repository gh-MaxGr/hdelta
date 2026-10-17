"""Will the test suite catch this code change? Mutants of CPython stdlib modules (mutgen.py).

The agent-loop question in its smallest form. The tool is the module's own test file, run
against the mutant; that is the expensive call (seconds per mutant for statistics). Why a reason
cache fails here: every mutant is unique, the tool's reason is a list of failing tests or
tracebacks, and the split is by TOP-LEVEL FUNCTION OR CLASS, so a held-out mutant lives in code
the learner never saw - a cache keyed on the function has nothing to recall.

Labels are lopsided (~87 % of mutants are caught), and a constant "yes" would look good, so
`balance` sets the share of caught mutants drawn - 0.5 by default for the stream AND the probes.
Survivors are the "exception" branch, so probe metrics split caught / survived.

    Will the module's own test suite catch this code change?

    Change in statistics, function fmean:
    @@ -290,4 +290,4 @@
    -    return total / n
    +    return total * n
    Caught: yes
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .base import Task

LABELS = ("yes", "no")


class MutantTask(Task):
    name = "mutants"
    header = "Will the module's own test suite catch this code change?\n\n"
    input_label = "Change"
    answer_label = "Caught"
    rule_noun = "suite"

    def __init__(self, rules: Dict[str, dict], data_dir: str = "results/mutants",
                 modules=None, heldout_mod: int = 4, balance: float = 0.5,
                 split: str = "train", max_diff_lines: int = 12, max_line_chars: int = 110,
                 context: bool = False, context_dir: str = "results/mutants/context"):
        super().__init__(rules)
        self.heldout_mod = int(heldout_mod)
        self.balance = float(balance)
        self.split = split
        self.max_diff_lines, self.max_line_chars = int(max_diff_lines), int(max_line_chars)
        self.rec: Dict[str, dict] = {}
        for f in sorted(Path(data_dir).glob("*.jsonl")):
            if modules and f.stem not in modules:
                continue
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                self.rec[f"{r['module']}:{r['site']}"] = r
        self.pool: Dict[Tuple[str, str], List[str]] = {}
        for k, r in self.rec.items():
            self.pool.setdefault((self.bucket(k), LABELS[0] if r["killed"] else LABELS[1]), []).append(k)
        for v in self.pool.values():
            v.sort()
        self.calls = 0
        # context=True also shows the mutated function (mut_context.py), so a model can use what
        # it knows about code; off by default, which keeps every earlier prompt byte-identical
        self.context = bool(context)
        self.ctx: Dict[str, Dict[str, str]] = {}
        if self.context:
            for m in {r["module"] for r in self.rec.values()}:
                self.ctx[m] = json.load(open(Path(context_dir) / f"{m}.json", encoding="utf-8"))

    def bucket(self, key: str) -> str:
        r = self.rec[key]
        group = f"{r['module']}:{r['func'].split('.')[0]}"
        h = int(hashlib.md5(group.encode()).hexdigest()[:8], 16)
        return "heldout" if h % self.heldout_mod == 0 else "train"

    def stats(self) -> dict:
        return {f"{s}/{l}": len(v) for (s, l), v in sorted(self.pool.items())}

    # ---- Task interface ---------------------------------------------------------------
    def sample_input(self, rule: str, rng: random.Random) -> str:
        split = getattr(self, "split", None) or "train"
        label = LABELS[0] if rng.random() < self.balance else LABELS[1]
        return rng.choice(self.pool[(split, label)])

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        self.calls += 1
        r = self.rec[input_str]
        if r["killed"]:
            why = ("timed out" if r["timeout"] else
                   f"{r['n_fail']} test(s) fail, e.g. {r['failing'][0].split('.')[-1]}")
            return [why], LABELS[0]
        return [f"all {r['tests_run']} tests pass"], LABELS[1]

    def n_steps(self, rule: str) -> int:
        return 1

    def max_answer_tokens(self) -> int:
        return 3

    def describe(self, rule: str) -> str:
        return ("The change is applied to the module and the module's own standard-library test "
                "file is run. Caught = at least one test fails or errors.")

    def tool_calls(self) -> int:
        return self.calls

    def is_valid_answer(self, rule: str, ans: str) -> bool:
        return ans.strip() in LABELS

    def is_exception(self, rule: str, input_str: str) -> bool:
        return not self.rec[input_str]["killed"]           # survivors are the minority branch

    def _input_line(self, pz) -> str:
        r = self.rec[pz.input_str]
        lines = [l[: self.max_line_chars] for l in r["diff"].splitlines()][: self.max_diff_lines]
        text = f"{self.input_label} in {r['module']}, {r['func']}:\n" + "\n".join(lines)
        if self.context:
            code = [l[: self.max_line_chars] for l in self.ctx[r["module"]][str(r["site"])].splitlines()]
            text += "\nCode after the change:\n" + "\n".join(code)
        return text

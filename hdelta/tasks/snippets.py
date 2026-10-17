"""Does this Python snippet raise? Labels from EXECUTING the snippet (snipgen.py).

The quadrant the earlier tasks never tested: the answer is not a small table, AND the
pretrained model already has a prior (Python semantics). Connect-Four and mutants both started
the frozen model at chance, so the fast weights were always learning from scratch; here the
question is whether they improve a model that is already partly right - the one niche a tiny
learner on tokens cannot enter, because it has no prior to start from.

Split: md5 of the code, 1 in `heldout_mod` held out - every snippet is unique, so held-out
snippets have never been seen. Draws are BALANCED (raise rate ~0.65 natural); snippets that fail
to even compile are dropped (a SyntaxError is not a semantics question).

    Does this Python snippet raise an exception when run?

    Code:
    b = {'k': 1}
    n = b['k']
    b *= n + 10
    Raises: yes
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .base import Task

LABELS = ("yes", "no")


class SnippetTask(Task):
    name = "snippets"
    header = "Does this Python snippet raise an exception when run?\n\n"
    input_label = "Code"
    answer_label = "Raises"
    rule_noun = "snippet"

    def __init__(self, rules: Dict[str, dict], data: str = "results/snippets.jsonl",
                 heldout_mod: int = 4, balance: float = 0.5, split: str = "train"):
        super().__init__(rules)
        self.heldout_mod, self.balance, self.split = int(heldout_mod), float(balance), split
        self.rec: Dict[str, dict] = {}
        for line in open(data, encoding="utf-8"):
            r = json.loads(line)
            if r["exc"] != "SyntaxError":
                self.rec[hashlib.md5(r["code"].encode()).hexdigest()[:12]] = r
        self.pool: Dict[Tuple[str, str], List[str]] = {}
        for k, r in self.rec.items():
            b = "heldout" if int(k, 16) % self.heldout_mod == 0 else "train"
            self.pool.setdefault((b, LABELS[0] if r["raises"] else LABELS[1]), []).append(k)
        for v in self.pool.values():
            v.sort()
        self.calls = 0

    def stats(self) -> dict:
        return {f"{s}/{l}": len(v) for (s, l), v in sorted(self.pool.items())}

    def sample_input(self, rule: str, rng: random.Random) -> str:
        split = getattr(self, "split", None) or "train"
        label = LABELS[0] if rng.random() < self.balance else LABELS[1]
        return rng.choice(self.pool[(split, label)])

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        self.calls += 1
        r = self.rec[input_str]
        return ([f"raises {r['exc']}"], LABELS[0]) if r["raises"] else (["runs to the end"], LABELS[1])

    def n_steps(self, rule: str) -> int:
        return 1

    def max_answer_tokens(self) -> int:
        return 3

    def describe(self, rule: str) -> str:
        return "The snippet is executed with only len/int/str/float/list/sum available."

    def tool_calls(self) -> int:
        return self.calls

    def is_valid_answer(self, rule: str, ans: str) -> bool:
        return ans.strip() in LABELS

    def is_exception(self, rule: str, input_str: str) -> bool:
        return not self.rec[input_str]["raises"]           # the minority class: runs cleanly

    def _input_line(self, pz) -> str:
        return f"{self.input_label}:\n{self.rec[pz.input_str]['code']}"

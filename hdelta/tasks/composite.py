"""Several single-rule tasks as ONE multi-rule task, for sequential (continual) streams.

Every stationary comparison so far (Connect-Four, mutants, snippets) favoured LoRA or a tiny
learner. Decay and homeostasis were built for the opposite case - several skills arriving in
sequence, where one must be learned without erasing another - so that is the last place the
fast weights could have a niche. This task lets a stream go snippets -> mutants -> snippets with
probes on BOTH skills at every probe (retention and acquisition read off the same run).

Each rule name maps to a sub-task config; puzzles, prompts (each sub-task keeps its own
header), answers, splits and tool calls are all delegated to that sub-task.

    task:
      name: composite
      rules:
        snip: {task: {name: snippets, heldout_mod: 4, balance: 0.5, rules: {snip: {}}}}
        mut:  {task: {name: mutants, modules: [...], rules: {mut: {}}}}
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .base import Puzzle, Task


class CompositeTask(Task):
    name = "composite"

    def __init__(self, rules: Dict[str, dict], make):
        super().__init__(rules)
        self.sub = {r: make(spec["task"]) for r, spec in rules.items()}
        self.inner = {r: t.rule_names()[0] for r, t in self.sub.items()}
        self._split = "train"

    # the experiment flips `split` for probe sets: pass it to every sub-task
    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, v):
        self._split = v
        for t in getattr(self, "sub", {}).values():
            t.split = v

    def sample_input(self, rule: str, rng: random.Random) -> str:
        return self.sub[rule].sample_input(self.inner[rule], rng)

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        return self.sub[rule].solve(self.inner[rule], input_str)

    def answer_under(self, rule: str, input_str: str) -> str:
        """The probe asks every rule for its answer on every OTHER rule's inputs (to catch one
        rule's mapping leaking into another). A snippet key means nothing to the mutant task,
        so a foreign input gets an answer no model output can match."""
        try:
            return self.solve(rule, input_str)[1]
        except (KeyError, ValueError, AssertionError):
            return "<n/a>"

    def sample(self, rule: str, rng: random.Random) -> Puzzle:
        x = self.sample_input(rule, rng)
        steps, ans = self.solve(rule, x)
        return Puzzle(x, steps, ans, rule)

    def make_pair(self, pz: Puzzle, fmt: str, k_steps=None, shots=(), rule_text: bool = False):
        t = self.sub[pz.rule]
        inner = Puzzle(pz.input_str, pz.steps, pz.answer, self.inner[pz.rule], dict(pz.meta))
        return t.make_pair(inner, fmt, k_steps=k_steps,
                           shots=[Puzzle(s.input_str, s.steps, s.answer, self.inner[pz.rule], dict(s.meta))
                                  for s in shots], rule_text=rule_text)

    def n_steps(self, rule: str) -> int:
        return self.sub[rule].n_steps(self.inner[rule])

    def max_answer_tokens(self) -> int:
        return max(t.max_answer_tokens() for t in self.sub.values())

    def describe(self, rule: str) -> str:
        return self.sub[rule].describe(self.inner[rule])

    def tool_calls(self) -> int:
        return sum(t.tool_calls() for t in self.sub.values())

    def is_valid_answer(self, rule: str, ans: str) -> bool:
        return self.sub[rule].is_valid_answer(self.inner[rule], ans)

    def is_exception(self, rule: str, input_str: str) -> bool:
        return self.sub[rule].is_exception(self.inner[rule], input_str)

    def matches(self, generated: str, answer: str, rule: str, input_str=None) -> bool:
        return self.sub[rule].matches(generated, answer, self.inner[rule], input_str)

    def parse_generation(self, text: str):
        return next(iter(self.sub.values())).parse_generation(text)

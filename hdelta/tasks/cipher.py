"""A changing multi-step symbol cipher.

A rule set is an ordered list of primitive operations; each operation is one
scratchpad step.  Rule sets are swapped between phases ("the rules change").

The alphabet is either letters (a, b, c, ...) or digits (0, 1, 2, ...); symbols are
space-separated so each is one token.  Primitives operate on alphabet indices:
    ["shift", k]   add k (mod alphabet size)
    ["mirror"]     reflect the alphabet (index i -> N-1-i)
    ["reverse"]    reverse the string
    ["rotate", k]  rotate left by k
    ["swap_pairs"] swap adjacent pairs (odd tail stays)
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .base import Task


def _apply(name: str, args: list, idx: List[int], n: int) -> List[int]:
    if name == "shift":
        return [(i + int(args[0])) % n for i in idx]
    if name == "mirror":
        return [n - 1 - i for i in idx]
    if name == "reverse":
        return idx[::-1]
    if name == "rotate":
        k = int(args[0]) % max(1, len(idx))
        return idx[k:] + idx[:k]
    if name == "swap_pairs":
        out = list(idx)
        for i in range(0, len(out) - 1, 2):
            out[i], out[i + 1] = out[i + 1], out[i]
        return out
    raise ValueError(f"unknown cipher op {name}")


class CipherTask(Task):
    name = "cipher"

    def __init__(self, rules: Dict[str, List[list]], length=(4, 5), alphabet_size: int = 26,
                 alphabet: str = "letters", tagged: bool = False, tag_map=None):
        super().__init__(rules)
        # tag_map lets the prompt's cue be made deliberately confusable (K1..K8, which
        # share a token and differ in one character) instead of distinctive (A7, M6, ...)
        self.tag_map = dict(tag_map) if tag_map else None
        self.length = tuple(length)
        self.n = int(alphabet_size)
        self.kind = alphabet
        self.tagged = bool(tagged)   # put the rule's name in the prompt (interleaved rules)
        if alphabet == "digits":
            assert self.n <= 10, "digit alphabet supports at most 10 symbols"
            self.alphabet = [str(i) for i in range(self.n)]
            self.header = "Apply the secret cipher rule to the input digits.\n\n"
            self.unit = "digit"
        else:
            self.alphabet = [chr(97 + i) for i in range(self.n)]
            self.header = "Apply the secret cipher rule to the input letters.\n\n"
            self.unit = "letter"
        self.index = {c: i for i, c in enumerate(self.alphabet)}

    def tag(self, rule: str) -> str:
        return self.tag_map.get(rule, rule) if self.tag_map else rule

    def sample_input(self, rule: str, rng: random.Random) -> str:
        L = rng.randint(self.length[0], self.length[1])
        return " ".join(rng.choice(self.alphabet) for _ in range(L))

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        idx = [self.index[c] for c in input_str.split()]
        steps: List[str] = []
        for op in self.rules[rule]:
            idx = _apply(op[0], list(op[1:]), idx, self.n)
            steps.append(" ".join(self.alphabet[i] for i in idx))
        return steps, steps[-1]

    def n_steps(self, rule: str) -> int:
        return len(self.rules[rule])

    def _table(self, name: str, args: list) -> str:
        return ", ".join(f"{c}->{self.alphabet[_apply(name, args, [i], self.n)[0]]}"
                         for i, c in enumerate(self.alphabet))

    def describe(self, rule: str) -> str:
        u = self.unit
        parts = []
        if self.kind == "digits":
            parts.append(f"The {u}s are 0 to {self.n - 1}; arithmetic wraps around (so {self.n - 1} + 1 = 0).")
        else:
            parts.append(f"The alphabet is {' '.join(self.alphabet)} and it wraps around.")
        for i, op in enumerate(self.rules[rule]):
            name, args = op[0], list(op[1:])
            if name == "shift":
                k = int(args[0])
                if self.kind == "digits":
                    d = f"add {k} to each {u}, wrapping around after {self.n - 1} ({self._table(name, args)})"
                else:
                    d = f"shift each {u} forward by {k} ({self._table(name, args)})"
            elif name == "mirror":
                if self.kind == "digits":
                    d = f"replace each {u} d with {self.n - 1} - d ({self._table(name, args)})"
                else:
                    d = f"mirror each {u} in the alphabet ({self._table(name, args)})"
            elif name == "reverse":
                d = f"reverse the order of the {u}s"
            elif name == "rotate":
                d = f"rotate the {u}s left by {int(args[0])}"
            elif name == "swap_pairs":
                d = f"swap each adjacent pair of {u}s"
            else:
                d = name
            parts.append(f"Step {i + 1}: {d}.")
        parts.append("The answer is the result of the last step.")
        return " ".join(parts)

    def max_answer_tokens(self) -> int:
        return self.length[1] + 2

"""Common interface for procedural logic puzzles with changeable rules."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass
class Puzzle:
    input_str: str
    steps: List[str]
    answer: str
    rule: str
    meta: Dict = field(default_factory=dict)


class Task:
    """A puzzle family.  Subclasses define ``solve`` (steps + answer for a rule),
    ``sample`` (draw an input) and the textual labels.

    Formats produced by ``make_pair`` (prompt, response):

      scratchpad : "<header><shots>Input: x\\n"            -> "Step 1: ..\\nStep 2: ..\\nAnswer: a\\n"
      direct     : "<header><shots>Input: x\\nAnswer:"     -> " a\\n"
      partial(k) : "<header>Input: x\\nStep 1: ..\\n(k steps)Answer:" -> " a\\n"

    Scratchpad = excitation-based, step-by-step ("thinking out loud").
    Direct     = the compiled path: the answer must come straight from the input.
    """
    name = "task"
    header = ""
    input_label = "Input"
    answer_label = "Answer"
    # What a tagged rule is called in the prompt ("Rule for cipher K3: ...").  The default
    # keeps the exact prompt wording of the closed cipher runs; new tasks override it.
    rule_noun = "cipher"

    def __init__(self, rules: Dict[str, object]):
        self.rules = rules

    # ---- to implement --------------------------------------------------------
    def rule_names(self) -> List[str]:
        return list(self.rules)

    def sample_input(self, rule: str, rng: random.Random) -> str:
        raise NotImplementedError

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        raise NotImplementedError

    def n_steps(self, rule: str) -> int:
        raise NotImplementedError

    def max_answer_tokens(self) -> int:
        raise NotImplementedError

    def describe(self, rule: str) -> str:
        """Natural-language statement of the rule (the 'instructed' slow path)."""
        raise NotImplementedError

    def tool_calls(self) -> int:
        """Authoritative-tool calls made so far.  Tasks whose rule is a pure function with no
        cost to evaluate have no tool to count, so they report 0 - but the method must exist,
        because the probe records it for every task."""
        return 0

    def simulated_ms(self) -> float:
        return 0.0

    def is_exception(self, rule: str, input_str: str) -> bool:
        """Does this input exercise a rare branch of the rule?

        Random few-shot sampling gives the tail near-zero coverage: measured on the
        workspace task, the teacher scored 0.142 on exception inputs against 0.882 on
        ordinary ones, so only ~8 correct demonstrations per rule ever reached
        consolidation. Tasks that have a tail should say so, and the stream can then
        guarantee the context window contains one."""
        return False

    # ---- generic --------------------------------------------------------------
    def sample(self, rule: str, rng: random.Random) -> Puzzle:
        x = self.sample_input(rule, rng)
        steps, ans = self.solve(rule, x)
        return Puzzle(x, steps, ans, rule)

    def answer_under(self, rule: str, input_str: str) -> str:
        return self.solve(rule, input_str)[1]

    def is_valid_answer(self, rule: str, ans: str) -> bool:
        """Is this a legitimate answer for the task at all, right or wrong?

        Only a legitimate wrong answer may be punished. An unparseable string is not a bad
        decision, it is the absence of one, and a negative write on it lands on arbitrary
        tokens - in the world task that fed a runaway that took both signed runs from ~1 % to
        100 % invalid output within two fifths of the stream. Tasks with a closed answer
        vocabulary should override this with membership.
        """
        return bool(ans and ans.strip())

    def matches(self, generated: str, answer: str, rule: str,
                input_str: str | None = None) -> bool:
        """Is this generated output correct?

        Exact string match for every task whose answer is a single label. Tasks whose
        answer has many valid spellings - a shell command - override this so scoring
        measures the property that matters rather than transcription. Tasks scored by
        OUTCOME rather than by a target string need the input too, since "was this move
        any good" is a question about the situation, not about the answer key.
        """
        return generated == answer

    def tag(self, rule: str) -> str:
        """Short name the prompt uses to select a rule (only when ``tagged``)."""
        return rule

    def _input_line(self, pz: Puzzle) -> str:
        """The input line.  When several rules are interleaved the prompt must say which
        one applies, so the compiled path has to map (tag, input) -> answer."""
        if getattr(self, "tagged", False):
            return f"{self.input_label} [{self.tag(pz.rule)}]: {pz.input_str}"
        return f"{self.input_label}: {pz.input_str}"

    def _solved_text(self, pz: Puzzle, fmt: str) -> str:
        if fmt == "scratchpad":
            body = "".join(f"Step {i + 1}: {s}\n" for i, s in enumerate(pz.steps))
            return f"{self._input_line(pz)}\n{body}{self.answer_label}: {pz.answer}\n"
        return f"{self._input_line(pz)}\n{self.answer_label}: {pz.answer}\n"

    def make_pair(self, pz: Puzzle, fmt: str, k_steps: int | None = None,
                  shots: Sequence[Puzzle] = (), rule_text: bool = False) -> Tuple[str, str]:
        """rule_text=True prepends the rule statement: the slow path becomes *execution*
        of a stated procedure rather than induction from examples."""
        shot_fmt = "scratchpad" if fmt in ("scratchpad", "free") else "direct"
        prefix = self.header
        if rule_text:
            who = f" for {self.rule_noun} {self.tag(pz.rule)}" if getattr(self, "tagged", False) else ""
            prefix += f"Rule{who}: {self.describe(pz.rule)}\n\n"
        prefix += "".join(self._solved_text(s, shot_fmt) + "\n" for s in shots)
        line = self._input_line(pz)
        if fmt == "scratchpad":
            prompt = f"{prefix}{line}\n"
            body = "".join(f"Step {i + 1}: {s}\n" for i, s in enumerate(pz.steps))
            response = f"{body}{self.answer_label}: {pz.answer}\n"
            return prompt, response
        if fmt == "direct":
            prompt = f"{prefix}{line}\n{self.answer_label}:"
            return prompt, f" {pz.answer}\n"
        if fmt == "free":
            # open-ended: nothing after the input; the model decides whether to think first
            prompt = f"{prefix}{line}\n"
            return prompt, f"{self.answer_label}: {pz.answer}\n"
        if fmt == "partial":
            k = 0 if k_steps is None else k_steps
            body = "".join(f"Step {i + 1}: {s}\n" for i, s in enumerate(pz.steps[:k]))
            prompt = f"{prefix}{line}\n{body}{self.answer_label}:"
            return prompt, f" {pz.answer}\n"
        raise ValueError(fmt)

    def parse_answer(self, text: str) -> str:
        return text.split("\n")[0].strip()

    def parse_generation(self, text: str) -> Tuple[str, int]:
        """From free-form output return (answer, number of Step lines emitted before it)."""
        steps = 0
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Step"):
                steps += 1
                continue
            if ":" in line:             # "Answer: ...", or any other label the model chose
                return line.split(":", 1)[1].strip(), steps
            return line, steps          # answered without a label
        return "", steps

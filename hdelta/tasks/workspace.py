"""Resolving a source file to its build target, against a synthetic workspace.

The "expensive tool" is :class:`WorkspaceResolver` - a hermetic simulator of a
build-graph query (``bazel query`` and friends).  It is authoritative, exact and
deterministic, so it serves as teacher and as verifier, and it ships with the
repo: no live workspace, no proprietary API, nothing for a reader to supply.

A rule is one workspace layout: an ordered pattern table, last match wins, in the
manner of CODEOWNERS.  Changing the rule is a repo reorg.

Target labels are drawn from a pool that is *independent of the directory names*,
so the mapping cannot be guessed by echoing the path.  Real build graphs behave
this way (a file under ``libs/auth`` may well belong to ``//third_party:shim``);
here it also keeps the zero-shot baseline at the floor, which is what gate 1 of
the deployment recipe asks for.
"""
from __future__ import annotations

import random
import re
import time
from typing import Dict, List, Sequence, Tuple

from .base import Task

# Deliberately confusable sibling packages: shared stem, near-identical suffix.
# Separating these is the point - they stress cue overlap rather than avoiding it.
_STEMS = ["core", "data", "net", "ui", "sync", "auth"]
_SUFFIXES = [("auth", "audit"), ("store", "stream"), ("client", "cluster"),
             ("view", "viewer"), ("queue", "query"), ("index", "ingest")]
_AREAS = ["apps", "libs", "svc"]
_FILE_STEMS = ["handler", "model", "client", "config", "utils", "session", "parser",
               "worker", "registry", "adapter", "schema", "loader", "buffer", "policy"]
_EXTS = [".py", ".go", ".ts", ".rs"]
# Label components share no vocabulary with the directory names above.
_LABEL_HEADS = ["pkg", "mod", "bundle", "unit", "artifact", "component"]
_LABEL_TAILS = ["alpha", "beta", "gamma", "delta", "sigma", "omega", "kappa", "theta"]


def _glob_to_re(pat: str) -> re.Pattern:
    """Translate a CODEOWNERS-style glob to a regex.  ``**`` spans separators, ``*``
    does not."""
    out, i = [], 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            if pat[i:i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


class WorkspaceResolver:
    """The slow, authoritative tool.  Teacher and verifier both.

    ``latency_ms`` is what one real query would cost.  It is *accounted* on every
    call and only *slept* when ``sleep=True``: a training run makes thousands of
    calls, and sleeping through them would add half an hour of wall clock to
    produce a number that ``calls * latency_ms`` already gives exactly.  Turn the
    sleep on for an end-to-end demonstration, not for measurement.
    """

    def __init__(self, patterns: Sequence[Tuple[str, str]], latency_ms: float = 750.0,
                 sleep: bool = False):
        self.patterns = [(p, t) for p, t in patterns]
        self._compiled = [(_glob_to_re(p), t) for p, t in self.patterns]
        self.latency_ms = float(latency_ms)
        self.sleep = bool(sleep)
        self.calls = 0

    @property
    def simulated_ms(self) -> float:
        return self.calls * self.latency_ms

    def trace(self, path: str) -> List[Tuple[str, str]]:
        """Every pattern that matches, in table order.  The last one wins."""
        return [(p, t) for (rx, t), (p, _) in zip(self._compiled, self.patterns)
                if rx.match(path)]

    def resolve(self, path: str) -> str:
        self.calls += 1
        if self.sleep:
            time.sleep(self.latency_ms / 1000.0)
        hits = self.trace(path)
        return hits[-1][1] if hits else "//:unowned"


def build_workspace(seed: int, n_packages: int = 6, n_overrides: int = 2,
                    area: str | None = None):
    """Generate one workspace: (pattern table, list of package directories).

    The table is ordered general -> specific; the overrides at the end are what
    stop a 'just read the directory' heuristic from working.

    Pin ``area`` to the same value across rules when interleaving them: otherwise
    the top-level directory gives the workspace away and the prompt's tag carries
    no load, which is not the competition the capacity result was measured under.
    """
    rng = random.Random(seed)
    # One area for all packages, so the area-wide catch-all below matches every input
    # and "last match wins" always has something to resolve between.
    area = area or rng.choice(_AREAS)
    pairs = rng.sample(_SUFFIXES, k=(n_packages + 1) // 2)
    stems = rng.sample(_STEMS, k=(n_packages + 1) // 2)

    dirs: List[str] = []
    for stem, (a, b) in zip(stems, pairs):
        dirs.append(f"{area}/{stem}-{a}")
        dirs.append(f"{area}/{stem}-{b}")
    dirs = dirs[:n_packages]

    heads = rng.sample(_LABEL_HEADS, k=len(_LABEL_HEADS))
    tails = rng.sample(_LABEL_TAILS, k=len(_LABEL_TAILS))
    labels = [f"//{heads[i % len(heads)]}:{tails[i % len(tails)]}" for i in range(len(dirs))]
    rng.shuffle(labels)

    patterns: List[Tuple[str, str]] = [(f"{area}/**", f"//{heads[-1]}:default")]
    for d, lab in zip(dirs, labels):
        patterns.append((f"{d}/**", lab))
    # Overrides: a file-shaped pattern that cuts across packages and fires last.
    over_exts = rng.sample(_EXTS, k=n_overrides)
    for j, ext in enumerate(over_exts):
        patterns.append((f"**/*_test{ext}", f"//{heads[j]}:{tails[-(j + 1)]}"))
    return patterns, dirs


class WorkspaceResolutionTask(Task):
    name = "workspace"
    header = "Resolve the source file to its build target.\n\n"
    input_label = "File"
    answer_label = "Target"
    rule_noun = "workspace"

    def __init__(self, rules: Dict[str, dict], n_packages: int = 6, n_overrides: int = 2,
                 depth: Tuple[int, int] = (1, 2), latency_ms: float = 750.0,
                 sleep: bool = False, tagged: bool = False, tag_map=None,
                 exception_fraction=None):
        super().__init__(rules)
        self.exception_fraction = (None if exception_fraction is None
                                   else float(exception_fraction))
        self.depth = (int(depth[0]), int(depth[1]))
        self.tagged = bool(tagged)
        self.tag_map = dict(tag_map) if tag_map else None
        self.tables: Dict[str, List[Tuple[str, str]]] = {}
        self.dirs: Dict[str, List[str]] = {}
        self.tools: Dict[str, WorkspaceResolver] = {}
        for r, spec in rules.items():
            spec = spec or {}
            pats, dirs = build_workspace(int(spec.get("seed", 0)),
                                         int(spec.get("n_packages", n_packages)),
                                         int(spec.get("n_overrides", n_overrides)),
                                         area=spec.get("area"))
            self.tables[r], self.dirs[r] = pats, dirs
            self.tools[r] = WorkspaceResolver(pats, latency_ms=latency_ms, sleep=sleep)

    def tag(self, rule: str) -> str:
        return self.tag_map.get(rule, rule) if self.tag_map else rule

    # ---- Task interface -------------------------------------------------------
    def sample_input(self, rule: str, rng: random.Random) -> str:
        """Draw a path.  The filename space is far larger than the pattern space,
        so held-out inputs are genuinely unseen while the rule that governs them
        is not - which is what the generalisation probe needs.

        ``exception_fraction`` biases the draw towards inputs that fire an override.
        Natural frequency is ~15 %, which after the teacher's own error rate leaves the
        exception branch roughly 25 verified writes against 333 for the ordinary branch.
        Raising it is how we ask whether that 1:13 ratio is what starves the exception."""
        if self.exception_fraction is not None:
            want = rng.random() < self.exception_fraction
            for _ in range(200):
                x = self._draw_input(rule, rng)
                if self.is_exception(rule, x) == want:
                    return x
        return self._draw_input(rule, rng)

    def _draw_input(self, rule: str, rng: random.Random) -> str:
        d = rng.choice(self.dirs[rule])
        sub = [rng.choice(_FILE_STEMS) for _ in range(rng.randint(*self.depth) - 1)]
        stem = rng.choice(_FILE_STEMS)
        if rng.random() < 0.25:
            stem += "_test"
        return "/".join([d, *sub, stem + rng.choice(_EXTS)])

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        tool = self.tools[rule]
        hits = tool.trace(input_str)
        tool.calls += 1                      # the teacher pays for the call too
        if not hits:
            return ["no pattern matches"], "//:unowned"
        steps = [f"{p} matches -> {t}" for p, t in hits]
        steps.append(f"last match wins -> {hits[-1][1]}")
        return steps, hits[-1][1]

    def is_exception(self, rule: str, input_str: str) -> bool:
        """More than the area catch-all plus the package pattern matched, i.e. an
        override fired and last-match-wins actually has to be applied."""
        return len(self.tools[rule].trace(input_str)) > 2

    def n_steps(self, rule: str) -> int:
        return 4

    def describe(self, rule: str) -> str:
        table = "; ".join(f"{p} -> {t}" for p, t in self.tables[rule])
        return (f"The build table is: {table}. Check the file path against every "
                f"pattern in order; the target is the one from the last pattern "
                f"that matches.")

    def max_answer_tokens(self) -> int:
        return 16

    # ``//pkg:alpha`` contains a colon, so the generic parser would split it in the
    # wrong place when the model answers without a label.
    def parse_generation(self, text: str) -> Tuple[str, int]:
        steps = 0
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Step"):
                steps += 1
                continue
            if line.startswith("//"):
                return line, steps
            if ":" in line:
                return line.split(":", 1)[1].strip(), steps
            return line, steps
        return "", steps

    # ---- accounting -----------------------------------------------------------
    def tool_calls(self) -> int:
        return sum(t.calls for t in self.tools.values())

    def simulated_ms(self) -> float:
        return sum(t.simulated_ms for t in self.tools.values())

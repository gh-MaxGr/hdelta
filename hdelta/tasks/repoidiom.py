"""Repo idioms: the conventions that currently live in CLAUDE.md because the model
cannot produce them cold.

Every rule here is a real convention from this project's own instruction files, each one
written down after watching a model get it wrong repeatedly. That file is a hand-maintained
prosthesis for missing procedural memory: the cost is paid on every turn, in context, forever.
The question this task exists to answer is whether that cost can be paid ONCE, into weights.

The verifier is free and exact - a pure-Python check over the emitted command, the same
thing a linter does. It does not run anything. That matters twice: a wrong answer costs
nothing, and the check is available in any repo without inventing ground truth, which is
where every other candidate application died.

Success is the verifier going quiet with the context window empty. Not exact string match -
`matches()` accepts any command that satisfies the idiom, because "stops tripping the
linter" is the actual deliverable.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Sequence, Tuple

from .base import Puzzle, Task

# Pools are built from one homogeneous generator and then SPLIT, rather than hand-assigned.
# The first version had 14 markers and 18 scripts against 360 stream puzzles per rule, so
# every probe item had certainly been trained on: those arms measured a cached dictionary,
# not an idiom. Here the stream and the probes draw from DISJOINT halves, so a correct
# held-out answer requires transferring the pattern to a filler never seen in any position.
#
# Both halves come from the same generator and a seeded shuffle, so they are exchangeable -
# hand-picking the held-out names would confound "never seen" with "lexically odd".
_PREFIX = ["analyse", "probe", "check", "run", "audit", "calibrate", "aggregate",
           "summarise", "inspect", "verify"]
_STEM = ["variance", "poison", "ladder", "locus", "capacity", "drift", "gate", "monitor",
         "spectra", "decay", "energy", "routing", "cues", "exceptions", "seeds", "budget",
         "yield", "transfer", "recall", "margins"]


def _pools(seed: int = 20260921):
    import random as _r
    rng = _r.Random(seed)
    scripts = sorted({f"{a}_{b}.py" for a in _PREFIX for b in _STEM})
    names = sorted({f"{b}{sfx}" for b in _STEM for sfx in ("", "_s1", "_s2")})
    rng.shuffle(scripts)
    rng.shuffle(names)
    cut_s, cut_n = int(len(scripts) * 0.6), int(len(names) * 0.6)
    return {
        "train": {"scripts": scripts[:cut_s], "markers": names[:cut_n], "outs": names[:cut_n]},
        "heldout": {"scripts": scripts[cut_s:], "markers": names[cut_n:], "outs": names[cut_n:]},
    }


POOLS = _pools()
SCRIPTS = POOLS["train"]["scripts"]      # kept for callers that want the training pool
MARKERS = POOLS["train"]["markers"]
OUTS = POOLS["train"]["outs"]

FAMILIES = ("launch", "gate", "invoke")


class RepoIdiomTask(Task):
    name = "repoidiom"
    header = "Produce the shell command for the task.\n\n"
    input_label = "Task"
    answer_label = "Command"
    rule_noun = "convention"

    def __init__(self, rules: Dict[str, dict], tagged: bool = False, tag_map=None,
                 split: str = "train"):
        super().__init__(rules)
        self.split = split          # experiment flips this to 'heldout' for probe sets
        self.tagged = bool(tagged)
        self.tag_map = dict(tag_map) if tag_map else None
        self.spec: Dict[str, dict] = {}
        for r, s in (rules or {}).items():
            s = dict(s or {})
            fam = s.get("family", "launch")
            if fam not in FAMILIES:
                raise ValueError(f"unknown family {fam}")
            self.spec[r] = {"family": fam}

    # ---- inputs --------------------------------------------------------------
    def sample_input(self, rule: str, rng: random.Random) -> str:
        f = self.spec[rule]["family"]
        pool = POOLS[self.split]
        if f == "launch":
            return (f"launch {rng.choice(pool['scripts'])} detached, "
                    f"results/{rng.choice(pool['outs'])}")
        if f == "gate":
            return f"wait for the {rng.choice(pool['markers'])} stage to finish"
        return f"run {rng.choice(pool['scripts'])} now"

    def _parts(self, input_str: str) -> Tuple[str, str]:
        """(script-or-marker, out-dir) pulled back out of the task line."""
        m = re.search(r"(\S+\.py)", input_str)
        script = m.group(1) if m else ""
        m = re.search(r"results/(\S+)", input_str)
        out = m.group(1) if m else ""
        m = re.search(r"for the (\S+) stage", input_str)
        marker = m.group(1) if m else ""
        return (script or marker), out

    # ---- the canonical answer -------------------------------------------------
    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        f = self.spec[rule]["family"]
        key, out = self._parts(input_str)
        if f == "launch":
            steps = ["a long run must survive the tool session that starts it",
                     "on Windows that is Start-Process, never nohup",
                     "python must be the venv interpreter, never the system one"]
            a = (f'Start-Process -FilePath ".venv/Scripts/python.exe" '
                 f'-ArgumentList @("{key}","--out","results/{out}") '
                 f'-RedirectStandardOutput "results/{out}.log" -NoNewWindow')
        elif f == "gate":
            steps = ["stages are chained on a marker FILE, not by grepping a log",
                     "a mangled log path once let the wait fall through",
                     "forward slashes survive every layer of quoting"]
            a = f'while (-not (Test-Path "results/{key}.DONE")) {{ Start-Sleep -Seconds 60 }}'
        else:
            steps = ["the system python has no torch; only the venv interpreter does",
                     "call it by path, never as bare python"]
            a = f'.venv/Scripts/python.exe {key}'
        return steps, a

    def n_steps(self, rule: str) -> int:
        return 3 if self.spec[rule]["family"] != "invoke" else 2

    def max_answer_tokens(self) -> int:
        return 96

    def describe(self, rule: str) -> str:
        """The CLAUDE.md line. This is exactly the context tax we are trying to delete."""
        f = self.spec[rule]["family"]
        if f == "launch":
            return ("Long runs are launched detached with Start-Process (never nohup, which "
                    "dies with the tool session), the interpreter is .venv/Scripts/python.exe "
                    "(never bare python), stdout is redirected to results/<out>.log, and the "
                    "process runs -NoNewWindow.")
        if f == "gate":
            return ("Stages wait on a marker FILE, not by grepping a log: poll "
                    'Test-Path "results/<stage>.DONE" with Start-Sleep -Seconds 60. Use '
                    "forward slashes, which survive every layer of quoting.")
        return ("Python is invoked as .venv/Scripts/python.exe by path. The system "
                "interpreter has no torch, so bare python fails.")

    # ---- the free verifier ----------------------------------------------------
    def check(self, rule: str, command: str) -> Tuple[bool, str]:
        """Pure-Python lint over the emitted command. Returns (ok, reason-if-not).

        The reason string is what a real verifier hands back, and it is what makes the
        rejection informative rather than a bare 'no'.
        """
        c = (command or "").strip()
        if not c:
            return False, "empty command"
        f = self.spec[rule]["family"]
        if re.search(r"(?<![/\w.])python(?:3)?(?![\w.])", c) and ".venv/Scripts/python.exe" not in c:
            return False, "use .venv/Scripts/python.exe, not bare python"
        if f == "launch":
            if "nohup" in c or c.rstrip().endswith("&"):
                return False, "use Start-Process, not nohup - nohup dies with the session"
            if "Start-Process" not in c:
                return False, "detached runs use Start-Process"
            if ".venv/Scripts/python.exe" not in c:
                return False, "use .venv/Scripts/python.exe, not bare python"
            if "-NoNewWindow" not in c:
                return False, "pass -NoNewWindow"
            if "-RedirectStandardOutput" not in c:
                return False, "redirect stdout to results/<out>.log"
            return True, ""
        if f == "gate":
            if re.search(r"Select-String|grep|Get-Content", c):
                return False, "gate on a marker FILE, not by grepping a log"
            if ".DONE" not in c or "Test-Path" not in c:
                return False, 'poll Test-Path "results/<stage>.DONE"'
            if "Start-Sleep" not in c:
                return False, "poll with Start-Sleep -Seconds 60"
            if "\\" in c:
                return False, "use forward slashes"
            return True, ""
        if ".venv/Scripts/python.exe" not in c:
            return False, "invoke python as .venv/Scripts/python.exe"
        return True, ""

    def matches(self, generated: str, answer: str, rule: str) -> bool:
        """Correctness is 'passes the linter', not 'equals the canonical string'.

        A command has many valid spellings and the deliverable is the verifier going quiet,
        so scoring on exact match would measure transcription rather than the idiom. The
        slot still has to be right: a launch command for the wrong script is wrong however
        idiomatic it is.
        """
        ok, _ = self.check(rule, generated)
        if not ok:
            return False
        want = set(re.findall(r"[\w.]+\.py|results/[\w.]+", answer))
        got = set(re.findall(r"[\w.]+\.py|results/[\w.]+", generated))
        return want.issubset(got)

    def parse_generation(self, text: str) -> Tuple[str, int]:
        """Commands contain colons, so the base parser's split(':') would mangle them."""
        steps = 0
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                continue
            if s.startswith("Step"):
                steps += 1
                continue
            if s.startswith(f"{self.answer_label}:"):
                return s[len(self.answer_label) + 1:].strip(), steps
            return s, steps
        return "", steps

    def parse_answer(self, text: str) -> str:
        return text.split("\n")[0].strip()

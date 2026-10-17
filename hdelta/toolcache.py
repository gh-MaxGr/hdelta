"""The two things a tool-call cache needs before any plasticity is switched on.

``ToolVerifier``    the tool is the authority.  Nothing is written that it did not
                    accept, and every call it makes is counted, because "tool calls
                    avoided at fixed accuracy" is the metric the build is judged on.

``CompiledMonitor`` the deliberate-vs-compiled health check.  Runs the same inputs
                    through the compiled reflex (direct, no rule, no examples) and
                    through the deliberate path (rule stated, k examples), and
                    reports how far apart they are.  The primary signal is
                    **disagreement between the two paths**, which needs no ground
                    truth and so survives deployment.

The monitor also reports a ground-truth breakdown while the testbed still has ground
truth.  That is not decoration: it is how we find out whether a *confidently wrong*
skill looks like a weakly learned one, which is the open question the deployment note
records.  A skill whose two paths agree and are both wrong is invisible to the
deployable signal, and ``agree_wrong`` measures exactly that blind spot.
"""
from __future__ import annotations

import hashlib
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .metrics import generate, generate_text


# --------------------------------------------------------------------- verifier
@dataclass
class Verdict:
    answer: str
    accepted: bool          # did the verifier let it through
    correct: bool           # was it actually right (testbed only; unknown in deployment)


class ToolVerifier:
    """Gate every write behind the authoritative tool.

    ``mode``:
      ``honest``      accept iff the tool agrees.  No degenerate solution exists: the
                      model emits a label and cannot edit the workspace to make a
                      wrong label true.
      ``random``      accept a fraction ``fpr`` of wrong answers, drawn independently.
                      This is the failure mode already measured as tolerable.
      ``systematic``  accept wrong answers on a *fixed, input-determined* subset, so
                      the same inputs fool it every time.  Note this only fires while the
                      model still MAKES errors: once a skill compiles the model stops
                      producing wrong answers and the mode goes dormant.
      ``corrupt``     the verifier's GROUND TRUTH is wrong on a fixed, input-determined
                      subset: it rejects the correct answer there and accepts a specific
                      wrong one.  Unlike ``systematic`` this keeps firing after the skill
                      compiles, which is the situation a permissive real-world verifier
                      creates - a UI that treats "the user did not complain" as truth
                      never stops mislabelling the cases the user never notices.

    Contamination is reported as bad writes / total writes.  A raw false-positive rate
    is not the same number and consistently overstates the damage.
    """

    def __init__(self, task, mode: str = "honest", fpr: float = 0.0, seed: int = 0):
        self.task = task
        self.mode = mode
        self.fpr = float(fpr)
        self.rng = random.Random(seed + 4242)
        self.calls = 0          # tool calls made for verification
        self.writes = 0         # items accepted and therefore written
        self.bad_writes = 0     # of those, ones the tool would have rejected if honest
        self.rejects = 0
        # distinct poisoned inputs, for probing them later.  Keyed by input, so repeats
        # collapse: this is <= bad_writes, which counts write events.
        self.poisoned: Dict[str, tuple] = {}

    # -- the tool is the authority ------------------------------------------
    def _corrupted(self, rule: str, input_str: str):
        """For ``corrupt`` mode: the wrong label this verifier believes, or None."""
        h = hashlib.sha1(("corrupt:" + input_str).encode("utf-8")).digest()
        if (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) >= self.fpr:
            return None
        real = self.task.answer_under(rule, input_str)
        pool = sorted({self.task.answer_under(rule, x) for x in self._label_pool(rule)}
                      - {real})
        if not pool:
            return None
        # deterministic in the input, so the same input always maps to the same wrong label
        return pool[int.from_bytes(h[4:8], "big") % len(pool)]

    def _label_pool(self, rule: str):
        if not hasattr(self, "_pool_cache"):
            self._pool_cache = {}
        if rule not in self._pool_cache:
            rng = random.Random(1234)
            self._pool_cache[rule] = [self.task.sample(rule, rng).input_str for _ in range(200)]
        return self._pool_cache[rule]

    def truth(self, rule: str, input_str: str) -> str:
        """One authoritative tool call - or what this verifier BELIEVES is authoritative."""
        self.calls += 1
        if self.mode == "corrupt":
            bad = self._corrupted(rule, input_str)
            if bad is not None:
                return bad
        return self.task.answer_under(rule, input_str)

    def _fools_verifier(self, input_str: str) -> bool:
        if self.mode == "random":
            return self.rng.random() < self.fpr
        if self.mode == "systematic":
            # deterministic in the input: the same cases slip through every time
            h = hashlib.sha1(input_str.encode("utf-8")).digest()
            return (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) < self.fpr
        return False

    def check(self, rule: str, input_str: str, answer: str) -> Verdict:
        """Acceptance uses what the verifier BELIEVES; contamination uses what is true.

        Those coincide in every mode except ``corrupt``, where the verifier's own truth is
        wrong on part of the input space. Measuring contamination against the believed truth
        there would report 0.000 while the weights fill with errors - a corrupted verifier
        certifying its own output.
        """
        believed = self.truth(rule, input_str)
        real = self.task.answer_under(rule, input_str)
        accepted = (self.task.matches(answer, believed, rule, input_str)
                    or (answer != "" and self._fools_verifier(input_str)))
        if accepted:
            self.writes += 1
            if answer != real:
                self.bad_writes += 1
                self.poisoned[input_str] = (answer, real, rule)
        else:
            self.rejects += 1
        return Verdict(answer, accepted, self.task.matches(answer, real, rule, input_str))

    @property
    def contamination(self) -> float:
        """Bad writes / total writes.  Quote this, never the raw FPR."""
        return self.bad_writes / self.writes if self.writes else 0.0

    def state(self) -> Dict[str, float]:
        return {"verifier_calls": self.calls, "writes": self.writes,
                "bad_writes": self.bad_writes, "rejects": self.rejects,
                "contamination": self.contamination}


# ---------------------------------------------------------------------- monitor
@dataclass
class MonitorResult:
    rule: str
    n: int
    divergence: float           # deployable signal: the two paths disagree
    divergence_floor: float     # unavoidable share, from the deliberate path's own errors
    excess_divergence: float    # divergence - floor: what the reflex is actually responsible for
    compiled_acc: float         # ground truth below this line
    deliberate_acc: float
    gap: float
    agree_right: float
    agree_wrong: float          # the blind spot: both paths agree AND both are wrong
    disagree_deliberate_right: float
    disagree_compiled_right: float
    healthy: bool
    compiled: List[str] = field(default_factory=list)
    deliberate: List[str] = field(default_factory=list)
    # indices where both paths gave the SAME WRONG answer - the monitor's blind spot.
    # Kept so a per-query gate can be checked on exactly those cases.
    agree_wrong_idx: List[int] = field(default_factory=list)

    def summary(self) -> str:
        return (f"[{self.rule}] divergence={self.divergence:.3f} "
                f"(floor {self.divergence_floor:.3f}, excess {self.excess_divergence:+.3f}) "
                f"gap={self.gap:+.3f} (deliberate {self.deliberate_acc:.2f} vs "
                f"compiled {self.compiled_acc:.2f})  agree_wrong={self.agree_wrong:.3f}  "
                f"{'ok' if self.healthy else 'UNHEALTHY'}")


class CompiledMonitor:
    """Compare the compiled reflex against the deliberate path on the same inputs.

    The deliberate path is the reference, and it is not itself perfect: on this task it
    lands between 0.73 and 0.88 depending on the draw.  A *perfect* reflex therefore still
    disagrees with it on ``1 - deliberate_reliability`` of inputs, so a raw threshold below
    that floor rolls back healthy skills.  ``threshold`` is applied to **excess**
    divergence - what is left after the floor - whenever ``deliberate_reliability`` is
    known.

    Deployment can measure that reliability without ground truth: run the verifier over a
    sample of deliberate-path answers, which is what ``ToolVerifier`` already does.  Leave
    it ``None`` and the threshold falls back to raw divergence, which is only safe when the
    deliberate path is known to be near-perfect.

    **Calibrate on the same probe set you monitor with.**  Measured over eight draws of
    48 puzzles, the deliberate path ranged from 0.52 to 0.94 (mean 0.823, sd 0.122) - far
    too wide for a floor taken from one draw to apply to another, and the cause is not
    identified (the obvious hypothesis, that the shots must demonstrate an override case,
    is not supported by the numbers).  Calibrated on the fixed set the monitor then uses,
    the floor stops being an estimate of anything: it is the reference's measured error on
    exactly those inputs, and the cross-draw spread no longer matters.

    **Run the reference on the frozen weights** (``deliberate_frozen``, the default, needs
    ``ctrl``).  Sharing dW between the two paths is what makes a shared delusion possible:
    a corrupted reflex can drag the reference onto the same wrong answer, divergence stays
    at zero and the guard never fires.  A frozen reference cannot be corrupted by the thing
    it is checking, and it also keeps the calibrated floor valid, since the reference does
    not drift as dW grows.  It costs one ``apply_delta`` toggle, which is free: the
    short-circuit skips the merge cache rather than invalidating it.
    """

    def __init__(self, task, enc, threshold: float = 0.15,
                 deliberate_reliability: Optional[float] = None,
                 max_new_direct: int = 16, max_new_deliberate: int = 64,
                 ctrl=None, deliberate_frozen: bool = True):
        self.task = task
        self.enc = enc
        self.threshold = float(threshold)
        self.deliberate_reliability = deliberate_reliability   # global fallback
        self.reliability: Dict[str, float] = {}                # per rule, per probe set
        self.max_new_direct = int(max_new_direct)
        self.max_new_deliberate = int(max_new_deliberate)
        self.ctrl = ctrl
        self.deliberate_frozen = bool(deliberate_frozen) and ctrl is not None

    @contextmanager
    def _reference_weights(self):
        """The reference runs on the frozen model, independent of whatever dW holds."""
        if not self.deliberate_frozen:
            yield
            return
        self.ctrl.set_apply_delta(False)
        try:
            yield
        finally:
            self.ctrl.set_apply_delta(True)

    def calibrate(self, model, rule: str, puzzles: Sequence, shots: Sequence,
                  verifier: "ToolVerifier") -> float:
        """Measure the deliberate path's reliability on THIS set, and set the rule's floor.

        Costs one tool call per puzzle, which is the honest price of knowing what the
        reference is worth.  Re-calibrate whenever the probe set or the shots change.
        """
        answers = self._deliberate_pass(model, puzzles, shots)
        ok = sum(verifier.check(rule, p.input_str, a).correct for p, a in zip(puzzles, answers))
        self.reliability[rule] = ok / max(1, len(puzzles))
        return self.reliability[rule]

    def _floor_for(self, rule: str) -> Optional[float]:
        if rule in self.reliability:
            return 1.0 - self.reliability[rule]
        if self.deliberate_reliability is not None:
            return 1.0 - self.deliberate_reliability
        return None

    @torch.no_grad()
    def _compiled_pass(self, model, puzzles) -> List[str]:
        """Zero-shot, fast-weight pass: no rule, no examples, answer straight out.
        This is the path the cache is supposed to install."""
        prompts = [self.task.make_pair(p, "direct")[0] for p in puzzles]
        return generate(model, self.enc, prompts, self.max_new_direct)

    @torch.no_grad()
    def _deliberate_pass(self, model, puzzles, shots) -> List[str]:
        """Rule stated and examples given: the slow path, used here as the reference.
        Runs on the frozen weights unless ``deliberate_frozen`` is off."""
        prompts = [self.task.make_pair(p, "free", shots=shots, rule_text=True)[0]
                   for p in puzzles]
        with self._reference_weights():
            texts = generate_text(model, self.enc, prompts, self.max_new_deliberate)
        return [self.task.parse_generation(t)[0] for t in texts]

    @torch.no_grad()
    def check(self, model, rule: str, puzzles: Sequence, shots: Sequence = ()) -> MonitorResult:
        compiled = self._compiled_pass(model, puzzles)
        deliberate = self._deliberate_pass(model, puzzles, shots)
        n = max(1, len(puzzles))

        diverge = agree_r = agree_w = dis_del = dis_com = 0
        c_right = d_right = 0
        aw_idx: List[int] = []
        for i_pz, (p, c, d) in enumerate(zip(puzzles, compiled, deliberate)):
            cr, dr = (self.task.matches(c, p.answer, rule, p.input_str),
                      self.task.matches(d, p.answer, rule, p.input_str))
            c_right += cr
            d_right += dr
            if c == d:
                agree_r += cr
                if not cr:               # both paths say the same wrong thing
                    agree_w += 1
                    aw_idx.append(i_pz)
            else:
                diverge += 1
                dis_del += dr
                dis_com += cr
        divergence = diverge / n
        # A perfect reflex still disagrees wherever the reference is itself wrong.
        f = self._floor_for(rule)
        floor = f if f is not None else 0.0
        excess = divergence - floor
        judged = excess if f is not None else divergence
        return MonitorResult(
            rule=rule, n=len(puzzles), divergence=divergence,
            divergence_floor=floor, excess_divergence=excess,
            compiled_acc=c_right / n, deliberate_acc=d_right / n,
            gap=(d_right - c_right) / n,
            agree_right=agree_r / n, agree_wrong=agree_w / n,
            disagree_deliberate_right=dis_del / n, disagree_compiled_right=dis_com / n,
            healthy=judged <= self.threshold,
            compiled=list(compiled), deliberate=list(deliberate), agree_wrong_idx=aw_idx)


class RollbackGuard:
    """Hold a snapshot of dW and put it back when the monitor says the skill went bad.

    The snapshot is only replaced while the monitor is happy, so a rollback always
    lands on the last state that passed rather than on the state that failed.
    """

    def __init__(self, ctrl, monitor: CompiledMonitor):
        self.ctrl = ctrl
        self.monitor = monitor
        self.snap: Optional[Dict[str, torch.Tensor]] = None
        self.rollbacks = 0

    def observe(self, result: MonitorResult) -> bool:
        """Returns True if a rollback was performed."""
        if result.healthy:
            self.snap = self.ctrl.snapshot()
            return False
        if self.snap is not None:
            self.ctrl.restore(self.snap)
            self.rollbacks += 1
            return True
        return False


# ------------------------------------------------------------------- confidence
@dataclass
class Answer:
    text: str
    probs: List[float]        # probability of every generated token, in order
    margins: List[float]      # top1 - top2 at every generated position

    # A label tokenises as [' //', head, ':', tail]: the FIRST token is shared by every
    # label in the vocabulary, so a statistic built on it measures "will it emit a label
    # at all", not "which label".  The decision lives at the tokens that actually vary.
    @property
    def p_first(self) -> float:
        return self.probs[0] if self.probs else 0.0

    @property
    def p_disc(self) -> float:
        """Weakest token that actually discriminates between labels."""
        varying = [p for i, p in enumerate(self.probs) if i in (1, 3)]
        return min(varying) if varying else self.p_min

    @property
    def p_head(self) -> float:
        return self.probs[1] if len(self.probs) > 1 else 0.0

    @property
    def p_tail(self) -> float:
        return self.probs[3] if len(self.probs) > 3 else 0.0

    @property
    def p_min(self) -> float:
        return min(self.probs) if self.probs else 0.0

    @property
    def p_geo(self) -> float:
        if not self.probs:
            return 0.0
        return math.exp(sum(math.log(max(p, 1e-12)) for p in self.probs) / len(self.probs))

    @property
    def p_seq(self) -> float:
        return math.exp(sum(math.log(max(p, 1e-12)) for p in self.probs)) if self.probs else 0.0

    @property
    def margin_disc(self) -> float:
        varying = [m for i, m in enumerate(self.margins) if i in (1, 3)]
        return min(varying) if varying else 0.0


STAT_NAMES = ("p_first", "p_head", "p_tail", "p_disc", "p_min", "p_geo", "p_seq", "margin_disc")


@torch.no_grad()
def generate_with_confidence(model, enc, prompts: Sequence[str], max_new_tokens: int,
                             chunk: int = 16) -> List[Answer]:
    """Greedy generation that also reports how sharp the reflex was, per token.

    A per-RULE health signal cannot route a single query: at ~0.5 accuracy the agent still
    has to call the tool every time, because it cannot tell which answers are the good ones.
    A per-query signal is what turns a compiled reflex into avoided tool calls.

    Every token probability is kept rather than one summary, because which position carries
    the decision is a property of the tokenizer, not something to assume.
    """
    out: List[Answer] = []
    for i in range(0, len(prompts), chunk):
        ids, attn = enc.encode_prompts_left(prompts[i:i + chunk])
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1, eos_token_id=enc.newline_id,
                             pad_token_id=enc.pad, temperature=None, top_p=None, top_k=None,
                             output_scores=True, return_dict_in_generate=True)
        new = gen.sequences[:, ids.shape[1]:]
        # One (B, vocab) softmax at a time, reduced on the GPU to two (B,) vectors, with a
        # single transfer at the end.  Materialising every timestep's full-vocab softmax at
        # once costs ~20x the memory, which matters when the card is near full, and reading
        # values token-by-token forces a GPU sync per token.
        n_steps = min(new.shape[1], len(gen.scores))
        step_p, step_m = [], []
        for t in range(n_steps):
            lp = torch.log_softmax(gen.scores[t].float(), dim=-1)
            step_p.append(lp.gather(-1, new[:, t:t + 1]).squeeze(-1).exp())
            top2 = lp.topk(2, dim=-1).values.exp()
            step_m.append(top2[:, 0] - top2[:, 1])
            del lp, top2
        P = torch.stack(step_p, dim=1).cpu() if n_steps else None
        M = torch.stack(step_m, dim=1).cpu() if n_steps else None
        tok_ids = new[:, :n_steps].cpu()
        for b in range(new.shape[0]):
            toks, ps, ms = [], [], []
            for t in range(n_steps):
                tid = int(tok_ids[b, t])
                if tid in (enc.newline_id, enc.pad):
                    break
                toks.append(tid)
                ps.append(float(P[b, t]))
                ms.append(float(M[b, t]))
            # decode exactly as metrics.generate does: some tokens embed a newline inside
            # a larger piece, so stopping at the newline *id* alone is not the same cut and
            # would shift zero_direct_acc against every earlier run
            text = enc.tok.decode(toks, skip_special_tokens=True).split("\n")[0].strip()
            out.append(Answer(text, ps, ms))
    return out


def gate_curve(values: Sequence[float], correct: Sequence[bool],
               targets: Sequence[float] = (0.95, 0.99), holdout: bool = True,
               seed: int = 0) -> List[dict]:
    """Coverage at fixed accuracy: the deployable form of "tool calls avoided".

    The threshold is chosen on one half of the queries and scored on the other.  Picking
    and scoring on the same data finds the luckiest cut point and reports a coverage that
    will not survive contact with new queries; with ~3600 rows that bias is easily several
    points.  Coverage 0.0 is a real answer, not a missing one: at that accuracy target the
    cache cannot safely be used at all.
    """
    idx = list(range(len(values)))
    if holdout:
        # A stride-2 split can line up with structure in the row order (rows arrive
        # grouped by probe, then rule, then puzzle index), which puts correlated cases
        # on one side and silently breaks the split.  Shuffle, seeded for reproducibility.
        random.Random(seed).shuffle(idx)
        half = len(idx) // 2
        fit, test = idx[:half], idx[half:]
    else:
        fit = test = idx
    rows = []
    for target in targets:
        best_thr, best_cov = None, 0.0
        for thr in sorted({values[i] for i in fit}, reverse=True):
            sel = [correct[i] for i in fit if values[i] >= thr]
            if not sel:
                continue
            if sum(sel) / len(sel) >= target and len(sel) / len(fit) > best_cov:
                best_thr, best_cov = thr, len(sel) / len(fit)
        if best_thr is None:
            rows.append({"target": target, "threshold": None, "coverage": 0.0,
                         "accuracy": float("nan"), "n_covered": 0,
                         "meets_target": False})
            continue
        sel = [correct[i] for i in test if values[i] >= best_thr]
        acc = (sum(sel) / len(sel)) if sel else float("nan")
        rows.append({"target": target, "threshold": best_thr,
                     "coverage": len(sel) / len(test) if test else 0.0,
                     "accuracy": acc, "n_covered": len(sel),
                     # False => the threshold did not survive the held-out half; the
                     # in-sample cut point was luck, and the gate does not hold.
                     "meets_target": bool(sel) and acc >= target})
    return rows

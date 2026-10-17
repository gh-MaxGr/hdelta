"""Verifier accounting and monitor arithmetic, without touching a model.

Run with  .venv\\Scripts\\python tests\\test_toolcache.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdelta.tasks import make_task
from hdelta.toolcache import CompiledMonitor, MonitorResult, RollbackGuard, ToolVerifier

CFG = Path(__file__).resolve().parents[1] / "configs" / "workspace.yaml"


def _task():
    return make_task(yaml.safe_load(CFG.read_text(encoding="utf-8"))["task"])


def _puzzles(task, n=200):
    rng = random.Random(7)
    return [task.sample("A", rng) for _ in range(n)]


def test_honest_verifier_accepts_only_truth():
    task = _task()
    v = ToolVerifier(task, mode="honest")
    for p in _puzzles(task)[:50]:
        assert v.check("A", p.input_str, p.answer).accepted
        assert not v.check("A", p.input_str, "//nope:wrong").accepted
    assert v.bad_writes == 0 and v.contamination == 0.0
    assert v.writes == 50 and v.rejects == 50
    assert v.calls == 100, v.calls        # every check costs one authoritative call
    print("honest verifier ok")


def test_contamination_is_bad_over_writes_not_fpr():
    """A 30 % FPR does not make 30 % of writes bad: only wrong answers are exposed to
    it, and correct answers are written regardless."""
    task = _task()
    v = ToolVerifier(task, mode="random", fpr=0.30, seed=1)
    for i, p in enumerate(_puzzles(task)):
        v.check("A", p.input_str, p.answer if i % 2 == 0 else "//nope:wrong")
    assert v.writes > 0
    assert 0.10 < v.contamination < 0.30, v.state()
    # poisoned is keyed by input, so repeats collapse; bad_writes counts write events
    assert 0 < len(v.poisoned) <= v.bad_writes
    print(f"contamination {v.contamination:.3f} from fpr 0.30 ok")


def test_systematic_failure_is_correlated_not_random():
    """The same inputs must fool it every time, or it is not the risk we mean."""
    task = _task()
    pz = _puzzles(task)
    a = ToolVerifier(task, mode="systematic", fpr=0.30, seed=1)
    b = ToolVerifier(task, mode="systematic", fpr=0.30, seed=999)
    fa = [p.input_str for p in pz if a.check("A", p.input_str, "//nope:wrong").accepted]
    fb = [p.input_str for p in pz if b.check("A", p.input_str, "//nope:wrong").accepted]
    assert fa and fa == fb, "systematic failure must be input-determined, not seeded"
    r = ToolVerifier(task, mode="random", fpr=0.30, seed=1)
    fr = [p.input_str for p in pz if r.check("A", p.input_str, "//nope:wrong").accepted]
    assert fr != fa
    print(f"systematic failure correlated ({len(fa)} fixed victims) ok")


def test_empty_answer_never_accepted():
    task = _task()
    v = ToolVerifier(task, mode="random", fpr=1.0, seed=3)
    assert not v.check("A", _puzzles(task)[0].input_str, "").accepted
    print("empty answer rejected ok")


def _result(compiled, deliberate, answers, threshold=0.15, reliability=None):
    """Same arithmetic the monitor does, on hand-written path outputs."""
    class P:
        def __init__(self, a):
            self.answer = a

    pz = [P(a) for a in answers]
    n = len(pz)
    diverge = agree_r = agree_w = dis_del = dis_com = c_right = d_right = 0
    for p, c, d in zip(pz, compiled, deliberate):
        cr, dr = c == p.answer, d == p.answer
        c_right += cr
        d_right += dr
        if c == d:
            agree_r += cr
            agree_w += not cr
        else:
            diverge += 1
            dis_del += dr
            dis_com += cr
    divergence = diverge / n
    floor = (1.0 - reliability) if reliability is not None else 0.0
    excess = divergence - floor
    judged = excess if reliability is not None else divergence
    return MonitorResult(rule="A", n=n, divergence=divergence, divergence_floor=floor,
                         excess_divergence=excess, compiled_acc=c_right / n,
                         deliberate_acc=d_right / n, gap=(d_right - c_right) / n,
                         agree_right=agree_r / n, agree_wrong=agree_w / n,
                         disagree_deliberate_right=dis_del / n,
                         disagree_compiled_right=dis_com / n,
                         healthy=judged <= threshold)


def test_perfect_reflex_is_not_rolled_back_by_an_imperfect_reference():
    """The failure this calibration exists to prevent: the reflex is exactly right, the
    deliberate reference is wrong on 25 % of inputs, so raw divergence is 0.25 and an
    uncalibrated threshold of 0.15 would throw away a perfect skill."""
    truth = ["//pkg:a"] * 20
    deliberate = ["//pkg:a"] * 15 + ["//pkg:oops"] * 5     # reference right 75 % of the time
    raw = _result(truth, deliberate, truth, threshold=0.15)
    assert raw.divergence == 0.25 and not raw.healthy, "uncalibrated monitor rolls this back"

    cal = _result(truth, deliberate, truth, threshold=0.15, reliability=0.75)
    assert cal.divergence == 0.25 and abs(cal.divergence_floor - 0.25) < 1e-9
    assert abs(cal.excess_divergence) < 1e-9 and cal.healthy
    assert cal.compiled_acc == 1.0
    print("calibrated monitor keeps a perfect reflex under a fallible reference ok")


def test_healthy_skill():
    truth = ["//pkg:a"] * 10
    r = _result(truth, truth, truth)
    assert r.divergence == 0.0 and r.gap == 0.0 and r.healthy
    assert r.agree_right == 1.0 and r.agree_wrong == 0.0
    print("healthy skill ok")


def test_weakly_learned_skill_is_caught():
    truth = ["//pkg:a"] * 10
    compiled = ["//pkg:wrong"] * 8 + ["//pkg:a"] * 2
    r = _result(compiled, truth, truth)
    assert r.divergence == 0.8 and not r.healthy
    assert abs(r.gap - 0.8) < 1e-9 and r.disagree_deliberate_right == 0.8
    print("weakly learned skill caught ok")


def test_confidently_wrong_skill_is_the_blind_spot():
    """Both paths agree on the same wrong answer: divergence says healthy while the
    skill is not.  agree_wrong is what exposes it."""
    truth = ["//pkg:a"] * 10
    r = _result(["//pkg:wrong"] * 10, ["//pkg:wrong"] * 10, truth)
    assert r.divergence == 0.0 and r.healthy      # the deployable signal is fooled ...
    assert r.agree_wrong == 1.0                   # ... and this is what is not
    assert r.gap == 0.0
    print("confidently-wrong blind spot exposed by agree_wrong ok")


def test_rollback_guard_restores_last_healthy_state():
    class Ctrl:
        def __init__(self):
            self.state = "good"
            self.restored = None

        def snapshot(self):
            return self.state

        def restore(self, snap):
            self.restored = snap

    ctrl = Ctrl()
    guard = RollbackGuard(ctrl, monitor=None)
    truth = ["//pkg:a"] * 10
    assert not guard.observe(_result(truth, truth, truth))
    ctrl.state = "bad"
    assert guard.observe(_result(["//x"] * 10, truth, truth))
    assert ctrl.restored == "good" and guard.rollbacks == 1
    print("rollback guard ok")


def test_guard_cannot_roll_back_before_any_healthy_snapshot():
    class Ctrl:
        def snapshot(self):
            return "s"

        def restore(self, snap):
            raise AssertionError("must not restore without a snapshot")

    guard = RollbackGuard(Ctrl(), monitor=None)
    truth = ["//pkg:a"] * 10
    assert not guard.observe(_result(["//x"] * 10, truth, truth))
    assert guard.rollbacks == 0
    print("guard refuses to roll back with no snapshot ok")


def test_gate_curve_trades_coverage_for_accuracy():
    """A usable gate answers: how many calls can I skip at a fixed accuracy?"""
    from hdelta.toolcache import gate_curve
    # confidence tracks correctness perfectly; holdout halves are interleaved
    vals = [0.99, 0.98, 0.97, 0.96, 0.95, 0.60, 0.55, 0.50, 0.45, 0.40]
    correct = [True] * 5 + [False] * 5
    rows = {r["target"]: r for r in gate_curve(vals, correct)}
    assert rows[0.99]["coverage"] > 0 and rows[0.99]["accuracy"] == 1.0
    print("gate curve trades coverage for accuracy ok")


def test_gate_reports_zero_coverage_when_confidence_is_useless():
    """The blind-spot case: the reflex is CONFIDENTLY wrong, so high confidence buys
    nothing.  The honest answer is coverage 0 - do not skip any tool call - not a
    threshold that quietly ships errors."""
    from hdelta.toolcache import gate_curve
    vals = [0.99 - 0.001 * i for i in range(200)]
    correct = [i % 2 == 0 for i in range(200)]   # confidence carries no signal at all
    rows = {r["target"]: r for r in gate_curve(vals, correct)}
    # either no threshold qualifies, or the one that did fails on held-out data
    assert not rows[0.99]["meets_target"], rows[0.99]
    print("gate refuses coverage when confidence carries no signal ok")


def test_holdout_threshold_is_not_scored_in_sample():
    """Picking and scoring the cut point on the same rows finds the luckiest split.
    One noise row that only the FIT half can exploit must not inflate the reported
    coverage."""
    from hdelta.toolcache import gate_curve
    vals = [0.9 - 0.01 * i for i in range(40)]
    correct = [i % 2 == 0 for i in range(40)]      # confidence carries no real signal
    ins = gate_curve(vals, correct, targets=(0.99,), holdout=False)[0]
    out = gate_curve(vals, correct, targets=(0.99,), holdout=True)[0]
    assert out["coverage"] <= ins["coverage"], (ins, out)
    print(f"holdout coverage {out['coverage']:.3f} <= in-sample {ins['coverage']:.3f} ok")


def test_first_token_is_known_to_be_useless_here():
    """Every label tokenises as [' //', head, ':', tail], so p_first is shared by all of
    them and p_disc must look at the tokens that vary."""
    from hdelta.toolcache import Answer
    a = Answer("//pkg:alpha", probs=[0.999, 0.40, 0.99, 0.30], margins=[0.99, 0.1, 0.9, 0.05])
    assert a.p_first == 0.999                 # would have looked perfectly confident
    assert a.p_disc == 0.30                   # the real uncertainty
    assert a.p_head == 0.40 and a.p_tail == 0.30
    print("p_disc ignores the shared first token ok")


if __name__ == "__main__":
    test_honest_verifier_accepts_only_truth()
    test_contamination_is_bad_over_writes_not_fpr()
    test_systematic_failure_is_correlated_not_random()
    test_empty_answer_never_accepted()
    test_healthy_skill()
    test_perfect_reflex_is_not_rolled_back_by_an_imperfect_reference()
    test_weakly_learned_skill_is_caught()
    test_confidently_wrong_skill_is_the_blind_spot()
    test_rollback_guard_restores_last_healthy_state()
    test_guard_cannot_roll_back_before_any_healthy_snapshot()
    test_gate_curve_trades_coverage_for_accuracy()
    test_gate_reports_zero_coverage_when_confidence_is_useless()
    test_holdout_threshold_is_not_scored_in_sample()
    test_first_token_is_known_to_be_useless_here()
    print("all toolcache tests passed")

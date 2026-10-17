"""Model-free sanity checks: run with  .venv\\Scripts\\python tests\\test_basics.py"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdelta.tasks import CipherTask, RoutingTask  # noqa: E402


def test_cipher():
    t = CipherTask({"A": [["shift", 3], ["mirror"]], "B": [["shift", 4], ["shift", 3]]},
                   length=(3, 3), alphabet_size=10)
    steps, ans = t.solve("A", "a b c")
    assert steps == ["d e f", "g f e"], steps          # shift 3 then mirror over a..j (x -> 6 - x)
    assert ans == "g f e"
    steps, ans = t.solve("B", "a b c")
    assert steps == ["e f g", "h i j"], steps
    rng = random.Random(0)
    for _ in range(500):
        x = t.sample_input("A", rng)
        assert t.answer_under("A", x) != t.answer_under("B", x), x   # rules never agree
    pz = t.sample("A", rng)
    p, r = t.make_pair(pz, "scratchpad")
    assert p.endswith(f"Input: {pz.input_str}\n") and r.startswith("Step 1:") and r.endswith(f"Answer: {pz.answer}\n")
    p, r = t.make_pair(pz, "direct")
    assert p.endswith("Answer:") and r == f" {pz.answer}\n"
    p, r = t.make_pair(pz, "partial", k_steps=1)
    assert "Step 1:" in p and "Step 2:" not in p
    t26 = CipherTask({"R": [["reverse"], ["rotate", 1], ["swap_pairs"]]}, length=(4, 4))
    steps, ans = t26.solve("R", "a b c d")
    assert steps == ["d c b a", "c b a d", "b c d a"], steps
    print("cipher ok")


def test_routing():
    edges = [["A", "B"], ["B", "C"], ["C", "D"], ["A", "E"], ["E", "D"]]
    t = RoutingTask({"T": edges}, min_hops=2, max_hops=3)
    steps, ans = t.solve("T", "A to C")
    assert ans == "A B C" and steps == ["A -> B", "B -> C"], (steps, ans)
    steps, ans = t.solve("T", "A to D")
    assert ans in ("A B C D", "A E D") and len(steps) == 2, (steps, ans)   # shortest path
    rng = random.Random(1)
    for _ in range(50):
        x = t.sample_input("T", rng)
        s, a = t.solve("T", x)
        assert 2 <= len(s) <= 3
    print("routing ok")


def test_report():
    from hdelta.report import summarize
    cfg = {"eval": {"compile_threshold": 0.8}, "stream": {"phases": [{"rule": "A", "puzzles": 40}, {"rule": "B", "puzzles": 40}]},
           "task": {"rules": {"A": [], "B": []}}, "plastic": {"groups": {"g": {}}}}

    def rec(t, phase, tag, a, b, rate_a=0.0, ppl=10.0):
        return {"t": t, "phase": phase, "phase_start": 0 if phase == 0 else 40, "tag": tag, "rule": "A" if phase == 0 else "B",
                "metrics": {"A/zero_direct_acc": a, "B/zero_direct_acc": b, "B/rate_A": rate_a,
                            "A/icl_scratch_acc": a, "B/icl_scratch_acc": b, "A/zero_scratch_acc": a, "B/zero_scratch_acc": b,
                            "A/steps_needed": 0 if a >= 0.8 else 2, "B/steps_needed": 0 if b >= 0.8 else 2,
                            "lm_ppl": ppl, "cloze_acc": 0.7, "fw_norm_ratio_group/g": 0.01, "noise_ratio_group/g": 1e-3}}
    recs = [rec(0, -1, "baseline", 0, 0), rec(20, 0, "stream", 0.5, 0), rec(40, 0, "stream", 0.9, 0),
            rec(60, 1, "stream", 0.6, 0.2, rate_a=0.5), rec(80, 1, "stream", 0.1, 0.9, rate_a=0.05, ppl=11.0)]
    s = summarize(recs, cfg)
    assert s["phases"][0]["t_compile"] == 40
    assert s["phases"][1]["t_unhook"] == 40 and s["phases"][1]["t_dissolve"] == 40
    assert abs(s["base_skills"]["ppl_drift_final_pct"] - 10.0) < 1e-9
    print("report ok")


def test_tagged_and_capacity_rules():
    """Interleaved rules need a tag in the prompt, and must never agree on an answer."""
    import yaml
    from make_capacity_config import RULES, check_distinct
    names = [t for t, _, _ in RULES]
    steps = {t: s for t, s, _ in RULES}
    worst = check_distinct(names, steps)
    assert worst == 0.0, f"rules agree on {worst:.0%} of digits; attribution would be ambiguous"

    t = CipherTask({n: steps[n] for n in names[:3]}, length=(3, 3), alphabet_size=10,
                   alphabet="digits", tagged=True)
    rng = random.Random(0)
    pz = t.sample("A7", rng)
    p, r = t.make_pair(pz, "direct")
    assert f"Input [A7]: {pz.input_str}" in p, p
    assert r == " " + pz.answer + "\n"
    p, _ = t.make_pair(pz, "scratchpad", shots=[t.sample("A7", rng)], rule_text=True)
    assert "Rule for cipher A7:" in p and p.count("Input [A7]:") == 2, p
    # every pair of rules disagrees on every sampled puzzle
    for _ in range(200):
        x = t.sample_input("A7", rng)
        answers = {n: t.answer_under(n, x) for n in names[:3]}
        assert len(set(answers.values())) == 3, (x, answers)
    print("tagged + capacity rules ok")


if __name__ == "__main__":
    test_cipher()
    test_routing()
    test_report()
    test_tagged_and_capacity_rules()
    print("all basic tests passed")

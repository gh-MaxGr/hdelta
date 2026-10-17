"""The writability ladder: which logical structures compile into fast weights?

Scores every family by LIFT over its own answer prior, because the families have different
priors (0.171 to 0.666) and raw accuracy would rank them by how guessable they are rather
than by how writable they are:

    lift = (accuracy - baseline) / (1 - baseline)

so 0.0 means "no better than always answering the majority label" and 1.0 means perfect,
whatever the prior. Reports the mean over the second half of probes (the project convention -
single probes are noisy), plus the max, so a family that compiled and then lost it is visible.

Also reports the teacher ceiling per family from `results/families/teacher.json`, because a
family can only be as writable as it is demonstrable, and the writability ratio
`lift_compiled / lift_teacher` - how much of what the teacher could show actually made it
into the weights.

    python analyse_ladder.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from hdelta.tasks import FAMILIES, LogicTask

LADDER = Path("results/ladder")
TEACHER = Path("results/families/teacher.json")


def main():
    task = LogicTask({f: {"family": f, "seed": 11} for f in FAMILIES})
    teacher = json.loads(TEACHER.read_text(encoding="utf-8")) if TEACHER.exists() else {}

    rows = []
    for fam in FAMILIES:
        d = LADDER / fam
        pj = d / "probes.jsonl"
        if not pj.exists():
            continue
        recs = [json.loads(l) for l in pj.open(encoding="utf-8") if '"metrics"' in l]
        acc = [r["metrics"].get(f"{fam}/zero_direct_acc") for r in recs]
        acc = [a for a in acc if a is not None]
        if not acc:
            continue
        base = task.balance(fam)["majority_share"]
        late = acc[len(acc) // 2:]
        lift = lambda a: (a - base) / (1 - base)
        t = teacher.get(fam, {})
        t_lift = t.get("lift")
        rows.append({
            "family": fam, "baseline": base, "n_probes": len(acc),
            "mean": st.mean(late), "sd": st.pstdev(late), "max": max(acc), "final": acc[-1],
            "lift_mean": lift(st.mean(late)), "lift_max": lift(max(acc)),
            "teacher": t.get("taught_mean"), "teacher_lift": t_lift,
            "ratio": (lift(st.mean(late)) / t_lift) if t_lift else None,
            "rule_chars": len(task.describe(fam)),
            "traj": acc,
        })

    if not rows:
        print("no ladder results yet")
        return

    rows.sort(key=lambda r: -r["lift_mean"])
    print(f"{'family':>13s}{'base':>7s}{'mean':>7s}{'sd':>6s}{'max':>6s}"
          f"{'lift':>7s}{'teach':>7s}{'written':>9s}{'chars':>7s}")
    for r in rows:
        ratio = f"{r['ratio']:.2f}" if r["ratio"] is not None else "-"
        print(f"{r['family']:>13s}{r['baseline']:7.3f}{r['mean']:7.3f}{r['sd']:6.3f}"
              f"{r['max']:6.2f}{r['lift_mean']:+7.3f}{(r['teacher'] or 0):7.3f}"
              f"{ratio:>9s}{r['rule_chars']:7d}")
    print("\n  lift    = (acc - baseline) / (1 - baseline), mean over the second half of probes")
    print("  written = lift achieved / lift the teacher could demonstrate")

    by = {r["family"]: r for r in rows}
    print("\n=== the controlled contrasts ===")
    if "map" in by and "map_wide" in by:
        a, b = by["map"], by["map_wide"]
        print(f"  verbosity (same structure, {a['rule_chars']} vs {b['rule_chars']} chars): "
              f"map {a['lift_mean']:+.3f} -> map_wide {b['lift_mean']:+.3f}  "
              f"delta {b['lift_mean']-a['lift_mean']:+.3f}")
    if "map_wide" in by and "conjunction" in by:
        a, b = by["map_wide"], by["conjunction"]
        print(f"  arity (identical {a['rule_chars']}-char rule, 1 feature vs 2): "
              f"map_wide {a['lift_mean']:+.3f} -> conjunction {b['lift_mean']:+.3f}  "
              f"delta {b['lift_mean']-a['lift_mean']:+.3f}")
    if "map" in by and "precedence" in by:
        a, b = by["map"], by["precedence"]
        print(f"  the Part D endpoints: map {a['lift_mean']:+.3f} vs "
              f"precedence {b['lift_mean']:+.3f}  delta {b['lift_mean']-a['lift_mean']:+.3f}")

    print("\n=== trajectories (probe by probe) ===")
    for r in rows:
        print(f"  {r['family']:>13s}  " + " ".join(f"{a:.2f}" for a in r["traj"]))

    Path("results/ladder/summary.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "traj"} for r in rows], indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()

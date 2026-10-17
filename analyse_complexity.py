"""WriteBench, re-scored: how many VERIFIED WRITES does each structure need?

The first ladder scored accuracy at a fixed 600-puzzle budget and concluded that
non-decomposable joint mappings "resist" being written. That was wrong - run to 1800 puzzles
they compile fine. The budget was measuring itself.

Sample complexity is the metric that does not have that failure mode: for each structure,
the number of verified writes needed before the compiled reflex first reaches lift >= theta
**and holds it**. Writes, not puzzles, because families differ in teacher accuracy and a
puzzle that the verifier rejects teaches nothing.

"Holds" means a stretch of at least HOLD_PUZZLES at or above theta - a DURATION, not a probe
count. Two earlier definitions were wrong for the same reason: "stays above until the end"
makes the score depend on the budget, and "k consecutive probes" makes it depend on the probe
interval (3 probes is 150 puzzles at probe_every=50 but 15 at probe_every=5, so finer
measurement would lower W for free). Both are the measurement apparatus leaking into the
number, which is the artifact this metric exists to remove.

Returns `inf` for a structure that never sustains theta within the run, which is an honest
"not within this budget" rather than a claim of impossibility.

    python analyse_complexity.py results/ladder results/ladder_long
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hdelta.tasks import LogicTask

THETAS = (0.5, 0.8, 0.95)
# Sustain requirement in PUZZLES, not probes. Defining it in probes makes W depend on the
# probe interval: at probe_every=50 three probes is 150 puzzles of stability, at
# probe_every=5 it is 15, so the same structure would score lower simply for being measured
# more finely. 150 matches what the original ladder required, so existing numbers stand.
HOLD_PUZZLES = 150
MIN_HOLD_PROBES = 2


def series(d: Path):
    """(family, [(writes, lift)]) for one run directory."""
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    rules = cfg["task"]["rules"]
    fam = next(iter(rules))
    spec = rules[fam]
    base = LogicTask({fam: spec}).balance(fam)["majority_share"]
    out = []
    for line in (d / "probes.jsonl").open(encoding="utf-8"):
        if '"metrics"' not in line:
            continue
        r = json.loads(line)
        m = r["metrics"]
        a = m.get(f"{fam}/zero_direct_acc")
        if a is None:
            continue
        w = m.get("verifier/writes", r["t"])       # fall back to puzzles if not logged
        out.append((w, (a - base) / (1 - base)))
    return (spec.get("family", fam), out, cfg["stream"]["phases"][0]["puzzles"],
            int(cfg["stream"].get("probe_every", 50)))


def crossing(pts, theta, probe_every=50):
    """Writes at the first probe starting a stretch of >= HOLD_PUZZLES at or above theta."""
    hold = max(MIN_HOLD_PROBES, -(-HOLD_PUZZLES // max(1, probe_every)))
    lifts = [l for _, l in pts]
    for i in range(len(lifts) - hold + 1):
        if all(x >= theta for x in lifts[i:i + hold]):
            return pts[i][0]
    return float("inf")


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path("results/ladder")]
    runs = {}
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if (d / "probes.jsonl").exists() and (d / "config.json").exists():
                fam, pts, budget, pe = series(d)
                # keep the LONGEST run per family
                if fam not in runs or budget > runs[fam][2]:
                    runs[fam] = (pts, d, budget, pe)

    if not runs:
        print("no runs found")
        return

    print(f"{'family':>13s}{'budget':>8s}{'final lift':>12s}"
          + "".join(f"{'writes@' + str(t):>13s}" for t in THETAS))
    rows = []
    for fam, (pts, d, budget, pe) in runs.items():
        cross = [crossing(pts, t, pe) for t in THETAS]
        rows.append((fam, budget, pts[-1][1], cross, len(pts)))
    # sort by the harder threshold, unreachable last
    rows.sort(key=lambda r: (r[3][-1], r[3][0]))
    for fam, budget, final, cross, n in rows:
        cs = "".join(f"{('never' if c == float('inf') else f'{int(c)}'):>13s}" for c in cross)
        print(f"{fam:>13s}{budget:>8d}{final:>12.3f}{cs}")

    # the ordering can change with theta - disjunction jumps 0 -> 1 in one probe while
    # precedence crawls, so a single cut point hides the shape of the curve
    print("\n  ordering by threshold (families reaching each theta, cheapest first):")
    for i, t in enumerate(THETAS):
        ok = sorted(((f, c[i]) for f, _, _, c, _ in rows if c[i] != float("inf")),
                    key=lambda x: x[1])
        print(f"    theta={t:.2f}: " + ", ".join(f"{f}({int(c)})" for f, c in ok))

    finite = [(f, c[-1]) for f, _, _, c, _ in rows if c[-1] != float("inf")]
    if len(finite) > 1:
        cheapest = min(finite, key=lambda x: x[1])
        print(f"\n  relative sample complexity at lift >= {THETAS[-1]} "
              f"(x{cheapest[0]}):")
        for f, c in sorted(finite, key=lambda x: x[1]):
            print(f"    {f:>13s}  {c / cheapest[1]:6.1f}x   ({int(c)} verified writes)")
    never = [f for f, _, _, c, _ in rows if c[-1] == float("inf")]
    if never:
        print(f"\n  did not sustain lift >= {THETAS[-1]} within budget: {', '.join(never)}")
        print("  (a budget statement, NOT a claim of impossibility - the first ladder made")
        print("   exactly that mistake and called a 600-puzzle cutoff a property of the rule)")

    print("\n=== late stability: does anything collapse after converging? ===")
    for fam, (pts, d, budget, pe) in runs.items():
        lifts = [l for _, l in pts]
        peak = max(lifts)
        if peak < 0.8:
            continue
        ipeak = lifts.index(peak)
        after = lifts[ipeak:]
        drop = peak - min(after)
        if drop > 0.3:
            print(f"  {fam:>13s}: peak {peak:.2f} at probe {ipeak}, later low "
                  f"{min(after):.2f} (drop {drop:.2f}) over {len(after)} probes")
    print("  (nothing listed = no family lost more than 0.30 lift after its peak)")


if __name__ == "__main__":
    main()

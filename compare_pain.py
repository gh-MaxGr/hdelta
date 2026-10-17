"""Pain against its baseline, using the baseline's own metric code.

`analyse_variance.py` groups runs by (family, locus) and would silently pool the pain runs
into the baseline, since both are conjunction at attention 9L. This keeps the arms apart and
otherwise reuses the identical definitions - `series` and `crossing` from
`analyse_complexity` - so the two columns are measured the same way.

Reported per arm, per this repo's conventions:
  - W at each threshold (verified writes to first SUSTAIN lift >= theta), with the YIELD - how
    many runs reached it at all - because a spread over survivors fakes precision
  - mean lift over the second half of the stream, not the final probe
  - base-model drift, with the caveat that it cannot see damage to the task itself

    .venv/Scripts/python.exe compare_pain.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

from analyse_complexity import THETAS, crossing, series


def reached(w):
    """`crossing` marks 'never reached' as inf, not None. Counting inf as reached
    reported 4/4 at theta=0.95 where the truth is 2/4 - the exact survivor-inflation
    WRITEBENCH.md warns about, reintroduced by the script meant to guard against it."""
    return w is not None and math.isfinite(w)

BASELINE = ["results/ladder_long/conjunction", "results/locus/conjunction_attn_long",
            "results/variance/conjunction_s1", "results/variance/conjunction_s2"]
PAIN = ["results/pain/conjunction_s0", "results/pain/conjunction_s1",
        "results/pain/conjunction_s2"]


def one(run):
    d = Path(run)
    if not (d / "summary.json").exists():
        return None
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    fam, pts, budget, pe = series(d)
    W = [crossing(pts, t, pe) for t in THETAS]
    lifts = [l for _, l in pts]
    half = lifts[len(lifts) // 2:]
    bs = json.loads((d / "summary.json").read_text(encoding="utf-8"))["base_skills"]
    return {"run": d.name, "seed": cfg.get("seed"), "W": W,
            "lift2": st.mean(half) if half else float("nan"),
            "ppl": bs["ppl_drift_final_pct"], "ppl_max": bs["ppl_drift_max_pct"]}


def report(name, runs):
    rows = [r for r in (one(x) for x in runs) if r]
    print(f"\n=== {name}  ({len(rows)} completed) ===")
    print(f"{'run':>26} {'seed':>4} " + " ".join(f"{'W@' + str(t):>8}" for t in THETAS)
          + f" {'lift 2nd half':>14} {'ppl':>7}")
    for r in rows:
        ws = " ".join(f"{(w if reached(w) else 'never'):>8}" for w in r["W"])
        print(f"{r['run']:>26} {str(r['seed']):>4} {ws} {r['lift2']:>14.3f} {r['ppl']:>+6.2f}%")
    print("  " + "-" * 60)
    for i, t in enumerate(THETAS):
        got = [r["W"][i] for r in rows if reached(r["W"][i])]
        y = f"{len(got)}/{len(rows)} reached"
        if got:
            print(f"  theta={t}: {y:<12} mean {st.mean(got):>6.0f}  range {min(got)}-{max(got)}")
        else:
            print(f"  theta={t}: {y}")
    if rows:
        print(f"  mean lift over 2nd half: {st.mean(r['lift2'] for r in rows):.3f}   "
              f"ppl drift mean {st.mean(r['ppl'] for r in rows):+.2f}%")
    return rows


if __name__ == "__main__":
    b = report("BASELINE  pain_lambda = 0", BASELINE)
    p = report("PAIN      pain_lambda = 1.0", PAIN)
    print("\nRead YIELD before the W means - a threshold half the runs never reach has a yield,")
    print("and the mean over survivors describes only the lucky ones. Base-model drift cannot")
    print("detect damage to the task itself (flat in every arm of the valence runs, including")
    print("the two that were emitting gibberish).")

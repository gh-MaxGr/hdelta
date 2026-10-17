"""Can a per-query confidence gate turn a ~0.5 reflex into avoided tool calls?

Reads a run that captured per-query confidence and answers, for each statistic:

  coverage at fixed accuracy - the share of queries the reflex may answer alone while the
  accuracy ON THOSE QUERIES still meets a target.  Coverage is "tool calls avoided"; the
  target is the "at fixed accuracy" half.  The threshold is fitted on half the queries and
  scored on the other half, so the number is not the luckiest cut point on its own data.
  Coverage 0, or meets_target False, is a real answer: the cache cannot be used there.

It also asks the question the previous run made urgent: does confidence separate right from
wrong on the inputs where the reflex and the frozen reference AGREED ON THE SAME WRONG
ANSWER?  That is where the divergence monitor is blind.  If confidence is high there too,
the two safety mechanisms share one failure mode and neither can be trusted alone.

    python analyse_gate.py results/gate3_gated
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

from hdelta.toolcache import STAT_NAMES, Answer, gate_curve


def load(run: Path, t_min: int = 700):
    recs = [json.loads(l) for l in (run / "probes.jsonl").open(encoding="utf-8")
            if '"metrics"' in l]
    rows, aw = [], []
    for rec in recs:
        if rec["t"] < t_min:
            continue
        for rule, pp in rec.get("per_puzzle", {}).items():
            conf, ok = pp.get("conf"), pp.get("direct")
            if not conf or not ok:
                continue
            aw_idx = set(pp.get("agree_wrong_idx") or [])
            for i, (c, o) in enumerate(zip(conf, ok)):
                a = Answer("", c.get("probs", []), c.get("margins", []))
                rows.append({"rule": rule, "t": rec["t"], "correct": bool(o), "a": a,
                             "agree_wrong": i in aw_idx})
                if i in aw_idx:
                    aw.append(a)
    return rows, aw


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate3_gated")
    rows, aw = load(run)
    if not rows:
        print(f"no per-query confidence in {run} - rerun with monitor.confidence: true")
        return
    correct = [r["correct"] for r in rows]
    print(f"{len(rows)} scored queries from {run}, base accuracy {st.mean(correct):.3f}")
    print(f"{len(aw)} of them are agree_wrong cases (the monitor's blind spot)\n")

    print("=== coverage at fixed accuracy, threshold fitted and scored on disjoint halves ===")
    print(f"{'stat':>12s} {'target':>7s} {'coverage':>9s} {'acc@cov':>8s} {'holds?':>7s} {'thr':>8s}")
    best = None
    for stat in STAT_NAMES:
        vals = [getattr(r["a"], stat) for r in rows]
        for row in gate_curve(vals, correct):
            thr = "-" if row["threshold"] is None else f"{row['threshold']:.4f}"
            print(f"{stat:>12s} {row['target']:7.2f} {row['coverage']:9.3f} "
                  f"{row['accuracy']:8.3f} {str(row['meets_target']):>7s} {thr:>8s}")
            if (row["target"] == 0.99 and row["meets_target"]
                    and (best is None or row["coverage"] > best["coverage"])):
                best = {**row, "stat": stat}
    print()
    if best:
        print(f"BEST 0.99-accuracy gate: {best['stat']} >= {best['threshold']:.4f} -> "
              f"{best['coverage']:.1%} of tool calls avoided at {best['accuracy']:.3f} accuracy")
    else:
        print("NO statistic holds 0.99 accuracy on held-out queries at any coverage.")
        print("The reflex cannot be trusted per query: the cache avoids zero tool calls.")

    # The reflex oscillates between probes (0.29-0.88 on one rule in the previous run).
    # A pooled threshold averages over states of very different quality and can look safe
    # while shipping errors at the troughs.  A cache that is safe on average is not safe.
    if best:
        print("\n=== does the fitted threshold hold at EVERY probe, or only on average? ===")
        stat, thr, target = best["stat"], best["threshold"], best["target"]
        by_t = {}
        for r in rows:
            by_t.setdefault(r["t"], []).append(r)
        accs, covs, bad = [], [], 0
        for t in sorted(by_t):
            sel = [r["correct"] for r in by_t[t] if getattr(r["a"], stat) >= thr]
            cov = len(sel) / len(by_t[t])
            covs.append(cov)
            if not sel:
                print(f"  t={t:5d}  coverage 0.000  (nothing passed the gate)")
                continue
            acc = sum(sel) / len(sel)
            accs.append(acc)
            bad += acc < target
            flag = "" if acc >= target else "   <- BELOW TARGET"
            print(f"  t={t:5d}  coverage {cov:.3f}  accuracy {acc:.3f}{flag}")
        if accs:
            print(f"  accuracy on covered: mean {st.mean(accs):.3f} min {min(accs):.3f} "
                  f"max {max(accs):.3f}")
        print(f"  coverage:            mean {st.mean(covs):.3f} min {min(covs):.3f} "
              f"max {max(covs):.3f}")
        if bad:
            print(f"  {bad} of {len(by_t)} probes fall BELOW the target at this threshold: "
                  f"the pooled number hides an unsafe trough.")
        else:
            print("  the threshold holds at every probe.")

    print("\n=== does confidence separate right from wrong at all? ===")
    for stat in STAT_NAMES:
        ok = [getattr(r["a"], stat) for r in rows if r["correct"]]
        no = [getattr(r["a"], stat) for r in rows if not r["correct"]]
        if ok and no:
            print(f"  {stat:>12s}: correct {st.mean(ok):.3f} vs wrong {st.mean(no):.3f} "
                  f"(separation {st.mean(ok) - st.mean(no):+.3f})")

    print("\n=== the shared-delusion check ===")
    if not aw:
        print("  no agree_wrong cases in this run - nothing to check")
    else:
        for stat in STAT_NAMES:
            v = [getattr(a, stat) for a in aw]
            allwrong = [getattr(r["a"], stat) for r in rows if not r["correct"]]
            allright = [getattr(r["a"], stat) for r in rows if r["correct"]]
            print(f"  {stat:>12s}: agree_wrong {st.mean(v):.3f} (n={len(v)})  "
                  f"other wrong {st.mean(allwrong):.3f}  correct {st.mean(allright):.3f}")
        print("\n  If agree_wrong confidence sits up with the CORRECT answers, the gate")
        print("  inherits the monitor's blind spot and neither mechanism suffices alone.")


if __name__ == "__main__":
    main()

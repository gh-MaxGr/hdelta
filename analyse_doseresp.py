"""Writability dose-response: how much non-decomposability blocks a write?

k of 24 table cells depend on the (zone, cluster) pair; the rest factor as f(cluster). The
rule text is 552 chars with 6 labels at every k, so decomposability is the only variable.
k=0 is map_wide, k=24 is conjunction.

Scored as lift over each run's own answer prior, mean over the second half of probes.

    python analyse_doseresp.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from hdelta.tasks import LogicTask

ROOT = Path("results/doseresp")


def one(d: Path):
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    spec = cfg["task"]["rules"]["j"]
    k, seed = int(spec.get("k", 0)), int(cfg.get("seed", 0))
    task = LogicTask({"j": spec})
    base = task.balance("j")["majority_share"]
    recs = [json.loads(l) for l in (d / "probes.jsonl").open(encoding="utf-8")
            if '"metrics"' in l]
    acc = [r["metrics"].get("j/zero_direct_acc") for r in recs]
    acc = [a for a in acc if a is not None]
    if not acc:
        return None
    late = acc[len(acc) // 2:]
    m = st.mean(late)
    return {"k": k, "seed": seed, "baseline": base, "mean": m, "sd": st.pstdev(late),
            "max": max(acc), "lift": (m - base) / (1 - base),
            "lift_max": (max(acc) - base) / (1 - base), "traj": acc}


rows = [r for d in sorted(ROOT.iterdir()) if (d / "probes.jsonl").exists()
        for r in [one(d)] if r]
if not rows:
    print("no results yet")
    raise SystemExit

rows.sort(key=lambda r: (r["k"], r["seed"]))
print(f"{'k':>4s}{'seed':>6s}{'non-dec':>9s}{'base':>7s}{'mean':>7s}{'sd':>6s}"
      f"{'max':>6s}{'lift':>8s}{'lift@max':>10s}")
for r in rows:
    print(f"{r['k']:4d}{r['seed']:6d}{r['k']/24:9.0%}{r['baseline']:7.3f}{r['mean']:7.3f}"
          f"{r['sd']:6.3f}{r['max']:6.2f}{r['lift']:+8.3f}{r['lift_max']:+10.3f}")

by_k = {}
for r in rows:
    by_k.setdefault(r["k"], []).append(r["lift"])
print("\n=== dose-response (mean lift across available seeds) ===")
for k in sorted(by_k):
    v = by_k[k]
    bar = "#" * max(0, int(round(40 * max(0.0, st.mean(v)))))
    print(f"  k={k:2d} ({k/24:4.0%} non-decomposable)  lift {st.mean(v):+.3f}"
          f"{'' if len(v) < 2 else f' +- {st.pstdev(v):.3f}'}  n={len(v)}  {bar}")

ends = {k: by_k[k] for k in (0, 24) if k in by_k}
if len(ends) == 2 and all(len(v) > 1 for v in ends.values()):
    a, b = st.mean(ends[0]), st.mean(ends[24])
    print(f"\n  endpoints across {len(ends[0])} seeds: k=0 {a:+.3f} +- {st.pstdev(ends[0]):.3f}"
          f"   k=24 {b:+.3f} +- {st.pstdev(ends[24]):.3f}   delta {b - a:+.3f}")

print("\n=== trajectories ===")
for r in rows:
    print(f"  k={r['k']:2d} s{r['seed']}  " + " ".join(f"{a:.2f}" for a in r["traj"]))
Path("results/doseresp/summary.json").write_text(
    json.dumps([{k: v for k, v in r.items() if k != "traj"} for r in rows], indent=2),
    encoding="utf-8")

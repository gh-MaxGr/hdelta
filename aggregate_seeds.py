"""Part D across seeds: which numbers are stable and which were one draw?

The qualitative findings are mechanistic and should not move.  The coverage figures are
sensitive to stream ordering and fast-weight initialisation, so they need a spread rather
than a point estimate.  This reports mean +- sd over seeds for the claims Part D makes,
and flags any that flip sign or cross a decision boundary between seeds.

    python aggregate_seeds.py results/gate3_gated results/gate3_gated_s1 results/gate3_gated_s2
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

from hdelta.tasks import make_task
from hdelta.toolcache import STAT_NAMES, Answer, gate_curve


def per_run(run: Path, t_min: int = 700):
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    task = make_task(cfg["task"])
    seed = int(cfg.get("seed", 0))
    rules = task.rule_names()
    probe_n = int(cfg["stream"]["probe_n"])
    order, ov = {}, set()
    for i, r in enumerate(rules):
        rng = random.Random(10_000 + 17 * i + seed)
        order[r] = [task.sample(r, rng).input_str for _ in range(probe_n)]
        for x in order[r]:
            if len(task.tools[r].trace(x)) > 2:
                ov.add((r, x))

    recs = [json.loads(l) for l in (run / "probes.jsonl").open(encoding="utf-8")
            if '"metrics"' in l]
    rows, acc_bands = [], {}
    for rec in recs:
        if rec["t"] < t_min:
            continue
        for r, pp in rec.get("per_puzzle", {}).items():
            conf, direct = pp.get("conf"), pp.get("direct")
            if not conf or not direct:
                continue
            awi = set(pp.get("agree_wrong_idx") or [])
            acc = sum(direct) / len(direct)
            band = min(int(acc * 5), 4)
            aw_rate = len(awi) / len(direct) if awi is not None else 0.0
            if pp.get("agree_wrong_idx") is not None:
                acc_bands.setdefault(band, []).append(aw_rate)
            for i, (c, ok) in enumerate(zip(conf, direct)):
                rows.append({"a": Answer("", c.get("probs", []), c.get("margins", [])),
                             "correct": bool(ok), "ov": (r, order[r][i]) in ov,
                             "aw": i in awi})
    return cfg, rows, acc_bands


def gate(rows, target):
    correct = [r["correct"] for r in rows]
    best = None
    for s in STAT_NAMES:
        v = [getattr(r["a"], s) for r in rows]
        row = gate_curve(v, correct, targets=(target,))[0]
        if row["meets_target"] and (best is None or row["coverage"] > best["coverage"]):
            best = {**row, "stat": s}
    return best


def main():
    runs = [Path(a) for a in sys.argv[1:]] or [Path("results/gate3_gated")]
    runs = [r for r in runs if (r / "probes.jsonl").exists()]
    print(f"aggregating {len(runs)} run(s): {', '.join(r.name for r in runs)}\n")

    acc, ov_err, ord_err, ov_aw, ord_aw, aw_share = [], [], [], [], [], []
    cov = {0.90: [], 0.95: [], 0.99: []}
    bands = {}
    for run in runs:
        cfg, rows, ab = per_run(run)
        ordinary = [r for r in rows if not r["ov"]]
        override = [r for r in rows if r["ov"]]
        acc.append(st.mean(r["correct"] for r in rows))
        ov_err.append(1 - st.mean(r["correct"] for r in override))
        ord_err.append(1 - st.mean(r["correct"] for r in ordinary))
        ov_aw.append(st.mean(r["aw"] for r in override))
        ord_aw.append(st.mean(r["aw"] for r in ordinary))
        wrong = [r for r in rows if not r["correct"]]
        aw_share.append(st.mean(r["aw"] for r in wrong) if wrong else float("nan"))
        for b, v in ab.items():
            bands.setdefault(b, []).append(st.mean(v))
        for tgt in cov:
            g = gate(ordinary, tgt)
            cov[tgt].append(g["coverage"] * len(ordinary) / len(rows) if g else 0.0)
        print(f"  {run.name}: seed {cfg.get('seed')}, {len(rows)} queries, "
              f"accuracy {acc[-1]:.3f}")

    def line(name, xs, pct=True):
        f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
        sd = st.pstdev(xs) if len(xs) > 1 else 0.0
        vals = " ".join(f(x) for x in xs)
        print(f"  {name:38s} {f(st.mean(xs)):>8s} +- {f(sd):>7s}   [{vals}]")

    print("\n=== structural claims (expected stable) ===")
    line("override-eligible error rate", ov_err)
    line("ordinary error rate", ord_err)
    line("agree_wrong on override inputs", ov_aw)
    line("agree_wrong on ordinary inputs", ord_aw)
    line("share of errors that are agree_wrong", aw_share)

    print("\n=== blind spot vs skill quality (expected monotonic) ===")
    for b in sorted(bands):
        line(f"  accuracy band {b*20}-{b*20+20}%", bands[b], pct=False)

    print("\n=== coverage claims (expected to move) ===")
    line("compiled accuracy", acc, pct=False)
    for tgt in (0.90, 0.95, 0.99):
        line(f"total tool calls avoided @ {tgt:.2f}", cov[tgt])

    print("\n=== decision boundaries ===")
    if all(x == 0.0 for x in ord_aw):
        print("  HOLDS: ordinary inputs carry zero blind-spot errors in every seed")
    else:
        print(f"  MOVED: ordinary agree_wrong is not zero in every seed -> {ord_aw}")
    if all(x > 0.9 for x in ov_err):
        print("  HOLDS: override rule never compiles (>90% error in every seed)")
    else:
        print(f"  MOVED: override error rate varies -> {ov_err}")
    if len(bands) >= 4:
        means = [st.mean(bands[b]) for b in sorted(bands)]
        print("  HOLDS: blind spot grows with accuracy" if means == sorted(means)
              else f"  MOVED: blind spot not monotonic across seeds -> {means}")


if __name__ == "__main__":
    main()

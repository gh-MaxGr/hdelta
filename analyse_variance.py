"""Does the friction coefficient survive true sampling variance?

Everything measured so far is either n=1 or n=2 same-seed, which only exercises bf16
numerical noise. These runs vary the SEED, which changes the probe set and the stream order
as well - the variance a published coefficient actually has to survive.

Reports, per family: W at each threshold for every seed, the mean and spread, and - the thing
that decides whether WriteBench emits scalars or distributions - whether the ORDINAL hierarchy
survives. A coefficient with a +-300 write interval is still useful if the ordering is stable;
it is useless if the intervals overlap enough to reorder the ladder.

    python analyse_variance.py results/variance results/ladder results/ladder_long
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

from analyse_complexity import THETAS, crossing, series


def collect(roots):
    """{family: {seed: [W per theta]}} over every run found."""
    out = {}
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not ((d / "probes.jsonl").exists() and (d / "config.json").exists()):
                continue
            # Only COMPLETED runs. An in-flight run scores "never" at every threshold and
            # would masquerade as a structure that failed to compile.
            if not (d / "summary.json").exists():
                continue
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            fam, pts, budget, pe = series(d)
            if budget < 1800:          # short runs cannot resolve the expensive families
                continue
            # Runs are only replicates of each other if the WRITE LOCUS matches - pooling an
            # MLP run with attention runs compares two experimental conditions as if they
            # were repeats of one.
            groups = cfg["plastic"]["groups"]
            locus = "+".join(sorted(
                ("mlp" if "mlp" in g["patterns"][0] else "attn") + str(len(g["layers"]))
                for g in groups.values()))
            seed = int(cfg.get("seed", 0))
            out.setdefault((fam, locus), {})[(seed, d.name)] = [
                crossing(pts, t, pe) for t in THETAS]
    return out


def fmt(x):
    return "never" if x == float("inf") else f"{int(x)}"


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path("results/variance")]
    data = collect(roots)
    if not data:
        print("no completed 1800-puzzle runs found")
        return

    for key in sorted(data):
        fam, locus = key
        runs = data[key]
        print(f"\n=== {fam}  [locus {locus}]  ({len(runs)} completed runs) ===")
        print(f"  {'run':>26s}" + "".join(f"{'W@' + str(t):>10s}" for t in THETAS))
        for (seed, name), ws in sorted(runs.items()):
            print(f"  {name:>26s}" + "".join(f"{fmt(w):>10s}" for w in ws))
        for i, t in enumerate(THETAS):
            vals = [ws[i] for ws in runs.values() if ws[i] != float("inf")]
            n_never = sum(1 for ws in runs.values() if ws[i] == float("inf"))
            if not vals:
                print(f"  theta={t}: never reached in any run")
                continue
            spread = f" sd {st.pstdev(vals):.0f}" if len(vals) > 1 else ""
            ratio = f"  max/min {max(vals)/min(vals):.2f}x" if len(vals) > 1 else ""
            never = f"  ({n_never} run(s) never reached it)" if n_never else ""
            print(f"  theta={t}: mean {st.mean(vals):.0f}{spread}  "
                  f"range {min(vals):.0f}-{max(vals):.0f}{ratio}{never}")

    # the decisive question: does the ORDERING survive, even if the numbers do not?
    print("\n=== does the ordinal hierarchy survive? ===")
    for i, t in enumerate(THETAS):
        per_fam = {}
        for (fam, locus), runs in data.items():
            if locus != "attn9":        # compare like with like
                continue
            vals = [ws[i] for ws in runs.values() if ws[i] != float("inf")]
            if vals:
                per_fam[fam] = (min(vals), max(vals), st.mean(vals))
        if len(per_fam) < 2:
            continue
        order = sorted(per_fam, key=lambda f: per_fam[f][2])
        print(f"\n  theta={t}: " + " < ".join(order))
        # an inversion is possible wherever two families' observed ranges overlap
        overlaps = []
        for a, b in zip(order, order[1:]):
            lo_a, hi_a, _ = per_fam[a]
            lo_b, hi_b, _ = per_fam[b]
            if hi_a >= lo_b:
                overlaps.append(f"{a}({lo_a:.0f}-{hi_a:.0f}) / {b}({lo_b:.0f}-{hi_b:.0f})")
        if overlaps:
            print("    RANGES OVERLAP, ordering not established between: " + "; ".join(overlaps))
        else:
            print("    all adjacent pairs separated - ordering holds across seeds")

    print("\n  Ordering stable + wide intervals -> WriteBench publishes distributions and an")
    print("  ordinal hierarchy. Ordering unstable -> it publishes neither, and the structural")
    print("  claim needs more seeds before it means anything.")


if __name__ == "__main__":
    main()

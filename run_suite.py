"""Run the ablation suite, then produce per-run figures and a comparison table.

    python run_suite.py --out results/suite                      # all ablations
    python run_suite.py --out results/suite --only full no_rem   # a subset
    python run_suite.py --out results/suite --plots-only         # re-plot without re-running
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hdelta.experiment import Experiment
from hdelta.plots import fig_suite, plot_run
from run_experiment import load_config

ABLATIONS = ["full", "no_decay", "no_homeo", "no_rem", "none", "hebbian", "frozen"]


def _f(v, nd=2):
    if v is None:
        return "never"
    if isinstance(v, float):
        return "n/a" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def suite_table(out: Path, names) -> str:
    rows = ["| run | A: t50 slow -> direct | A: t_compile | A: final direct | B: t50 slow -> direct | B: t_unhook | B: old-rule peak | B: t_dissolve | B: final direct | A again: t_compile | ppl drift max | cloze final | max fast-weight ratio | REM acc cost |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        p = out / n / "summary.json"
        if not p.exists():
            rows.append(f"| {n} | (missing) |" + " |" * 12)
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        ph = s["phases"]
        p0 = ph[0] if len(ph) > 0 else {}
        p1 = ph[1] if len(ph) > 1 else {}
        p2 = ph[2] if len(ph) > 2 else {}
        fw = max((v["norm_ratio_max"] or 0) for v in s["fast_weights"].values()) if s["fast_weights"] else 0
        rows.append("| {n} | {a50} | {a_tc} | {a_fa} | {b50} | {b_tu} | {b_op} | {b_td} | {b_fa} | {a2} | {ppl:+.1f}% | {cl} | {fw:.3f} | {rc} |".format(
            n=n, a50=f"{_f(p0.get('t50_slow_path'))} -> {_f(p0.get('t50_compile'))}",
            b50=f"{_f(p1.get('t50_slow_path'))} -> {_f(p1.get('t50_compile'))}",
            a_tc=_f(p0.get("t_compile")), a_fa=_f(p0.get("final_zero_direct_acc")),
            b_tu=_f(p1.get("t_unhook")), b_op=_f(p1.get("oldrule_rate_peak")), b_td=_f(p1.get("t_dissolve")),
            b_fa=_f(p1.get("final_zero_direct_acc")), a2=_f(p2.get("t_compile")),
            ppl=s["base_skills"]["ppl_drift_max_pct"], cl=_f(s["base_skills"]["cloze_final"]), fw=fw,
            rc=_f(s["rem"]["mean_acc_cost"])))
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="results/suite")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="extra override(s) applied before each ablation, e.g. configs/instructed_3b.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.only or ABLATIONS
    if not args.plots_only:
        for n in names:
            override = f"configs/ablations/{n}.yaml"
            cfg = load_config(args.config, args.override + [override], args.set)
            print(f"\n##### {n} #####", flush=True)
            Experiment(cfg, out / n).run()
    for n in names:
        if (out / n / "probes.jsonl").exists():
            plot_run(out / n)
    fig_suite([out / n for n in names], out)
    table = suite_table(out, names)
    from hdelta.handover import analyse_run, suite_table as handover_table
    for n in names:
        if (out / n / "probes.jsonl").exists() and not (out / n / "handover.json").exists():
            analyse_run(out / n)
    htable = handover_table([out / n for n in names])
    (out / "suite_summary.md").write_text("# Ablation suite\n\n" + table + "\n\n## Hand-over tracking\n\n"
                                          + htable + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()

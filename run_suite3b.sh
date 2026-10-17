#!/usr/bin/env bash
# 3B suite (v2): three seeds of `full` at probe cadence 20 with all 48 puzzles in the
# generated/free probes, then five ablations at cadence 40.  Sequential (one 3B run
# holds ~9 GB).  Post-processing: hand-over analyses, synthetic cadence-40 views of the
# full runs, seed aggregation, figures and tables.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
OUT=results/suite3b_v2
mkdir -p "$OUT"
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "suite start"
for s in 0 1 2; do
  say "full_s$s start (cadence 20, free_n 48)"
  $PY run_experiment.py --config configs/base.yaml --override configs/instructed_3b.yaml --override configs/ablations/full.yaml \
     --set seed=$s --set stream.probe_every=20 --set stream.free_n=48 --out "$OUT/full_s$s" > "$OUT/full_s$s.log" 2>&1
  say "full_s$s exit $?"
done
for a in no_rem no_decay no_homeo none frozen; do
  say "$a start (cadence 40, free_n 32)"
  $PY run_experiment.py --config configs/base.yaml --override configs/instructed_3b.yaml --override "configs/ablations/$a.yaml" \
     --out "$OUT/$a" > "$OUT/$a.log" 2>&1
  say "$a exit $?"
done

say "post-processing"
$PY - <<'PYEOF'
from pathlib import Path
from hdelta.handover import analyse_run, aggregate_seeds, suite_table
from hdelta.plots import plot_run, fig_suite
out = Path("results/suite3b_v2")
fulls = [out / f"full_s{s}" for s in range(3)]
abls = [out / a for a in ["no_rem", "no_decay", "no_homeo", "none", "frozen"]]
for d in fulls + abls:
    if (d / "probes.jsonl").exists():
        analyse_run(d)
        plot_run(d)
for d in fulls:
    if (d / "probes.jsonl").exists():
        analyse_run(d, cadence=40, suffix="_c40")
aggregate_seeds([d for d in fulls if (d / "handover.json").exists()], out, name="full")
fig_suite([fulls[0]] + abls, out)
md = ["# 3B suite (v2): hand-over tracking", "",
      "## Native cadence (full runs at 20, ablations at 40)", "", suite_table(fulls + abls), "",
      "## Full runs on the synthetic cadence-40 grid (quantisation check)", "", suite_table(fulls, suffix="_c40"), "",
      "## Seed aggregation (full)", "", (out / "handover_seeds_full.md").read_text(encoding="utf-8")]
(out / "suite_handover.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("post-processing done")
PYEOF
$PY run_suite.py --out "$OUT" --only full_s0 no_rem no_decay no_homeo none frozen --plots-only > "$OUT/plots.log" 2>&1
say "SUITE3B_V2_DONE"

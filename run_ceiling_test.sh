#!/usr/bin/env bash
# The true ceiling: does a 3x plastic slice hold 3x the skills, or does capacity plateau?
#
# Identical to the original 8-rule run (configs/capacity_c80_n8.yaml: 8 tagged rules,
# 10-puzzle visits, 80-puzzle cycle, 320 puzzles per rule) with ONE change: the plastic
# slice goes from 3 layers / 3.1 % to 9 layers / 9.3 %. Per-synapse dynamics are untouched
# (eta 1e-4, decay 0.004, budget 0.03).
#
#   3 layers, 8 rules  -> summed 0.06 of 8   (measured)
#   9 layers, 3 rules  -> summed 2.47 of 3   (measured, but CEILING-LIMITED: 2 of 3 rules
#                                             pinned at 1.00, third one thrashing)
#   9 layers, 8 rules  -> this run. The ceiling is 8, so the metric cannot saturate.
#
# ~6 skills supports the scaling story; ~2-3 means capacity plateaus and dense gradients
# interfere structurally, which makes sparse routing unavoidable.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
OUT=results/capacity
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "ceiling test start: 9 layers, 8 rules"
$PY run_experiment.py --config configs/capacity_c80_n8.yaml --out "$OUT/d9_n8" \
    --set "plastic.max_fraction=0.15" \
    --set "plastic.groups.hdeltaH.layers=[-1,-2,-3,-4,-5,-6,-7,-8,-9]" \
    --set "plastic.groups.hdeltaG.layers=[-1,-2,-3,-4,-5,-6,-7,-8,-9]" \
    > "$OUT/d9_n8.log" 2>&1
code=$?
say "d9_n8 exit $code"
if [ $code -ne 0 ]; then
  grep -iE "out of memory|CUDA|Error" "$OUT/d9_n8.log" | head -3 | sed 's/^/    /' | tee -a "$LOG"
fi

say "post-processing ceiling test"
$PY - <<'PYEOF'
import json
from pathlib import Path
from hdelta.capacity import analyse
out = Path("results/capacity")
rows = ["| plastic slice | rules offered | summed compiled | rules held at once | per-rule finals | perplexity worst / final | cloze final |",
        "|---|---|---|---|---|---|---|"]
for name, slice_lab in [("n8", "3 layers, 3.1 %"), ("d9_n3", "9 layers, 9.3 %"), ("d9_n8", "9 layers, 9.3 %")]:
    d = out / name
    if not (d / "probes.jsonl").exists():
        rows.append(f"| {slice_lab} | (did not run) | | | | | |")
        continue
    o = analyse(d)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    per = sorted((o["final"][r]["compiled"] for r in o["rules"]), reverse=True)
    tot = sum(per)
    b = s["base_skills"]
    rows.append(f"| {slice_lab} | {o['n_rules']} | {tot:.2f} of {o['n_rules']} | {o['max_mastered']} | "
                + " ".join(f"{v:.2f}" for v in per) + f" | {b['ppl_drift_max_pct']:+.1f} % / {b['ppl_drift_final_pct']:+.1f} % | {b['cloze_final']:.2f} |")
table = "\n".join(rows)
(out / "ceiling_test.md").write_text(
    "# Does a 3x plastic slice hold 3x the skills?\n\nSame 8-rule schedule throughout; only the "
    "plastic slice changes. The 9-layer / 3-rule row is shown because it was ceiling-limited, which "
    "is what this run removes.\n\n" + table + "\n", encoding="utf-8")
print(table)
PYEOF
say "CEILING_TEST_DONE"

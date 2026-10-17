#!/usr/bin/env bash
# Does more plastic ENERGY push the N=3 cliff out, or is the limit geometric?
#
# Correction to the obvious design: raising `budget` alone does nothing. The homeostatic
# clamp is not what caps the fast weights in the normal configuration -- switching the
# clamp off entirely (suite3b_v2/no_homeo) still plateaus at 0.030, the same as the capped
# runs. The ceiling is the equilibrium between deposition (eta) and passive decay, and it
# scales roughly as eta/decay. So energy is scaled through eta, with the budget raised in
# step purely so it does not become the binding constraint.
#
#   1x  eta 1e-4, budget 0.03   already measured: summed 0.38 of 3
#   2x  eta 2e-4, budget 0.06
#   4x  eta 4e-4, budget 0.12
#
# Parameter count is held at 3.1 % plastic throughout, so the design spec's 95 %-frozen
# constraint still holds. Watch perplexity: buying capacity with energy is only a win if
# the base model survives it.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
OUT=results/capacity
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "budget sweep start (N=3, energy via eta)"
for spec in "e2x_n3:0.0002:0.06" "e4x_n3:0.0004:0.12"; do
  name="${spec%%:*}"; rest="${spec#*:}"; eta="${rest%%:*}"; bud="${rest##*:}"
  say "$name start (eta=$eta budget=$bud)"
  $PY run_experiment.py --config configs/capacity_b40_n3.yaml --out "$OUT/$name" \
      --set "plastic.groups.hdeltaH.eta=$eta" --set "plastic.groups.hdeltaG.eta=$eta" \
      --set "plastic.groups.hdeltaH.budget=$bud" --set "plastic.groups.hdeltaG.budget=$bud" \
      > "$OUT/$name.log" 2>&1
  say "$name exit $?"
done

say "post-processing sweep"
$PY - <<'PYEOF'
import json
from pathlib import Path
from hdelta.capacity import analyse
out = Path("results/capacity")
rows = ["| energy | eta | budget | summed compiled (of 3) | per rule | peak held | fw ratio reached | perplexity worst / final | cloze final |",
        "|---|---|---|---|---|---|---|---|---|"]
for name, eta, bud, lab in [("b40_n3", "1e-4", "0.03", "1x"), ("e2x_n3", "2e-4", "0.06", "2x"), ("e4x_n3", "4e-4", "0.12", "4x")]:
    d = out / name
    if not (d / "probes.jsonl").exists():
        continue
    o = analyse(d)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    fw = max(v["norm_ratio_max"] or 0 for v in s["fast_weights"].values())
    tot = sum(o["final"][r]["compiled"] for r in o["rules"])
    b = s["base_skills"]
    rows.append(f"| {lab} | {eta} | {bud} | {tot:.2f} | {tot/3:.3f} | {o['max_mastered']} of 3 | {fw:.3f} | "
                f"{b['ppl_drift_max_pct']:+.1f} % / {b['ppl_drift_final_pct']:+.1f} % | {b['cloze_final']:.2f} |")
table = "\n".join(rows)
(out / "budget_sweep.md").write_text("# Does more plastic energy push the N=3 cliff out?\n\n" + table + "\n", encoding="utf-8")
print(table)
PYEOF
say "BUDGET_SWEEP_DONE"

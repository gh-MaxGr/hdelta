#!/usr/bin/env bash
# Can a BIGGER SUBSPACE buy the third skill, where more amplitude could not?
#
# The energy sweep scaled how hard each experience writes, inside a fixed 3.1 % plastic
# slice (15 modules over 3 layers). That is amplitude, not dimensionality. Here the
# per-synapse dynamics are held exactly fixed -- eta 1e-4, decay 0.004, budget 0.03, the
# same settings as the 1x run -- and only the NUMBER of plastic synapses changes, by
# extending the same module pattern down more layers.
#
#   3 layers   3.1 % plastic   already measured: summed 0.38 of 3
#   6 layers  ~6.2 % plastic
#   9 layers  ~9.3 % plastic
#
# This deliberately breaks the design spec's ">= 95 % frozen" constraint, which is part of the
# question: if capacity only scales by surrendering the frozen core, routing wins on the
# original terms anyway.
#
# Comparison against the energy sweep (same N=3, same task, same schedule):
#   3 layers @ 2x eta -> 0.59   vs   6 layers @ 1x eta   (more amplitude vs more directions)
#   3 layers @ 4x eta -> 0.09   vs   9 layers @ 1x eta
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
OUT=results/capacity
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "dimensionality sweep start (N=3, eta fixed at 1e-4)"
for spec in "d6_n3:-1,-2,-3,-4,-5,-6" "d9_n3:-1,-2,-3,-4,-5,-6,-7,-8,-9"; do
  name="${spec%%:*}"; layers="[${spec##*:}]"
  say "$name start (layers $layers)"
  $PY run_experiment.py --config configs/capacity_b40_n3.yaml --out "$OUT/$name" \
      --set "plastic.max_fraction=0.15" \
      --set "plastic.groups.hdeltaH.layers=$layers" \
      --set "plastic.groups.hdeltaG.layers=$layers" \
      > "$OUT/$name.log" 2>&1
  code=$?
  say "$name exit $code"
  if [ $code -ne 0 ]; then
    grep -iE "out of memory|CUDA|Error" "$OUT/$name.log" | head -2 | sed 's/^/    /' | tee -a "$LOG"
  fi
done

say "post-processing dimensionality sweep"
$PY - <<'PYEOF'
import json
from pathlib import Path
from hdelta.capacity import analyse
out = Path("results/capacity")
rows = ["| axis | setting | plastic params | % of model | summed compiled (of 3) | rules held | best single rule | perplexity worst / final | cloze final |",
        "|---|---|---|---|---|---|---|---|---|"]
spec = [("b40_n3", "dimensionality", "3 layers, eta 1e-4"),
        ("d6_n3", "dimensionality", "6 layers, eta 1e-4"),
        ("d9_n3", "dimensionality", "9 layers, eta 1e-4"),
        ("e2x_n3", "amplitude", "3 layers, eta 2e-4"),
        ("e4x_n3", "amplitude", "3 layers, eta 4e-4")]
for name, axis, label in spec:
    d = out / name
    if not (d / "probes.jsonl").exists():
        rows.append(f"| {axis} | {label} | (did not run) | | | | | | |")
        continue
    o = analyse(d)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    mi = json.loads((d / "model_info.json").read_text(encoding="utf-8"))
    tot = sum(o["final"][r]["compiled"] for r in o["rules"])
    best = max(o["final"][r]["best"] for r in o["rules"])
    b = s["base_skills"]
    rows.append(f"| {axis} | {label} | {mi['plastic_params']/1e6:.0f} M | {mi['plastic_fraction']:.1%} | {tot:.2f} | "
                f"{o['max_mastered']} of 3 | {best:.2f} | {b['ppl_drift_max_pct']:+.1f} % / {b['ppl_drift_final_pct']:+.1f} % | {b['cloze_final']:.2f} |")
table = "\n".join(rows)
(out / "dimension_sweep.md").write_text(
    "# Amplitude vs dimensionality at N=3\n\nPer-synapse dynamics identical across the "
    "dimensionality rows; only the number of plastic synapses changes.\n\n" + table + "\n", encoding="utf-8")
print(table)
PYEOF
say "DIMENSION_SWEEP_DONE"

#!/usr/bin/env bash
# Disentangle "number of rules" from "block length".
#
# The matched-cycle ladder varies both at once: (2,40) works, (4,20) fails, (8,10) fails,
# where the pairs are (rules, puzzles per visit). Two controls separate the causes:
#
#   b10_n2  (2 rules, 10-puzzle visits)  if this works, short visits are fine and the
#                                        rule count is what kills learning -> routing
#                                        if it fails, visit length is the driver and the
#                                        ladder's N effect is confounded
#   n4      (4 rules, 40-puzzle visits)  if this works, longer visits rescue 4 rules
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
OUT=results/capacity
mkdir -p "$OUT"
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "controls start"
for spec in "b10_n2:capacity_b10_n2" "b40_n4:capacity_n4"; do
  name="${spec%%:*}"; cfg="${spec##*:}"
  say "$name start ($cfg)"
  $PY run_experiment.py --config "configs/${cfg}.yaml" --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  say "$name exit $?"
done

say "post-processing controls"
$PY - <<'PYEOF'
from pathlib import Path
from hdelta.capacity import analyse, compare
out = Path("results/capacity")
names = ["n2", "b10_n2", "c80_n4" if (out / "c80_n4").exists() else "n4", "b40_n4", "n8"]
dirs = []
for n in ["n2", "b10_n2", "n4", "b40_n4", "n8"]:
    d = out / n
    if (d / "probes.jsonl").exists():
        analyse(d)
        dirs.append(d)
print(compare(dirs, out / "capacity_ladder.md"))
PYEOF
say "CONTROLS_DONE"

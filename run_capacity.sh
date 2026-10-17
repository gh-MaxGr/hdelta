#!/usr/bin/env bash
# Capacity ladder: how many skills can one global h-delta state hold at once?
#
# N tagged cipher rules are interleaved in 40-puzzle blocks that cycle, so every rule is
# repeatedly revisited and retention can be measured against time-since-last-trained.
# Order is 2 -> 8 -> 4: N=2 is the cheap baseline that proves the tagged task works at
# all, N=8 is the decisive test, N=4 locates the knee only if N=8 degrades.
#
# Default schedule is the MATCHED-CYCLE one (capacity_c80_n*): the block length shrinks
# as N grows so every rule is revisited every 80 puzzles regardless of how many rules are
# live. That holds passive decay constant across the ladder (76-85 % of a rule's trace
# survives each gap), so a failure at large N is interference and not decay. Set
# PREFIX=capacity_n for the realistic schedule, where the revisit gap grows with N.
#
#   bash run_capacity.sh            # full ladder, 2 then 8 then 4
#   bash run_capacity.sh 2 8        # a subset
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
PREFIX="${PREFIX:-capacity_c80_n}"
OUT=results/capacity
mkdir -p "$OUT"
LOG="$OUT/driver.log"
LADDER="${@:-2 8 4}"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "capacity ladder start: N = $LADDER, configs ${PREFIX}*.yaml"
for n in $LADDER; do
  say "n$n start"
  $PY run_experiment.py --config "configs/${PREFIX}${n}.yaml" --out "$OUT/n$n" > "$OUT/n$n.log" 2>&1
  code=$?
  say "n$n exit $code"
  [ $code -ne 0 ] && say "n$n FAILED, see $OUT/n$n.log"
done

say "post-processing"
$PY - "$LADDER" <<'PYEOF'
import sys
from pathlib import Path
from hdelta.capacity import analyse, compare
out = Path("results/capacity")
dirs = []
for n in sys.argv[1].split():
    d = out / f"n{n}"
    if (d / "probes.jsonl").exists():
        o = analyse(d)
        print(f"n{n}: peak {o['max_mastered']}/{o['n_rules']} held at once, "
              f"final {o['final_mastered']}, slope {o['interference_slope']}")
        dirs.append(d)
print(compare(dirs, out / "capacity_ladder.md"))
PYEOF
say "CAPACITY_DONE"

#!/usr/bin/env bash
# Two remaining questions, three runs. All at 9 layers, 8 rules, the configuration that
# sustains 4-5 skills with 0.000 cross-rule bleed (results/capacity/d9_n8).
#
# 1. ANATOMICAL ABLATION -- is the hdeltaH (attention) / hdeltaG (MLP) split real, or is
#    it flavour text?  Each group is run alone by emptying the other's layer list.
#    Note the two groups are NOT parameter-matched: attention is q/k/v/o (~30 % of the
#    slice) and MLP is down_proj alone (~70 %).  That is the point of the comparison --
#    if capacity tracks parameter count rather than which modules are plastic, there is
#    no functional split.  The reference points are 96 M (3 layers, both groups) -> 0.06
#    and 288 M (9 layers, both groups) -> 6.72.
#
# 2. CONFUSABLE CUES -- does the 0.000 bleed survive when the prompt cue stops being
#    distinctive?  Tags become K1..K8, which share a token and differ by one character,
#    instead of A7/M6/A3/...  The mappings are untouched, so attribution stays exact.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
OUT=results/capacity
LOG="$OUT/driver.log"
L9="[-1,-2,-3,-4,-5,-6,-7,-8,-9]"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run () {  # name, config, extra --set args
  local name="$1"; shift
  local cfg="$1"; shift
  say "$name start"
  $PY run_experiment.py --config "configs/${cfg}.yaml" --out "$OUT/$name" \
      --set "plastic.max_fraction=0.15" "$@" > "$OUT/$name.log" 2>&1
  local code=$?
  say "$name exit $code"
  [ $code -ne 0 ] && grep -iE "out of memory|Error" "$OUT/$name.log" | head -2 | sed 's/^/    /' | tee -a "$LOG"
  return 0
}

say "final two experiments start"
run attn_n8 capacity_c80_n8 --set "plastic.groups.hdeltaH.layers=$L9" --set "plastic.groups.hdeltaG.layers=[]"
run mlp_n8  capacity_c80_n8 --set "plastic.groups.hdeltaH.layers=[]" --set "plastic.groups.hdeltaG.layers=$L9"
run cue_n8  capacity_cue_n8 --set "plastic.groups.hdeltaH.layers=$L9" --set "plastic.groups.hdeltaG.layers=$L9"

say "post-processing"
$PY - <<'PYEOF'
import json
from pathlib import Path
import numpy as np
from hdelta.capacity import analyse
out = Path("results/capacity")
spec = [("n8", "both groups, 3 layers", "distinct"),
        ("attn_n8", "attention only, 9 layers", "distinct"),
        ("mlp_n8", "MLP only, 9 layers", "distinct"),
        ("d9_n8", "both groups, 9 layers", "distinct"),
        ("cue_n8", "both groups, 9 layers", "CONFUSABLE K1..K8")]
rows = ["| plastic modules | cue | plastic params | % of model | summed, last quarter | rules >= 0.8 (mean) | cross-rule bleed (mean/max) | perplexity final | cloze mean |",
        "|---|---|---|---|---|---|---|---|---|"]
for name, label, cue in spec:
    d = out / name
    if not (d / "probes.jsonl").exists():
        rows.append(f"| {label} | {cue} | (did not run) | | | | | | |")
        continue
    o = analyse(d)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    mi = json.loads((d / "model_info.json").read_text(encoding="utf-8"))
    A = np.array([o["acc"][r] for r in o["rules"]])
    q = max(1, A.shape[1] // 4)
    conf = [o["final"][r]["confusion_total"] for r in o["rules"]]
    rows.append("| {lab} | {cue} | {p:.0f} M | {f:.1%} | {tot:.2f} of {n} | {m:.1f} | {cm:.3f} / {cx:.3f} | {ppl:+.1f} % | {cz:.2f} |".format(
        lab=label, cue=cue, p=mi["plastic_params"] / 1e6, f=mi["plastic_fraction"],
        tot=A[:, -q:].sum(0).mean(), n=o["n_rules"], m=o["mean_mastered_last_quarter"],
        cm=float(np.mean(conf)), cx=float(max(conf)),
        ppl=s["base_skills"]["ppl_drift_final_pct"], cz=float(np.mean(o["cloze"]))))
table = "\n".join(rows)
(out / "final_two.md").write_text(
    "# Anatomical ablation and confusable cues\n\nAll runs: 8 interleaved rules, 10-puzzle visits, "
    "320 puzzles per rule, per-synapse dynamics identical (eta 1e-4, decay 0.004, budget 0.03).\n"
    "Capacity is the summed compiled accuracy averaged over the final quarter, which is the honest "
    "figure; the final probe alone is a single noisy draw.\n\n" + table + "\n", encoding="utf-8")
print(table)
PYEOF
say "FINAL_TWO_DONE"

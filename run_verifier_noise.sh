#!/usr/bin/env bash
# How fallible can the verifier be before the architecture stops working?
#
# Every result so far assumed a ground-truth oracle: only answers the model actually got
# right were consolidated. Real deployments have no such oracle. Here the verifier wrongly
# accepts a fraction of WRONG self-generated answers, and a falsely accepted answer is
# written back as the model's own -- on the slow path and in the zero-shot consolidation
# replay -- so the system is reinforcing its own mistakes.
#
# Architecture: attention-only, 9 layers, 85 M plastic (2.8 %), the configuration that
# sustains ~5.9 of 8 skills with a perfect verifier (results/capacity/attn_n8).
# Load: the same 8 interleaved rules, 10-puzzle visits, 320 puzzles per rule.
#
# Two failure modes are distinguished:
#   does the correct skill still compile?          -> summed compiled accuracy
#   does the network execute the reinforced error? -> poison/reproduces_error
# If passive decay filters infrequent false positives, low rates should be survivable.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
OUT=results/capacity
LOG="$OUT/driver.log"
L9="[-1,-2,-3,-4,-5,-6,-7,-8,-9]"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "verifier noise sweep start (attention-only, 8 rules)"
for fpr in 0.05 0.15 0.30; do
  name="fpr$(echo $fpr | tr -d '.')"
  say "$name start (false-positive rate $fpr)"
  $PY run_experiment.py --config configs/capacity_c80_n8.yaml --out "$OUT/$name" \
      --set "plastic.max_fraction=0.15" \
      --set "plastic.groups.hdeltaH.layers=$L9" --set "plastic.groups.hdeltaG.layers=[]" \
      --set "stream.verifier_fpr=$fpr" > "$OUT/$name.log" 2>&1
  say "$name exit $?"
done

say "post-processing noise sweep"
$PY - <<'PYEOF'
import json
from pathlib import Path
import numpy as np
from hdelta.capacity import analyse
out = Path("results/capacity")
rows = ["| verifier false-positive rate | skills sustained (of 8) | vs perfect oracle | reinforced errors reproduced | recovers truth on poisoned inputs | false accepts | perplexity final | cloze mean |",
        "|---|---|---|---|---|---|---|---|"]
base = None
for name, lab in [("attn_n8", "0 % (perfect oracle)"), ("fpr005", "5 %"), ("fpr015", "15 %"), ("fpr030", "30 %")]:
    d = out / name
    if not (d / "probes.jsonl").exists():
        rows.append(f"| {lab} | (did not run) | | | | | | |")
        continue
    o = analyse(d)
    s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    probes = [json.loads(l) for l in (d / "probes.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip() and '"metrics"' in l]
    A = np.array([o["acc"][r] for r in o["rules"]])
    q = max(1, A.shape[1] // 4)
    tot = A[:, -q:].sum(0).mean()
    if base is None:
        base = tot
    last = probes[-1]["metrics"]
    rep = last.get("poison/reproduces_error")
    rec = last.get("poison/recovers_truth")
    fa = last.get("poison/false_accepts_total", 0)
    rows.append("| {lab} | {tot:.2f} | {rel:+.0%} | {rep} | {rec} | {fa} | {ppl:+.1f} % | {cz:.2f} |".format(
        lab=lab, tot=tot, rel=(tot - base) / base if base else 0.0,
        rep="n/a" if rep is None else f"{rep:.3f}", rec="n/a" if rec is None else f"{rec:.2f}",
        fa=int(fa), ppl=s["base_skills"]["ppl_drift_final_pct"], cz=float(np.mean(o["cloze"]))))
table = "\n".join(rows)
(out / "verifier_noise.md").write_text(
    "# How fallible can the verifier be?\n\nAttention-only, 9 layers, 85 M plastic; 8 interleaved rules, "
    "320 puzzles per rule. A falsely accepted answer is consolidated as the model's own wrong answer.\n"
    "'Reinforced errors reproduced' is the share of poisoned inputs for which the compiled path now "
    "returns the wrong answer that was written into it.\n\n" + table + "\n", encoding="utf-8")
print(table)
PYEOF
say "VERIFIER_NOISE_DONE"

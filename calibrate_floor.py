"""How reliable is the deliberate path, really, and what moves it?

One draw put it at 0.958 and two others at 0.73 and 0.85 - too far apart for sampling
noise at n=48.  The monitor's floor is 1 - reliability, so if that number is unstable
the threshold is unstable with it and healthy skills get rolled back.

Hypothesis tested here: the shots decide it.  The override patterns (**/*_test.ext)
fire on a minority of inputs, so a shot set that happens to contain no override case
never demonstrates the last-match-wins behaviour the rule text describes.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys

import yaml

from hdelta.encoding import Encoder
from hdelta.modelwrap import load_model
from hdelta.tasks import make_task
from hdelta.toolcache import CompiledMonitor, ToolVerifier

cfg = yaml.safe_load(open(os.environ.get("PROBE_CONFIG", "configs/workspace.yaml"),
                          encoding="utf-8"))
task = make_task(cfg["task"])
name = sys.argv[1] if len(sys.argv) > 1 else cfg["model"]["name"]
n_shots = int(cfg["stream"].get("n_shots", 2))

model, tok = load_model(name, cfg["model"].get("dtype", "bfloat16"), "cuda")
enc = Encoder(tok, "cuda")
mon = CompiledMonitor(task, enc)


def override_in(rule, shots):
    """Does any shot exercise an override (more than the catch-all + package match)?"""
    return any(len(task.tools[rule].trace(s.input_str)) > 2 for s in shots)


rows = []
for i in range(8):
    rule = "AB"[i % 2]
    rng = random.Random(1000 + 37 * i)
    pz = [task.sample(rule, rng) for _ in range(48)]
    shots = [task.sample(rule, rng) for _ in range(n_shots)]
    v = ToolVerifier(task, mode="honest")
    answers = mon._deliberate_pass(model, pz, shots)
    ok = sum(v.check(rule, p.input_str, a).correct for p, a in zip(pz, answers))
    # how many probe inputs actually need the override
    need = sum(1 for p in pz if len(task.tools[rule].trace(p.input_str)) > 2)
    hit = override_in(rule, shots)
    acc = ok / len(pz)
    rows.append({"draw": i, "rule": rule, "acc": acc, "shots_show_override": hit,
                 "probe_override_share": need / len(pz)})
    print(f"draw {i} [{rule}] acc={acc:.3f}  shots_show_override={hit}  "
          f"probe_needs_override={need/len(pz):.2f}", flush=True)

accs = [r["acc"] for r in rows]
with_o = [r["acc"] for r in rows if r["shots_show_override"]]
without_o = [r["acc"] for r in rows if not r["shots_show_override"]]
print(f"\ndeliberate reliability over {len(accs)} draws: "
      f"mean={statistics.mean(accs):.3f} sd={statistics.pstdev(accs):.3f} "
      f"min={min(accs):.3f} max={max(accs):.3f}")
if with_o and without_o:
    print(f"  shots include an override case: mean={statistics.mean(with_o):.3f} (n={len(with_o)})")
    print(f"  shots do not:                   mean={statistics.mean(without_o):.3f} (n={len(without_o)})")
print(f"\nsuggested floor = 1 - mean = {1 - statistics.mean(accs):.3f}; "
      f"worst-draw floor = {1 - min(accs):.3f}")
print(f"tool calls used: {task.tool_calls()}")

os.makedirs("results/gate1", exist_ok=True)
with open("results/gate1/floor_calibration.json", "w", encoding="utf-8") as f:
    json.dump({"model": name, "rows": rows, "mean": statistics.mean(accs),
               "sd": statistics.pstdev(accs), "min": min(accs), "max": max(accs)}, f, indent=2)

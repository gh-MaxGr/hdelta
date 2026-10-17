"""Negative control for the monitor, on the FROZEN model, before any plasticity.

With no fast weights the compiled reflex does not exist, so the monitor must report
a large divergence and call the skill unhealthy.  If it does not, the monitor is not
measuring what it claims to and nothing downstream of it can be trusted.

    python check_monitor.py [model]        (PROBE_CONFIG selects the task)
"""
from __future__ import annotations

import json
import os
import random
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

model, tok = load_model(name, cfg["model"].get("dtype", "bfloat16"), "cuda")
enc = Encoder(tok, "cuda")
monitor = CompiledMonitor(task, enc, threshold=0.15)
n_shots = int(cfg["stream"].get("n_shots", 2))

# Calibrate the floor on the SAME fixed sets the monitor then uses.  The deliberate path
# swings from 0.52 to 0.94 across draws, so a floor borrowed from another draw is worthless;
# on the monitored set it is the reference's measured error on exactly those inputs.
cal_v = ToolVerifier(task, mode="honest")
probe_sets = {}
for rule in ("A", "B"):
    rng = random.Random(10_000 + ord(rule))
    probe_sets[rule] = ([task.sample(rule, rng) for _ in range(48)],
                        [task.sample(rule, rng) for _ in range(n_shots)])
    rel_r = monitor.calibrate(model, rule, *probe_sets[rule], cal_v)
    print(f"[{rule}] deliberate reliability = {rel_r:.3f} -> floor {1 - rel_r:.3f}", flush=True)
rel = monitor.reliability
print(f"({cal_v.calls} tool calls to establish)\n", flush=True)

results = {}
for rule in ("A", "B"):
    pz, shots = probe_sets[rule]
    r = monitor.check(model, rule, pz, shots)
    print(r.summary(), flush=True)
    results[rule] = {k: v for k, v in r.__dict__.items() if k not in ("compiled", "deliberate")}
    print(f"     compiled sample : {r.compiled[:3]}")
    print(f"     deliberate      : {r.deliberate[:3]}", flush=True)

# The verifier, on the model's own deliberate output: what a training chunk would write.
v = ToolVerifier(task, mode="honest")
rng = random.Random(5)
pz = [task.sample("A", rng) for _ in range(48)]
shots = [task.sample("A", rng) for _ in range(n_shots)]
answers = monitor._deliberate_pass(model, pz, shots)
for p, a in zip(pz, answers):
    v.check("A", p.input_str, a)
print("\nverifier on the deliberate path:", json.dumps(v.state()), flush=True)
print(f"task tool calls so far: {task.tool_calls()} "
      f"({task.simulated_ms()/1000:.0f}s simulated at {cfg['task']['latency_ms']}ms each)")

os.makedirs("results/gate1", exist_ok=True)
with open("results/gate1/monitor_control.json", "w", encoding="utf-8") as f:
    json.dump({"model": name, "frozen": True, "deliberate_reliability": rel,
               "rules": results, "verifier": v.state()}, f, indent=2)

bad = [r for r, d in results.items() if d["healthy"]]
print("\nCONTROL FAILED - monitor called a non-existent reflex healthy:", bad) if bad else \
    print("\ncontrol passed: monitor flags the frozen model as UNHEALTHY on every rule")

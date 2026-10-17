"""Would a bloom filter of valid build targets catch the reflex's errors?

A bloom filter can only reject a label that does not exist.  If the reflex's wrong answers
are *real* targets applied to the wrong input, the filter contains them and catches nothing.
This reloads the compiled state from delta.pt, regenerates the direct pass on every probe
set, and classifies each wrong answer as:

  nonexistent   not a target in ANY workspace        -> a bloom filter rejects it
  wrong_ws      a real target, but another workspace  -> a PER-WORKSPACE filter rejects it
  in_ws         a real target of the right workspace  -> no filter can reject it

    python check_bloom.py results/gate3_gated
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

from hdelta.encoding import Encoder
from hdelta.modelwrap import groups_from_config, install_hdelta, load_model
from hdelta.tasks import make_task
from hdelta.toolcache import CompiledMonitor, generate_with_confidence

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gate3_gated")
cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
task = make_task(cfg["task"])
rules = task.rule_names()

model, tok = load_model(cfg["model"]["name"], cfg["model"].get("dtype", "bfloat16"), "cuda")
ctrl, _ = install_hdelta(model, groups_from_config(cfg["plastic"]["groups"]),
                         max_fraction=cfg["plastic"].get("max_fraction", 0.05),
                         postsynaptic=cfg["plastic"].get("postsynaptic", "error"),
                         decay_enabled=True, homeo_enabled=True,
                         update_norm=cfg["plastic"].get("update_norm", "adam"))
snap = torch.load(run / "delta.pt", map_location="cuda")
ctrl.restore({k: v.to("cuda") for k, v in snap.items()})
ctrl.invalidate()
enc = Encoder(tok, "cuda")
print(f"restored dW from {run/'delta.pt'}: "
      f"{sum(float(v.norm())**2 for v in snap.values())**0.5:.4f} total norm", flush=True)

# the label vocabularies a filter could be built from
per_ws = {r: {t for _, t in task.tables[r]} for r in rules}
all_labels = set().union(*per_ws.values())
print(f"{len(all_labels)} distinct valid targets across {len(rules)} workspaces\n")

probe_n = int(cfg["stream"]["probe_n"])
seed = int(cfg.get("seed", 0))
mon = CompiledMonitor(task, enc, ctrl=ctrl, deliberate_frozen=True)
n_shots = int(cfg["stream"].get("n_shots", 2))
kinds, examples = Counter(), {}
aw_kinds = Counter()
err_by_ov, aw_by_ov, tot_by_ov = Counter(), Counter(), Counter()
n_wrong = n_tot = 0
for i, r in enumerate(rules):
    rng = random.Random(10_000 + 17 * i + seed)
    pz = [task.sample(r, rng) for _ in range(probe_n)]
    prompts = [task.make_pair(p, "direct")[0] for p in pz]
    ans = generate_with_confidence(model, enc, prompts, 16)
    # the frozen reference on the same inputs, to find the agree_wrong (blind-spot) cases
    shots = [task.sample(r, rng) for _ in range(n_shots)]
    delib = mon._deliberate_pass(model, pz, shots)
    for p, a, d in zip(pz, ans, delib):
        n_tot += 1
        tot_by_ov[len(task.tools[r].trace(p.input_str)) > 2] += 1
        if a.text == p.answer:
            continue
        n_wrong += 1
        if a.text not in all_labels:
            k = "nonexistent"
        elif a.text in per_ws[r]:
            k = "in_ws"
        else:
            k = "wrong_ws"
        kinds[k] += 1
        if a.text == d:                 # both paths agree on the same wrong answer
            aw_kinds[k] += 1
        examples.setdefault(k, []).append((p.input_str, p.answer, a.text))
        ov = len(task.tools[r].trace(p.input_str)) > 2
        err_by_ov[ov] += 1
        if a.text == d:
            aw_by_ov[ov] += 1

print(f"{n_tot} queries, {n_wrong} wrong ({n_wrong/n_tot:.1%})\n")
print("what a bloom filter would see:")
for k in ("nonexistent", "wrong_ws", "in_ws"):
    v = kinds.get(k, 0)
    print(f"  {k:12s} {v:5d}  {v/max(1,n_wrong):6.1%}")
print()
catch_all = kinds.get("nonexistent", 0) / max(1, n_wrong)
catch_ws = (kinds.get("nonexistent", 0) + kinds.get("wrong_ws", 0)) / max(1, n_wrong)
print(f"global filter of all valid targets catches : {catch_all:.1%} of errors")
print(f"per-workspace filter catches               : {catch_ws:.1%} of errors")
print()
n_aw = sum(aw_kinds.values())
print(f"=== of the {n_aw} AGREE_WRONG errors (the monitor-blind ones) ===")
for k in ("nonexistent", "wrong_ws", "in_ws"):
    v = aw_kinds.get(k, 0)
    print(f"  {k:12s} {v:5d}  {v/max(1,n_aw):6.1%}")
aw_catch = (aw_kinds.get("nonexistent", 0) + aw_kinds.get("wrong_ws", 0)) / max(1, n_aw)
print(f"  per-workspace filter catches {aw_catch:.1%} of the BLIND-SPOT errors")
print()
print("=== errors split by whether the INPUT is override-eligible (a regex can tell) ===")
for ov in (True, False):
    n, e, a_ = tot_by_ov[ov], err_by_ov[ov], aw_by_ov[ov]
    lab = "override-eligible" if ov else "ordinary"
    print(f"  {lab:18s} n={n:4d}  errors {e:3d} ({e/max(1,n):5.1%})  "
          f"agree_wrong {a_:3d} ({a_/max(1,n):5.1%})")
tot_aw = sum(aw_by_ov.values())
print(f"  -> {aw_by_ov[True]}/{tot_aw} of blind-spot errors sit on override-eligible inputs "
      f"({aw_by_ov[True]/max(1,tot_aw):.0%}), which are {tot_by_ov[True]/max(1,sum(tot_by_ov.values())):.0%} of traffic")
print()
for k, ex in examples.items():
    print(f"  {k} examples:")
    for inp, want, got in ex[:3]:
        print(f"    {inp:48s} want {want:20s} got {got!r}")

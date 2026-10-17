"""Read the sequential runs (results/seq): snippets 4000 -> mutants 4000 -> snippets 2000.

For every learner (fast-weight runs from probes.jsonl, LoRA from lora_log.json), each skill's
accuracy and AUC averaged over the probes in the LAST 1000 draws of each phase - a window, not
a single probe:
    snip @4000   acquired          mut @8000    acquired in phase 2
    snip @8000   RETAINED while mutants were learned
    snip @10000  after a 2000-draw refresh (relearning)
plus base-model health at the end.

    .venv/Scripts/python.exe seq_analyse.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from predict_failure import auc

ENDS = {"p1": 4000, "p2": 8000, "p3": 10000}


def fw_series(run: Path):
    out = {}
    for line in open(run / "probes.jsonl"):
        r = json.loads(line)
        if r.get("tag") not in ("stream", "final") or "metrics" not in r:
            continue
        for rule, pp in r["per_puzzle"].items():
            ok, surv = np.array(pp["direct"]), np.array(pp["exception"]).astype(bool)
            said_yes = np.where(surv, ~ok.astype(bool), ok.astype(bool))
            p = np.array([c["probs"][0] if c["probs"] else 0.5 for c in pp["conf"]])
            out.setdefault(rule, []).append((r["t"], float(ok.mean()), auc(np.where(said_yes, p, 1 - p), ~surv)))
    base = next(json.loads(l) for l in open(run / "probes.jsonl") if json.loads(l).get("tag") == "baseline")
    last = [json.loads(l) for l in open(run / "probes.jsonl") if json.loads(l).get("tag") in ("stream", "final")][-1]
    return out, (base["metrics"]["lm_ppl"], last["metrics"]["lm_ppl"]), (base["metrics"]["cloze_acc"], last["metrics"]["cloze_acc"])


def lora_series(run: Path):
    log = json.load(open(run / "lora_log.json"))["log"]
    out = {}
    for e in log[1:]:
        for rule, v in e["per_rule"].items():
            out.setdefault(rule, []).append((e["n"], v["acc_restricted"], v["auc"]))
    return out, (log[0].get("lm_ppl"), log[-1].get("lm_ppl")), (log[0].get("cloze_acc"), log[-1].get("cloze_acc"))


def window(series, end, width=1000):
    pts = [(a, u) for t, a, u in series if end - width < t <= end]
    return (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))) if pts else (float("nan"),) * 2


if __name__ == "__main__":
    root = Path("results/seq")
    print(f"{'learner':<22}{'snip@4000':>14}{'snip@8000':>14}{'mut@8000':>14}{'snip@10000':>14}   ppl / cloze")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            if (d / "lora_log.json").exists():
                s, ppl, cl = lora_series(d)
            elif (d / "probes.jsonl").exists():
                s, ppl, cl = fw_series(d)
            else:
                continue
        except (StopIteration, KeyError, IndexError, json.JSONDecodeError):
            print(f"{d.name:<22}still running")
            continue
        cells = [window(s.get("snip", []), ENDS["p1"]), window(s.get("snip", []), ENDS["p2"]),
                 window(s.get("mut", []), ENDS["p2"]), window(s.get("snip", []), ENDS["p3"])]
        f = lambda x: "-" if x is None else f"{x:.2f}"
        print(f"{d.name:<22}" + "".join(f"{a:>8.3f}/{u:.2f}" for a, u in cells)
              + f"   {f(ppl[0])}->{f(ppl[1])} / {f(cl[0])}->{f(cl[1])}")
    print("\nacc/AUC per cell, averaged over the probes in the last 1000 draws of that phase.")

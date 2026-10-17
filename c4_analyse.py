"""Read Connect-Four runs: held-out accuracy against the bars a model has to clear.

For every run in results/c4/<name>/:
  - held-out accuracy per probe; the MEAN over the second half of probes is the headline
    (single probes are noisy draws - a rule this project learned the hard way)
  - accuracy per TRUE class, from the probe set rebuilt exactly as the experiment drew it:
    a model that answers one constant label posts a respectable number (the prior is 0.43)
    and only the per-class split shows it
  - tactical / quiet split, memorisation (recent stream positions), base-model health
  - the confidence gate at the final probe: failure AUC and accuracy at fixed coverage
Bars come from results/c4_preflight.json (same task, same held-out definition), matched at
N = tool calls so far.

    .venv/Scripts/python.exe c4_analyse.py [run ...]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from hdelta.tasks.connect4 import LABELS, ConnectFourValueTask
from predict_failure import auc

BARS = None


def bars_at(n, size="5x4"):
    global BARS
    if BARS is None:
        pre = json.load(open("results/c4_preflight.json"))
        BARS = {r["size"]: {int(k): v for k, v in r["heldout_acc_vs_N"].items()} for r in pre}
    curve = BARS.get(size, {})
    if not curve:
        return {}
    k = max([m for m in curve if m <= max(n, min(curve))], default=min(curve))
    return curve[k] | {"at_N": k}


def probe_truth(cfg):
    """Rebuild the held-out probe set exactly as Experiment.__init__ draws it."""
    t = cfg["task"]
    task = ConnectFourValueTask(t["rules"], width=t.get("width", 5), height=t.get("height", 4),
                                plies=tuple(t.get("plies", (4, 14))),
                                heldout_mod=t.get("heldout_mod", 4))
    seed = int(cfg.get("seed", 0))
    probe_n = int(cfg["stream"].get("probe_n", 48))
    out = {}
    for i, r in enumerate(task.rule_names()):
        rng = random.Random(10_000 + 17 * i + seed)
        task.split = "heldout"
        out[r] = [task.sample(r, rng) for _ in range(probe_n)]
        task.split = "train"
    return task, out


def analyse(run):
    run = Path(run)
    cfg = json.load(open(run / "config.json"))
    size = f"{cfg['task'].get('width', 5)}x{cfg['task'].get('height', 4)}"
    recs = [json.loads(l) for l in open(run / "probes.jsonl")]
    probes = [r for r in recs if r.get("tag") in ("stream", "final") and "metrics" in r]
    if not probes:
        return None
    task, psets = probe_truth(cfg)
    rule = task.rule_names()[0]
    truth = [p.answer for p in psets[rule]]
    tac = [task.is_exception(rule, p.input_str) for p in psets[rule]]
    rows = []
    for r in probes:
        m, pp = r["metrics"], r["per_puzzle"][rule]
        ok = np.array(pp["direct"])
        assert len(ok) == len(truth), "probe set rebuilt with a different size"
        per_class = {l: float(ok[[i for i, t in enumerate(truth) if t == l]].mean()) for l in LABELS}
        rows.append({"t": r["t"], "acc": float(ok.mean()), "per_class": per_class,
                     "tactical": float(ok[np.array(tac)].mean()),
                     "quiet": float(ok[~np.array(tac)].mean()),
                     "seen": m.get("seen/zero_direct_acc"), "ppl": m.get("lm_ppl"),
                     "cloze": m.get("cloze_acc"), "conf": [c["probs"][0] if c["probs"] else 0.0
                                                           for c in pp.get("conf", [])],
                     "ok": ok})
    half = rows[len(rows) // 2:]
    last = rows[-1]
    conf = np.array(last["conf"])
    fail = 1 - last["ok"]
    gate = {}
    if len(conf):
        order = np.argsort(-conf)                         # most confident first
        for cov in (0.25, 0.5, 0.75):
            k = int(round(cov * len(conf)))
            gate[cov] = float(last["ok"][order[:k]].mean())
    base_ppl = next((r["metrics"].get("lm_ppl") for r in recs if r.get("tag") == "baseline"), None)
    return {"run": run.name, "size": size, "n_probes": len(rows), "t_final": last["t"],
            "mean_2nd_half": float(np.mean([r["acc"] for r in half])),
            "sd_2nd_half": float(np.std([r["acc"] for r in half])),
            "max": max(r["acc"] for r in rows), "final": last["acc"],
            "per_class_2nd_half": {l: float(np.mean([r["per_class"][l] for r in half])) for l in LABELS},
            "tactical_2nd_half": float(np.mean([r["tactical"] for r in half])),
            "quiet_2nd_half": float(np.mean([r["quiet"] for r in half])),
            "seen_2nd_half": float(np.mean([r["seen"] for r in half if r["seen"] is not None] or [np.nan])),
            "ppl_base": base_ppl, "ppl_final": last["ppl"], "cloze_final": last["cloze"],
            "final_conf_auc": auc(1 - conf, fail) if len(conf) else float("nan"),
            "final_acc_at_coverage": gate,
            "bars_at_final_N": bars_at(last["t"], size),
            "curve": [(r["t"], round(r["acc"], 3)) for r in rows]}


def show(a):
    b = a["bars_at_final_N"]
    pc = a["per_class_2nd_half"]
    print(f"\n=== {a['run']} ({a['size']}, t={a['t_final']}, {a['n_probes']} probes) ===")
    print(f"  held-out: mean 2nd half {a['mean_2nd_half']:.3f} (sd {a['sd_2nd_half']:.3f})  "
          f"max {a['max']:.3f}  final {a['final']:.3f}")
    print(f"  per true class: " + "  ".join(f"{l} {v:.2f}" for l, v in pc.items())
          + f"   tactical {a['tactical_2nd_half']:.2f} / quiet {a['quiet_2nd_half']:.2f}"
          + f"   recent stream (memorisation) {a['seen_2nd_half']:.2f}")
    if b:
        print(f"  bars at N={b['at_N']}: prior {b['prior']:.3f}  cache* {b['cache*']:.3f}  "
              f"logreg {b['logreg']:.3f}  mlp {b['mlp']:.3f}")
    print(f"  base model: ppl {a['ppl_base']} -> {a['ppl_final']}, cloze {a['cloze_final']}")
    print(f"  gate at final probe: failure AUC {a['final_conf_auc']:.3f}; accuracy of the most "
          f"confident " + ", ".join(f"{int(c * 100)}% = {v:.3f}" for c, v in a["final_acc_at_coverage"].items()))
    print("  curve: " + " ".join(f"{t}:{acc:.2f}" for t, acc in a["curve"][::2]))


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(str(p) for p in Path("results/c4").iterdir()
                                  if p.is_dir() and (p / "probes.jsonl").exists())
    out = []
    for r in runs:
        a = analyse(r)
        if a:
            show(a)
            out.append({k: v for k, v in a.items()})
    json.dump(out, open("results/c4_analysis.json", "w"), indent=1, default=float)

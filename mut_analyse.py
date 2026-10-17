"""Read the mutant runs: fast-weight runs (results/mut/F*/probes.jsonl) and LoRA controls
(results/mut/*/lora_log.json), against the model-free bars in results/mut_preflight.json.

Probes are balanced (caught/survived 50/50), so accuracy is balanced accuracy and 0.5 is the
prior. Headline = mean over the second half of probes. The generated label is recovered from
correctness and the true class (survivor = the task's 'exception' flag), so per-class accuracy
and an AUC can be computed: score = P(yes) = p if it said yes, else 1 - p.

    .venv/Scripts/python.exe mut_analyse.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from predict_failure import auc


def fast_weight_run(run: Path):
    recs = [json.loads(l) for l in open(run / "probes.jsonl")]
    probes = [r for r in recs if r.get("tag") in ("stream", "final") and "metrics" in r]
    base = next((r for r in recs if r.get("tag") == "baseline"), None)
    rows = []
    for r in probes:
        pp = next(iter(r["per_puzzle"].values()))      # single-rule tasks: mutants, snippets
        ok, surv = np.array(pp["direct"]), np.array(pp["exception"]).astype(bool)
        said_yes = np.where(surv, ~ok.astype(bool), ok.astype(bool))
        p = np.array([c["probs"][0] if c["probs"] else 0.5 for c in pp["conf"]])
        p_yes = np.where(said_yes, p, 1 - p)
        rows.append({"t": r["t"], "acc": ok.mean(), "caught": ok[~surv].mean(), "survived": ok[surv].mean(),
                     "auc": auc(p_yes, ~surv), "ppl": r["metrics"].get("lm_ppl"),
                     "cloze": r["metrics"].get("cloze_acc")})
    if len(rows) < 2:
        return None                       # still running: nothing to average yet
    half = rows[len(rows) // 2:]
    m = lambda k: float(np.mean([x[k] for x in half]))
    return {"run": run.name, "t": rows[-1]["t"], "acc": m("acc"), "caught": m("caught"),
            "survived": m("survived"), "auc": m("auc"), "sd_acc": float(np.std([x["acc"] for x in half])),
            "ppl": (base["metrics"].get("lm_ppl") if base else None, rows[-1]["ppl"]),
            "cloze": (base["metrics"].get("cloze_acc") if base else None, rows[-1]["cloze"]),
            "curve": [(x["t"], round(x["acc"], 3)) for x in rows]}


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/mut")
    pre = (json.load(open("results/mut_preflight.json"))
           if root.name == "mut" and Path("results/mut_preflight.json").exists() else None)
    if pre:
        print("model-free bars (balanced held-out set of", pre["held"], "):")
        for n, row in pre["curve"].items():
            print(f"  N={n:>5} ({row['distinct']} distinct)  op-rules {row['op_rules']:.3f}  "
                  f"logreg {row['logreg_acc']:.3f}/AUC {row['logreg_auc']:.3f}  mlp {row['mlp_acc']:.3f}/AUC {row['mlp_auc']:.3f}")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "lora_log.json").exists():
            log = json.load(open(d / "lora_log.json"))["log"]
            half = log[1:][len(log[1:]) // 2:]
            print(f"\n=== {d.name} (LoRA control) ===  frozen n=0: acc {log[0]['acc_restricted']:.3f} "
                  f"AUC {log[0]['auc']:.3f}")
            print(f"  2nd half mean: acc {np.mean([x['acc'] for x in half]):.3f} "
                  f"(restricted {np.mean([x['acc_restricted'] for x in half]):.3f})  AUC {np.mean([x['auc'] for x in half]):.3f}")
            print("  curve: " + " ".join(f"{x['n']}:{x['acc_restricted']:.2f}/{x['auc']:.2f}" for x in log))
        elif (d / "probes.jsonl").exists():
            a = fast_weight_run(d)
            if a is None:
                print(f"\n=== {d.name}: still running ===")
                continue
            print(f"\n=== {a['run']} (fast weights, t={a['t']}) ===")
            print(f"  2nd half mean: acc {a['acc']:.3f} (sd {a['sd_acc']:.3f})  caught {a['caught']:.2f} / "
                  f"survived {a['survived']:.2f}  AUC {a['auc']:.3f}")
            print(f"  base model: ppl {a['ppl'][0]} -> {a['ppl'][1]}, cloze {a['cloze'][0]} -> {a['cloze'][1]}")
            print("  curve: " + " ".join(f"{t}:{v:.2f}" for t, v in a["curve"]))

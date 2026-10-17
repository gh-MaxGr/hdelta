"""Model-free bars scored on the EXACT probe set the h-delta and LoRA mutant runs use
(Random(10_000 + seed), held-out split, probe_n draws), so every number is on identical items.

    .venv/Scripts/python.exe mut_bars_on_probes.py
"""
import json
import re
import sys
import zlib
import random
from collections import Counter, defaultdict

import numpy as np
import yaml

from hdelta.tasks import make_task
from mut_preflight import auc, feats, fit

CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/mut_base.yaml"
cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
task = make_task(cfg["task"])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else int(cfg.get("seed", 0))
rng = random.Random(10_000 + SEED)
task.split = "heldout"
probes = [task.sample("mut", rng).input_str for _ in range(int(cfg["stream"]["probe_n"]))]
yp = np.array([int(task.rec[k]["killed"]) for k in probes])
task.split = "train"
srng = random.Random(0)
draws = [task.sample_input("mut", srng) for _ in range(4000)]
yd = np.array([int(task.rec[k]["killed"]) for k in draws])
def feats_ctx(task, keys):
    """The diff features, plus the function context as its own token family when shown."""
    X = feats(task, keys)                            # already log1p-scaled
    if getattr(task, "context", False):
        C = np.zeros_like(X)
        for i, k in enumerate(keys):
            r = task.rec[k]
            for w in re.findall(r"[A-Za-z_]+|\d+", task.ctx[r["module"]][str(r["site"])]):
                C[i, zlib.crc32(("k" + w).encode()) % X.shape[1]] += 1.0
        X = X + np.log1p(C)                          # same scaling for the context family
    return X


Xd, Xp = feats_ctx(task, draws), feats_ctx(task, probes)
out = {}
for n in (1000, 2000, 4000):
    cnt = defaultdict(Counter)
    for k in draws[:n]:
        r = task.rec[k]
        cnt[(r["module"], r["op"].split()[0])][r["killed"]] += 1
    op = np.array([int(cnt[(task.rec[k]["module"], task.rec[k]["op"].split()[0])].most_common(1)[0][0])
                   if cnt[(task.rec[k]["module"], task.rec[k]["op"].split()[0])] else 1 for k in probes])
    lr, mlp = fit(Xd[:n], yd[:n])(Xp), fit(Xd[:n], yd[:n], hidden=256)(Xp)
    out[n] = {"op_rules": float(np.mean(op == yp)), "logreg_acc": float(np.mean((lr > .5) == yp)),
              "logreg_auc": auc(lr, yp), "mlp_acc": float(np.mean((mlp > .5) == yp)), "mlp_auc": auc(mlp, yp)}
    print(n, {k: round(v, 3) for k, v in out[n].items()}, flush=True)
print("probe set:", len(probes), "items,", len(set(probes)), "distinct, caught share", yp.mean())
json.dump(out, open(f"results/mut_bars_on_probes{'_ctx' if cfg['task'].get('context') else ''}_s{SEED}.json", "w"), indent=1)

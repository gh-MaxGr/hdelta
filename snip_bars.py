"""Model-free bars for the snippet task, on the EXACT probe set the LLM runs use
(Random(10_000 + seed), held-out split, probe_n draws, balanced).

Features: hashed unigrams of the whole snippet, plus the LAST line's tokens as their own family
(by construction the last line decides whether it raises - a fair baseline should get that),
plus a few type hints of the literals bound earlier. logreg and a 256-unit MLP, fitted on
N balanced training draws.

    .venv/Scripts/python.exe snip_bars.py [config] [seed]
"""
import json
import random
import re
import sys
import zlib

import numpy as np
import yaml

from hdelta.tasks import make_task
from mut_preflight import auc, fit

DIM = 4096
TOK = re.compile(r"[A-Za-z_]+|\d+\.\d+|\d+|'[^']*'|==|//|\+=|-=|\*=|[^\s\w]")


def feats(task, keys):
    X = np.zeros((len(keys), DIM), dtype=np.float32)
    for i, k in enumerate(keys):
        lines = task.rec[k]["code"].splitlines()
        toks = [f"u:{t}" for l in lines for t in TOK.findall(l)]
        toks += [f"L:{t}" for t in TOK.findall(lines[-1])]
        for l in lines[:-1]:                           # crude type of what each name was bound to
            m = re.match(r"(\w+) = (.*)", l)
            if m:
                v = m.group(2).strip()
                kind = ("str" if v.startswith("'") else "list" if v.startswith("[") else
                        "dict" if v.startswith("{") else "none" if v == "None" else
                        "bool" if v in ("True", "False") else "num" if re.fullmatch(r"-?[\d.]+", v) else "expr")
                toks.append(f"T:{m.group(1)}={kind}")
        for t in toks:
            X[i, zlib.crc32(t.encode()) % DIM] += 1.0
    return np.log1p(X)


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/snip_base.yaml", encoding="utf-8"))
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else int(cfg.get("seed", 0))
    task = make_task(cfg["task"])
    print("pools:", task.stats())
    rng = random.Random(10_000 + seed)
    task.split = "heldout"
    probes = [task.sample("snip", rng).input_str for _ in range(int(cfg["stream"]["probe_n"]))]
    yp = np.array([int(task.rec[k]["raises"]) for k in probes])
    task.split = "train"
    srng = random.Random(0)
    draws = [task.sample_input("snip", srng) for _ in range(8000)]
    yd = np.array([int(task.rec[k]["raises"]) for k in draws])
    Xd, Xp = feats(task, draws), feats(task, probes)
    out = {}
    for n in (1000, 2000, 4000, 8000):
        lr, mlp = fit(Xd[:n], yd[:n])(Xp), fit(Xd[:n], yd[:n], hidden=256)(Xp)
        out[n] = {"logreg_acc": float(np.mean((lr > .5) == yp)), "logreg_auc": auc(lr, yp),
                  "mlp_acc": float(np.mean((mlp > .5) == yp)), "mlp_auc": auc(mlp, yp)}
        print(n, {k: round(v, 3) for k, v in out[n].items()}, flush=True)
    print("probes:", len(probes), "items,", len(set(probes)), "distinct, raise share", yp.mean())
    json.dump(out, open(f"results/snip_bars_s{seed}.json", "w"), indent=1)

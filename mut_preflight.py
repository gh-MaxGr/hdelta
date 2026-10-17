"""Model-free bars for the mutant task (balanced held-out set; held-out = unseen top-level
functions/classes, so any cache keyed on the function has nothing to recall).

    prior      0.5 by construction (balanced set)
    op-rules   majority label per (module, mutation type) from the training draws - the best
               cache of CHECKABLE conditions available here
    logreg     on hashed word/char features of the diff + module + op
    mlp        one hidden layer of 256 on the same features

N = training draws at the task's balance (with replacement: the training pool is small, so
beyond a few thousand draws the learner sees repeats - reported beside each N).

    .venv/Scripts/python.exe mut_preflight.py
"""
from __future__ import annotations

import json
import random
import re
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch

import yaml

from hdelta.tasks import make_task

NS = (250, 500, 1000, 2000, 4000)
DIM = 4096


def feats(task, keys):
    X = np.zeros((len(keys), DIM), dtype=np.float32)
    for i, k in enumerate(keys):
        r = task.rec[k]
        toks = [f"mod={r['module']}", f"op={r['op']}", f"opk={r['op'].split()[0]}",
                f"priv={r['func'].split('.')[-1].startswith('_')}"]
        for line in r["diff"].splitlines():
            if line[:1] in "+-" and not line.startswith(("+++", "---")):
                toks += [f"{line[0]}{w}" for w in re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", line[1:])]
            else:
                toks += [f"c{w}" for w in re.findall(r"[A-Za-z_]+", line)]
        for t in toks:
            X[i, zlib.crc32(t.encode()) % DIM] += 1.0    # stable: str hash() is salted per process
    return np.log1p(X)


def fit(X, y, hidden=0, epochs=60, seed=0):
    torch.manual_seed(seed)
    layers = [torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 2)] if hidden \
        else [torch.nn.Linear(X.shape[1], 2)]
    net = torch.nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Xt[b]), yt[b]).backward()
            opt.step()
    return lambda Z: torch.softmax(net(torch.tensor(Z)), -1)[:, 1].detach().numpy()


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum())
                 / (len(pos) * len(neg)))


if __name__ == "__main__":
    task = make_task(yaml.safe_load(open("configs/mut_base.yaml", encoding="utf-8"))["task"])
    print("pools:", task.stats())
    rng = random.Random(0)
    held_no = task.pool[("heldout", "no")]
    held_yes = rng.sample(task.pool[("heldout", "yes")], min(len(held_no), len(task.pool[("heldout", "yes")])))
    held = held_no + held_yes
    yh = np.array([0] * len(held_no) + [1] * len(held_yes))          # 1 = caught
    Xh = feats(task, held)
    task.split = "train"
    draws = [task.sample_input("mut", rng) for _ in range(max(NS))]
    yd = np.array([int(task.rec[k]["killed"]) for k in draws])
    Xd = feats(task, draws)
    out = {"pools": task.stats(), "held": len(held), "curve": {}}
    for n in NS:
        cnt = defaultdict(Counter)
        for k in draws[:n]:
            r = task.rec[k]
            cnt[(r["module"], r["op"].split()[0])][r["killed"]] += 1
        op_pred = [cnt[(task.rec[k]["module"], task.rec[k]["op"].split()[0])].most_common(1)[0][0]
                   if cnt[(task.rec[k]["module"], task.rec[k]["op"].split()[0])] else True for k in held]
        op_acc = float(np.mean([int(p) == t for p, t in zip(op_pred, yh)]))
        lr = fit(Xd[:n], yd[:n])(Xh)
        mlp = fit(Xd[:n], yd[:n], hidden=256)(Xh)
        row = {"distinct": len(set(draws[:n])), "op_rules": round(op_acc, 3),
               "logreg_acc": round(float(np.mean((lr > 0.5) == yh)), 3), "logreg_auc": round(auc(lr, yh), 3),
               "mlp_acc": round(float(np.mean((mlp > 0.5) == yh)), 3), "mlp_auc": round(auc(mlp, yh), 3)}
        out["curve"][n] = row
        print(n, row, flush=True)
    json.dump(out, open("results/mut_preflight.json", "w"), indent=1)

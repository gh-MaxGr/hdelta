"""Pre-flight for the Connect-Four value task, no GPU.

1. CORRECTNESS  memoised solver vs a plain minimax with no memo, on random late positions;
                a position and its mirror image must have the same value; string round-trip.
2. SIZE         per board size: solver time, class balance, share of TACTICAL positions (the
                only ones with a checkable reason), how often a stream position repeats.
3. BASELINES    held-out accuracy (positions never seen, nor their mirrors) as a function of
                N tool calls, for what the LLM slice will have to beat:
                  prior       majority class of the training positions
                  cache*      the STRONGEST possible reason cache: tactical positions answered
                              by perfect tactical rules, everything else by the prior. An exact
                              memo adds nothing on held-out positions by construction.
                  logreg      multinomial logistic regression on one-hot cells
                  mlp         one hidden layer of 256, trained to convergence on the same N

    .venv/Scripts/python.exe c4_preflight.py
"""
from __future__ import annotations

import json
import random
import time

import numpy as np
import torch

from hdelta.tasks.connect4 import LABELS, Board, ConnectFourValueTask

SIZES = [(5, 4, (4, 14)), (6, 4, (4, 16)), (5, 5, (4, 18))]
NS = (250, 500, 1000, 2000, 4000, 8000, 16000)
N_HELD = 3000


def naive_value(b, cur, mask):
    if b.n_stones(mask) == b.cells:
        return 0
    if b.winning_moves(cur, mask):
        return 1
    return max(-naive_value(b, *b.play(cur, mask, c)) for c in range(b.W) if b.can_play(mask, c))


def correctness(task, rng, n=300):
    b = task.board
    bad_rt = bad_mir = bad_naive = 0
    for _ in range(n):
        s = task.random_position(rng)
        cur, mask = b.from_str(s)
        bad_rt += b.to_str(cur, mask) != s
        v = task.solver.value(cur, mask)
        bad_mir += v != task.solver.value(*b.from_str(Board.mirror(s)))
        if b.cells - b.n_stones(mask) <= 11:          # naive minimax is only feasible late
            bad_naive += v != naive_value(b, cur, mask)
    return bad_rt, bad_mir, bad_naive


def features(b, strs):
    X = np.zeros((len(strs), 3 * b.cells), dtype=np.float32)
    for i, s in enumerate(strs):
        for j, ch in enumerate(s.replace("/", "")):
            X[i, 3 * j + ".XO".index(ch)] = 1
    return X


def tactical(b, s):
    cur, mask = b.from_str(s)
    if b.winning_moves(cur, mask):
        return "win"
    if len(b.winning_moves(cur ^ mask, mask)) >= 2:
        return "loss"
    return None


def fit_logreg(X, y, steps=400, lr=0.5, l2=1e-3):
    Xt, yt = torch.tensor(X), torch.tensor(y)
    lin = torch.nn.Linear(X.shape[1], 3)
    opt = torch.optim.LBFGS(lin.parameters(), lr=lr, max_iter=steps)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xt), yt) + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return lambda Z: lin(torch.tensor(Z)).argmax(-1).numpy()


def fit_mlp(X, y, seed=0, epochs=300):
    torch.manual_seed(seed)
    idx = np.random.RandomState(seed).permutation(len(y))
    v = max(1, len(y) // 10)
    va, tr = idx[:v], idx[v:]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    best, best_state = -1.0, None
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 128):
            bi = torch.tensor(tr)[perm[i:i + 128]]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Xt[bi]), yt[bi]).backward()
            opt.step()
        if ep % 10 == 0:
            with torch.no_grad():
                acc = (net(Xt[va]).argmax(-1) == yt[va]).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return lambda Z: net(torch.tensor(Z)).argmax(-1).detach().numpy()


def study(W, H, plies):
    task = ConnectFourValueTask({"v": {}}, width=W, height=H, plies=plies)
    b = task.board
    rng = random.Random(0)
    out = {"size": f"{W}x{H}", "plies": plies}
    rt, mir, nv = correctness(task, rng)
    out["correctness_bad"] = {"roundtrip": rt, "mirror": mir, "vs_naive": nv}
    t0 = time.time()
    task.split = "train"
    train = [task.sample_input("v", rng) for _ in range(max(NS))]
    task.split = "heldout"
    held = [task.sample_input("v", rng) for _ in range(N_HELD)]
    ytr = [task.solve("v", s)[1] for s in train]
    yhe = [task.solve("v", s)[1] for s in held]
    out["solve_seconds"] = round(time.time() - t0, 1)
    out["solver_nodes"] = task.solver.nodes
    out["balance_train"] = {l: round(ytr.count(l) / len(ytr), 3) for l in LABELS}
    out["tactical_share_held"] = round(np.mean([tactical(b, s) is not None for s in held]), 3)
    out["distinct_in_16k_stream"] = len(set(train))
    out["held_seen_in_train"] = sum(s in set(train) or Board.mirror(s) in set(train) for s in held)
    Xtr, Xhe = features(b, train), features(b, held)
    yi = np.array([LABELS.index(l) for l in ytr])
    yh = np.array([LABELS.index(l) for l in yhe])
    tac = [tactical(b, s) for s in held]
    curves = {}
    for n in NS:
        prior = max(LABELS, key=lambda l: ytr[:n].count(l))
        nontac_prior = max(LABELS, key=lambda l: sum(1 for s, t in zip(train[:n], ytr[:n])
                                                      if t == l and tactical(b, s) is None))
        cache = np.mean([(t or nontac_prior) == l for t, l in zip(tac, yhe)])
        lr = np.mean(fit_logreg(Xtr[:n], yi[:n])(Xhe) == yh)
        mlp = np.mean(fit_mlp(Xtr[:n], yi[:n])(Xhe) == yh)
        curves[n] = {"prior": round(float(np.mean([prior == l for l in yhe])), 3),
                     "cache*": round(float(cache), 3), "logreg": round(float(lr), 3),
                     "mlp": round(float(mlp), 3)}
        print(f"  {W}x{H} N={n:<6} " + "  ".join(f"{k} {v:.3f}" for k, v in curves[n].items()), flush=True)
    out["heldout_acc_vs_N"] = curves
    out["example"] = {"input": train[0], "steps_answer": task.solve("v", train[0])}
    return out


if __name__ == "__main__":
    res = []
    for W, H, plies in SIZES:
        print(f"=== {W}x{H}, plies {plies} ===", flush=True)
        r = study(W, H, plies)
        print(json.dumps({k: v for k, v in r.items() if k != "heldout_acc_vs_N"}, indent=1), flush=True)
        res.append(r)
        json.dump(res, open("results/c4_preflight.json", "w"), indent=1)

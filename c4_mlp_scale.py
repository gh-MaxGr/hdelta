"""Non-LLM learners on 5x4 Connect-Four value at larger N - the bars for long runs.
Same task, same held-out definition and the same draw as c4_preflight.py (Random(0)), so its
curve continues the preflight's. Adds a bigger MLP (2 x 512) as the strongest cheap learner.

    .venv/Scripts/python.exe c4_mlp_scale.py
"""
import json
import random

import numpy as np
import torch

from c4_preflight import N_HELD, features, fit_logreg, fit_mlp
from hdelta.tasks.connect4 import LABELS, ConnectFourValueTask

NS = (16000, 32000, 64000)


def fit_mlp2(X, y, seed=0, epochs=200):
    torch.manual_seed(seed)
    idx = np.random.RandomState(seed).permutation(len(y))
    v = max(1, len(y) // 10)
    va, tr = idx[:v], idx[v:]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.ReLU(),
                              torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    best, best_state = -1.0, None
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            bi = torch.tensor(tr)[perm[i:i + 256]]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(net(Xt[bi]), yt[bi]).backward()
            opt.step()
        if ep % 5 == 0:
            with torch.no_grad():
                acc = (net(Xt[va]).argmax(-1) == yt[va]).float().mean().item()
            if acc > best:
                best, best_state = acc, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return lambda Z: net(torch.tensor(Z)).argmax(-1).detach().numpy()


if __name__ == "__main__":
    torch.set_num_threads(8)          # leave cores for the GPU run's data side
    task = ConnectFourValueTask({"v": {}}, width=5, height=4, plies=(4, 14))
    rng = random.Random(0)
    for _ in range(300):              # c4_preflight's correctness() consumed these first
        task.random_position(rng)
    task.split = "train"
    train = [task.sample_input("v", rng) for _ in range(max(NS))]
    task.split = "heldout"
    held = [task.sample_input("v", rng) for _ in range(N_HELD)]
    y = np.array([LABELS.index(task.solve("v", s)[1]) for s in train])
    yh = np.array([LABELS.index(task.solve("v", s)[1]) for s in held])
    X, Xh = features(task.board, train), features(task.board, held)
    out = {}
    for n in NS:
        out[n] = {"logreg": float(np.mean(fit_logreg(X[:n], y[:n])(Xh) == yh)),
                  "mlp": float(np.mean(fit_mlp(X[:n], y[:n])(Xh) == yh)),
                  "mlp2x512": float(np.mean(fit_mlp2(X[:n], y[:n])(Xh) == yh))}
        print(n, out[n], flush=True)
        json.dump(out, open("results/c4_mlp_scale.json", "w"), indent=1)

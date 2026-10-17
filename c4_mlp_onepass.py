"""Calibration: the MLP baseline, but trained like the stream - each example seen ONCE, in
batches of 4, Adam - so its N is comparable to the fast-weight and one-pass LoRA runs.

    .venv/Scripts/python.exe c4_mlp_onepass.py
"""
import json
import random

import numpy as np
import torch

from c4_preflight import N_HELD, features
from hdelta.tasks.connect4 import LABELS, ConnectFourValueTask

task = ConnectFourValueTask({"v": {}}, width=5, height=4, plies=(4, 14))
rng = random.Random(0)
for _ in range(300):
    task.random_position(rng)
task.split = "train"
train = [task.sample_input("v", rng) for _ in range(16000)]
task.split = "heldout"
held = [task.sample_input("v", rng) for _ in range(N_HELD)]
X = torch.tensor(features(task.board, train))
Xh = torch.tensor(features(task.board, held))
y = torch.tensor([LABELS.index(task.solve("v", s)[1]) for s in train])
yh = torch.tensor([LABELS.index(task.solve("v", s)[1]) for s in held])
out = {}
for lr in (1e-3, 3e-3):
    for passes in (1, 4):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 3))
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        curve = {}
        for p in range(passes):
            for i in range(0, len(X), 4):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(X[i:i + 4]), y[i:i + 4]).backward()
                opt.step()
                n = p * len(X) + i + 4
                if n % 4000 == 0:
                    with torch.no_grad():
                        curve[n] = round((net(Xh).argmax(-1) == yh).float().mean().item(), 3)
        out[f"lr{lr}_passes{passes}"] = curve
        print(lr, passes, curve, flush=True)
json.dump(out, open("results/c4_mlp_onepass.json", "w"), indent=1)

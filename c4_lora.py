"""Control for the Connect-Four runs: ordinary gradient training (LoRA) on the same slice.

If the fast-weight runs stay at the prior, that alone cannot say WHY: it could be the
fast-weight rule, or it could be that a 3B model reading this board as text cannot represent
the value at all. This separates the two. Everything is matched to the fast-weight runs:

    locus      q/k/v/o projections of the last 9 layers
    data       tool-labelled positions from the TRAIN split, each seen ONCE (online, like the
               stream), in batches of 4 - so N examples = N tool calls
    prompt     identical text (task.make_pair(..., "direct"))
    scoring    greedy first token over the FULL vocabulary == the label token, on (a) the
               experiment's own 256 held-out probes for seed 0 and (b) 1000 more held-out
               positions; plus accuracy restricted to the three labels

    .venv/Scripts/python.exe c4_lora.py --n 16000
    .venv/Scripts/python.exe c4_lora.py --device cpu --n 8 --eval-every 4 --held 8 --probe-n 8   # smoke
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from hdelta.tasks.base import Puzzle
from hdelta.tasks.connect4 import LABELS, ConnectFourValueTask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--held", type=int, default=1000)
    ap.add_argument("--probe-n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=1,
                    help="passes over the same N positions (replay); 1 = online, like the stream")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/c4/lora_s0")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    task = ConnectFourValueTask({"c4": {}}, width=5, height=4, plies=(4, 14))
    # (a) the experiment's own probe set for this seed, drawn exactly as Experiment does
    rng = random.Random(10_000 + a.seed)
    task.split = "heldout"
    probes = [task.sample("c4", rng) for _ in range(a.probe_n)]
    hrng = random.Random(424_242 + a.seed)
    held = [task.sample("c4", hrng) for _ in range(a.held)]
    task.split = "train"
    srng = random.Random(a.seed * 1000 + 3)            # the experiment's phase-0 stream seed
    stream = [task.sample("c4", srng) for _ in range(a.n)]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=dtype).to(a.device)
    L = model.config.num_hidden_layers
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     layers_to_transform=list(range(L - 9, L)))
    model = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    label_ids = [tok(" " + l, add_special_tokens=False)["input_ids"] for l in LABELS]
    assert all(len(x) == 1 for x in label_ids), label_ids
    label_ids = [x[0] for x in label_ids]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    def encode(prompts):
        seqs = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, T - len(s):] = torch.tensor(s)
            att[i, T - len(s):] = 1
        pos = (att.cumsum(-1) - 1).clamp_min(0)
        return ids.to(a.device), att.to(a.device), pos.to(a.device)

    @torch.no_grad()
    def evaluate(pz, chunk=32):
        model.eval()
        full = restricted = 0
        for i in range(0, len(pz), chunk):
            b = pz[i:i + chunk]
            ids, att, pos = encode([task.make_pair(p, "direct")[0] for p in b])
            lg = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :].float()
            gold = torch.tensor([label_ids[LABELS.index(p.answer)] for p in b], device=a.device)
            full += (lg.argmax(-1) == gold).sum().item()
            restricted += (torch.tensor(label_ids, device=a.device)[lg[:, label_ids].argmax(-1)] == gold).sum().item()
        model.train()
        return full / len(pz), restricted / len(pz)

    log = []
    t0 = time.time()
    p0, r0 = evaluate(probes)
    h0, _ = evaluate(held)
    log.append({"n": 0, "probe_acc": p0, "probe_acc_restricted": r0, "held_acc": h0})
    print(f"n=0 probe {p0:.3f} (restricted {r0:.3f}) held {h0:.3f}  trainable {n_train / 1e6:.1f}M", flush=True)
    model.train()
    for i in range(0, a.n * a.passes, a.batch):
        j = i % a.n
        b = stream[j:j + a.batch]
        ids, att, pos = encode([task.make_pair(p, "direct")[0] for p in b])
        lg = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :].float()
        gold = torch.tensor([label_ids[LABELS.index(p.answer)] for p in b], device=a.device)
        loss = torch.nn.functional.cross_entropy(lg, gold)
        opt.zero_grad()
        loss.backward()
        opt.step()
        n = i + len(b)
        if n % a.eval_every == 0 or n == a.n * a.passes:
            pa, ra = evaluate(probes)
            ha, _ = evaluate(held)
            log.append({"n": n, "probe_acc": pa, "probe_acc_restricted": ra, "held_acc": ha,
                        "loss": float(loss), "seconds": time.time() - t0})
            print(f"n={n} probe {pa:.3f} (restricted {ra:.3f}) held {ha:.3f} loss {float(loss):.3f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            json.dump({"args": vars(a), "trainable": n_train, "log": log},
                      open(out / "lora_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Ordinary gradient training (LoRA) on the fast-weight locus, for any task config.

Generalises c4_lora.py (kept as is for the Connect-Four runs already made with it). The task,
prompt and held-out probe sets come from the SAME config an h-delta run would use, drawn
exactly as Experiment.__init__ draws its probes (Random(10_000 + 17*i + seed) per rule i), so
the two are scored on identical items. n=0 is the FROZEN model zero-shot.

Two stream modes:
  default    one rule, --n draws from Random(seed*1000 + 3)   (the experiment's phase 0)
  --phases   follow cfg.stream.phases in order, phase i drawn from Random(seed*1000 + 7*i + 3)
             exactly as Experiment.run() does - a SEQUENTIAL stream (e.g. snippets -> mutants
             -> snippets), with every rule's probes scored at every checkpoint, so forgetting
             and acquisition come out of one run
Reports per rule: greedy-over-full-vocab accuracy, accuracy restricted to the labels, and the
AUC of P(first label). Base-model health (ppl, cloze) with the adapter ON at n=0 and the end,
measured exactly as the h-delta runs measure it.

    .venv/Scripts/python.exe lora_control.py --config configs/mut_base.yaml --n 8000
    .venv/Scripts/python.exe lora_control.py --config configs/seq_base.yaml --phases
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from hdelta.encoding import Encoder
from hdelta.metrics import cloze_accuracy, lm_perplexity
from hdelta.tasks import make_task
from predict_failure import auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--phases", action="store_true", help="follow cfg.stream.phases (sequential stream)")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--replay", type=int, default=0,
                    help="extra steps per fresh step, each on a batch drawn from EARLIER stream "
                         "items (no new tool calls); 0 = the original code path")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    seed = int(cfg.get("seed", 0) if a.seed is None else a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    task = make_task(cfg["task"])
    rules = task.rule_names()
    probe_n = int(cfg["stream"].get("probe_n", 48))
    probes = {}
    for i, r in enumerate(rules):                        # Experiment.__init__'s probe draw
        rng = random.Random(10_000 + 17 * i + seed)
        task.split = "heldout"
        probes[r] = [task.sample(r, rng) for _ in range(probe_n)]
        task.split = "train"
    if a.phases:                                         # Experiment.run()'s stream draw
        stream, bounds = [], []
        for i, ph in enumerate(cfg["stream"]["phases"]):
            srng = random.Random(seed * 1000 + 7 * i + 3)
            stream += [task.sample(ph["rule"], srng) for _ in range(int(ph["puzzles"]))]
            bounds.append((len(stream), ph["rule"]))
    else:
        srng = random.Random(seed * 1000 + 3)
        stream = [task.sample(rules[0], srng) for _ in range(a.n)]
        bounds = [(len(stream), rules[0])]
    total = len(stream) * a.passes
    labels = sorted({p.answer for ps in probes.values() for p in ps} | {p.answer for p in stream[:200]})

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    dtype = torch.bfloat16 if a.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], dtype=dtype).to(a.device)
    L = model.config.num_hidden_layers
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             layers_to_transform=list(range(L - 9, L))))
    lab_ids = [tok(" " + l, add_special_tokens=False)["input_ids"] for l in labels]
    assert all(len(x) == 1 for x in lab_ids), lab_ids
    lab_ids = [x[0] for x in lab_ids]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    def encode(prompts):
        seqs = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, T - len(s):] = torch.tensor(s)
            att[i, T - len(s):] = 1
        return ids.to(a.device), att.to(a.device), (att.cumsum(-1) - 1).clamp_min(0).to(a.device)

    @torch.no_grad()
    def evaluate(pz, chunk=16):
        model.eval()
        full, restr, p0, gold_all = [], [], [], []
        for i in range(0, len(pz), chunk):
            b = pz[i:i + chunk]
            ids, att, pos = encode([task.make_pair(p, "direct")[0] for p in b])
            lg = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :].float()
            gold = [lab_ids[labels.index(p.answer)] for p in b]
            full += (lg.argmax(-1).cpu().numpy() == np.array(gold)).tolist()
            sub = lg[:, lab_ids]
            restr += (np.array(lab_ids)[sub.argmax(-1).cpu().numpy()] == np.array(gold)).tolist()
            p0 += torch.softmax(sub, -1)[:, 0].cpu().tolist()
            gold_all += [int(p.answer == labels[0]) for p in b]
        model.train()
        return float(np.mean(full)), float(np.mean(restr)), auc(p0, gold_all)

    def evaluate_all():
        per = {}
        for r in rules:
            acc, racc, au = evaluate(probes[r])
            per[r] = {"acc": acc, "acc_restricted": racc, "auc": au}
        # the first rule's numbers stay at top level: every earlier log reads them there
        return {**per[rules[0]], "per_rule": per}

    # base-model health with the adapter ON, measured exactly as the h-delta runs measure it,
    # so "skill vs damage" compares like with like (an always-on adapter can harm too)
    ec = cfg["eval"]
    lm_text = open(ec["lm_reference"], encoding="utf-8").read()
    cloze = json.load(open(ec["cloze"], encoding="utf-8"))
    enc = Encoder(tok, a.device)

    @torch.no_grad()
    def health():
        model.eval()
        h = {"lm_ppl": lm_perplexity(model, enc, lm_text, max_windows=int(ec.get("lm_windows", 24))),
             "cloze_acc": cloze_accuracy(model, enc, cloze)}
        model.train()
        return h

    def phase_at(n):
        return next((r for end, r in bounds if n <= end * a.passes), bounds[-1][1])

    log, t0 = [], time.time()
    e = evaluate_all()
    log.append({"n": 0, **e, **health()})
    print(f"n=0 (frozen) acc {e['acc']:.3f} restricted {e['acc_restricted']:.3f} auc({labels[0]}) {e['auc']:.3f}  "
          f"labels {labels}  probes " + ", ".join(f"{r}:{len(probes[r])} ({len({p.input_str for p in probes[r]})} distinct)"
                                                 for r in rules), flush=True)
    model.train()
    for i in range(0, total, a.batch):
        j = i % len(stream)
        b = stream[j:j + a.batch]
        ids, att, pos = encode([task.make_pair(p, "direct")[0] for p in b])
        lg = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, -1, :].float()
        gold = torch.tensor([lab_ids[labels.index(p.answer)] for p in b], device=a.device)
        loss = torch.nn.functional.cross_entropy(lg, gold)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if a.replay and j >= a.batch:                   # revisit earlier items, labels already known
            rrng = random.Random(seed * 7919 + i)
            for _ in range(a.replay):
                rb = rrng.sample(stream[:j], a.batch)
                rids, ratt, rpos = encode([task.make_pair(p, "direct")[0] for p in rb])
                rlg = model(input_ids=rids, attention_mask=ratt, position_ids=rpos).logits[:, -1, :].float()
                rgold = torch.tensor([lab_ids[labels.index(p.answer)] for p in rb], device=a.device)
                opt.zero_grad()
                torch.nn.functional.cross_entropy(rlg, rgold).backward()
                opt.step()
        n = i + len(b)
        if n % a.eval_every == 0 or n == total:
            e = evaluate_all()
            log.append({"n": n, "phase_rule": phase_at(n), **e, "loss": float(loss),
                        "seconds": time.time() - t0, **(health() if n == total else {})})
            print(f"n={n} [{phase_at(n)}] " + "  ".join(
                f"{r}: {v['acc_restricted']:.3f}/{v['auc']:.3f}" for r, v in e["per_rule"].items())
                  + f"  loss {float(loss):.3f} [{time.time() - t0:.0f}s]", flush=True)
            json.dump({"args": vars(a), "labels": labels, "rules": rules, "log": log},
                      open(out / "lora_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()

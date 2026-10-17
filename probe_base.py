"""Frozen-model probe: can a candidate base model execute the slow path in context?

    python probe_base.py Qwen/Qwen2.5-1.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct ...

For each model (rule A, 48 held-out puzzles, no plasticity) it reports:
  instructed_scratch  exact match on steps + answer with the rule stated in the prompt
                      plus k solved examples                                      (slow path, execution)
  instructed_direct   answer only, rule stated, k examples, no steps
  icl_scratch         steps + answer, k examples, rule NOT stated                  (slow path, induction)
  zero_direct         answer only, nothing else                                   (compiled path)
  free                open-ended generation after "Input: x": accuracy, mean steps emitted, mean tokens
"""
from __future__ import annotations

import os
import random
import sys
import time

import torch
import yaml

from hdelta.encoding import Encoder
from hdelta.metrics import generate, score_pairs
from hdelta.modelwrap import load_model
from hdelta.tasks import make_task


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


@torch.no_grad()
def free_probe(model, enc, task, pz, rule_text: bool, max_new: int = 48):
    prompts = [task.make_pair(p, "free", rule_text=rule_text)[0] for p in pz]
    accs, steps, toks = [], [], []
    for i in range(0, len(prompts), 16):
        ids, attn = enc.encode_prompts_left(prompts[i:i + 16])
        out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=enc.pad, temperature=None, top_p=None, top_k=None)
        for row, p in zip(out[:, ids.shape[1]:], pz[i:i + 16]):
            text = enc.tok.decode(row, skip_special_tokens=True).split("\n\n")[0]
            ans, n_steps = task.parse_generation(text)
            accs.append(ans == p.answer)
            steps.append(n_steps)
            cut = text.find(f"{task.answer_label}:")
            used = text if cut < 0 else text[:cut + len(task.answer_label) + 1 + len(p.answer) + 1]
            toks.append(len(enc.ids(used)))
    return mean(accs), mean(steps), mean(toks)


def main():
    cfg = yaml.safe_load(open(os.environ.get("PROBE_CONFIG", "configs/base.yaml"),
                              encoding="utf-8"))
    if os.environ.get("PROBE_ALPHABET"):
        cfg["task"]["alphabet"] = os.environ["PROBE_ALPHABET"]
    task = make_task(cfg["task"])
    print("rule A:", task.describe("A"))
    rng = random.Random(10_000)
    pz = [task.sample("A", rng) for _ in range(48)]
    shots_pool = [task.sample("A", rng) for _ in range(4)]
    for name in sys.argv[1:]:
        t0 = time.time()
        model, tok = load_model(name, os.environ.get("PROBE_DTYPE", "float32"), "cuda")
        enc = Encoder(tok, "cuda")
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n=== {name}  ({n:.0f}M params, loaded in {time.time()-t0:.0f}s) ===", flush=True)
        with torch.no_grad():
            prompts = [task.make_pair(p, "direct")[0] for p in pz]
            budget = 2 + max(len(enc.ids(" " + p.answer + "\n")) for p in pz)
            gens = generate(model, enc, prompts, budget)
            zd = mean(g == p.answer for g, p in zip(gens, pz))
            fa, fs, ft = free_probe(model, enc, task, pz, rule_text=False)
            print(f"  zero_direct={zd:.2f}   free(no rule): acc={fa:.2f} steps={fs:.1f} tokens={ft:.1f}", flush=True)
            fa, fs, ft = free_probe(model, enc, task, pz, rule_text=True)
            print(f"  free(rule stated, no examples): acc={fa:.2f} steps={fs:.1f} tokens={ft:.1f}", flush=True)
            for k in (1, 2, 4):
                shots = shots_pool[:k]
                pr, rs = zip(*[task.make_pair(p, "scratchpad", shots=shots, rule_text=True) for p in pz])
                ex, _, ta = score_pairs(model, enc, pr, rs)
                pr2, rs2 = zip(*[task.make_pair(p, "direct", shots=shots, rule_text=True) for p in pz])
                ex2, _, _ = score_pairs(model, enc, pr2, rs2)
                pr3, rs3 = zip(*[task.make_pair(p, "scratchpad", shots=shots) for p in pz])
                ex3, _, _ = score_pairs(model, enc, pr3, rs3)
                print(f"  shots={k}: instructed_scratch={mean(ex):.2f} (tok {mean(ta):.2f})  "
                      f"instructed_direct={mean(ex2):.2f}  icl_scratch(no rule)={mean(ex3):.2f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

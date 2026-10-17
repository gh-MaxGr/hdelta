"""Gate 1, generated-output version: free-form generation WITH shots.

    python probe_generated.py Qwen/Qwen2.5-3B-Instruct      (PROBE_CONFIG selects the task)

probe_base.py's free probe states the rule but gives no examples, so a 0.00 there
does not separate 'cannot do the task' from 'did not use the expected format'.
This re-runs it at k shots and dumps raw generations so the difference is visible.
"""
import os
import random
import sys

import torch
import yaml

from hdelta.encoding import Encoder
from hdelta.metrics import score_pairs
from hdelta.modelwrap import load_model
from hdelta.tasks import make_task

cfg = yaml.safe_load(open(os.environ.get("PROBE_CONFIG", "configs/workspace.yaml"),
                          encoding="utf-8"))
task = make_task(cfg["task"])
rng = random.Random(10_000)
pz = [task.sample("A", rng) for _ in range(48)]
shots_pool = [task.sample("A", rng) for _ in range(4)]

name = sys.argv[1] if len(sys.argv) > 1 else cfg["model"]["name"]
model, tok = load_model(name, "bfloat16", "cuda")
enc = Encoder(tok, "cuda")
print(f"=== {name} : generated output, rule stated, k shots ===", flush=True)


@torch.no_grad()
def free_with_shots(k, dump=0):
    shots = shots_pool[:k]
    prompts = [task.make_pair(p, "free", shots=shots, rule_text=True)[0] for p in pz]
    accs, steps, shown = [], [], 0
    for i in range(0, len(prompts), 12):
        ids, attn = enc.encode_prompts_left(prompts[i:i + 12])
        out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=64,
                             do_sample=False, pad_token_id=enc.pad,
                             temperature=None, top_p=None, top_k=None)
        for row, p in zip(out[:, ids.shape[1]:], pz[i:i + 12]):
            text = enc.tok.decode(row, skip_special_tokens=True).split("\n\n")[0]
            ans, n = task.parse_generation(text)
            accs.append(ans == p.answer)
            steps.append(n)
            if shown < dump:
                shown += 1
                print(f"  --- {p.input_str}\n      want={p.answer!r} got={ans!r}\n"
                      f"      raw={text!r}", flush=True)
    return sum(accs) / len(accs), sum(steps) / len(steps)


with torch.no_grad():
    for k in (0, 1, 2, 4):
        acc, st = free_with_shots(k, dump=2 if k in (0, 2) else 0)
        print(f"  shots={k}: free_generated_acc={acc:.2f}  mean_steps={st:.1f}", flush=True)

    # Generated DIRECT: the compiled path the fast weights are meant to install.
    for k in (2,):
        shots = shots_pool[:k]
        prompts = [task.make_pair(p, "direct", shots=shots, rule_text=True)[0] for p in pz]
        accs = []
        for i in range(0, len(prompts), 12):
            ids, attn = enc.encode_prompts_left(prompts[i:i + 12])
            out = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=16,
                                 do_sample=False, pad_token_id=enc.pad,
                                 temperature=None, top_p=None, top_k=None)
            for row, p in zip(out[:, ids.shape[1]:], pz[i:i + 12]):
                text = enc.tok.decode(row, skip_special_tokens=True).split("\n")[0].strip()
                accs.append(text == p.answer)
        print(f"  shots={k}: direct_generated_acc={sum(accs)/len(accs):.2f}", flush=True)

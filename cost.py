"""What does each path cost, and does adding memory make the system expensive?

Per-call cost of the three paths, measured on the real model one call at a time (as a deployed
system answering one decision at a time would run them):

    instinct     bare prompt, answer out (stops at the newline)
    memory       the same prompt with 1 or 2 recalled notes in front
    deliberate   rule stated + 2 worked examples, reasoning written out
                   as_run   - generate_text's behaviour: runs on to 128 tokens
                   stopped  - stops at the end of the answer line, as a deployed system would

Dimensions: prompt tokens, generated tokens, GPU latency, compute (2 * params * tokens
processed), KV-cache bytes, CPU cost of retrieval, storage. Then the per-call costs are
combined with the call counts pipeline.py measured, giving the cost of each SYSTEM per 192
decisions.

One cost is not compute: a note can only be written once the system knows the instinct was
wrong. For a failure that went to deliberation the truth is already in hand. For a
CONFIDENT failure it takes an outcome signal from outside - free when a game engine or test
suite reports the result anyway, a real tool call when the verifier is the expensive tool.
Both counts are reported for the four no-pain runs.

    .venv/Scripts/python.exe cost.py
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from failure_memory import K, Answers, note, reason_key, recall_key, with_notes
from hdelta.encoding import Encoder
from hdelta.tasks.base import Puzzle
from idiom import load_compiled
from pipeline import TAUS
from predict_failure import BASELINE, RULE, Reader, circumstance, domain

TIMING_RUN = "results/variance/conjunction_s1"
N_TIME = 32          # inputs timed per path
N_TIME_ASRUN = 12    # the 128-token path is slow; fewer is plenty for a mean


class AnswerLineStop(StoppingCriteria):
    """Stop once the generated text contains a complete '<answer_label>: ...' line."""

    def __init__(self, tok, start, label):
        self.tok, self.start = tok, start
        self.rx = re.compile(rf"{re.escape(label)}:[^\n]*\n")

    def __call__(self, input_ids, scores, **kw):
        return bool(self.rx.search(self.tok.decode(input_ids[0, self.start:],
                                                    skip_special_tokens=True)))


def timed(model, enc, prompt, max_new, stop_newline=True, stop_answer_label=None):
    ids, attn = enc.encode_prompts_left([prompt])
    kw = dict(input_ids=ids, attention_mask=attn, max_new_tokens=max_new, do_sample=False,
              num_beams=1, pad_token_id=enc.pad, temperature=None, top_p=None, top_k=None)
    if stop_newline:
        kw["eos_token_id"] = enc.newline_id
    if stop_answer_label:
        kw["stopping_criteria"] = StoppingCriteriaList(
            [AnswerLineStop(enc.tok, ids.shape[1], stop_answer_label)])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**kw)
    torch.cuda.synchronize()
    return ids.shape[1], out.shape[1] - ids.shape[1], time.perf_counter() - t0


def measure_paths(device="cuda"):
    cfg, task, model, tok, ctrl, norm = load_compiled(Path(TIMING_RUN), device)
    enc = Encoder(tok, device)
    xs = domain(task)
    rng = random.Random(0)
    sample = rng.sample(xs, N_TIME)
    truth = {x: task.answer_under(RULE, x) for x in xs}
    pz = {x: Puzzle(x, [], truth[x], RULE, {}) for x in xs}
    direct = {x: task.make_pair(pz[x], "direct")[0] for x in xs}

    def notes_for(x, n):
        """n realistic notes: other items of the same circumstance, with a wrong move."""
        same = [y for y in xs if y != x and circumstance(y) == circumstance(x)]
        recs = []
        for y in rng.sample(same, n):
            wrong = rng.choice([f"#q{i}" for i in range(1, 7) if f"#q{i}" != truth[y]])
            recs.append({"input": y, "move": wrong, "truth": truth[y]})
        return recs

    mem1 = {x: with_notes(task, direct[x], notes_for(x, 1), True) for x in sample}
    mem2 = {x: with_notes(task, direct[x], notes_for(x, 2), True) for x in sample}
    shots = [task.sample(RULE, random.Random(7)) for _ in range(2)]
    delib = {x: task.make_pair(pz[x], "free", shots=shots, rule_text=True)[0] for x in sample}

    for p in list(direct.values())[:3]:                     # warm-up
        timed(model, enc, p, 8)
    paths = {
        "instinct": [timed(model, enc, direct[x], 8) for x in sample],
        "memory (1 note)": [timed(model, enc, mem1[x], 8) for x in sample],
        "memory (2 notes)": [timed(model, enc, mem2[x], 8) for x in sample],
        "deliberate, stopped": [timed(model, enc, delib[x], 128, stop_newline=False,
                                      stop_answer_label=task.answer_label) for x in sample],
        "deliberate, as run": [timed(model, enc, delib[x], 128, stop_newline=False)
                               for x in sample[:N_TIME_ASRUN]],
    }
    # the learned delta itself: is the instinct slower than the frozen model? (merge cache)
    ctrl.set_apply_delta(False)
    frozen = [timed(model, enc, direct[x], 8) for x in sample]
    ctrl.set_apply_delta(True)

    # the confidence readout pipeline.py used is a separate forward pass; a deployed system
    # reads the same numbers from the generation's own scores. Time the separate pass anyway.
    rd = Reader(model, enc, device)
    t_belief = []
    for x in sample:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rd.belief([direct[x]], chunk=1)
        torch.cuda.synchronize(); t_belief.append(time.perf_counter() - t0)

    mc = model.config
    n_params = sum(p.numel() for p in model.parameters())
    head_dim = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
    kv_bytes_per_token = mc.num_hidden_layers * 2 * mc.num_key_value_heads * head_dim * 2
    per = {}
    for name, rows in paths.items():
        pt = st.mean(r[0] for r in rows)
        gt = st.mean(r[1] for r in rows)
        per[name] = {"prompt_tokens": pt, "generated_tokens": gt,
                     "latency_ms": 1000 * st.mean(r[2] for r in rows),
                     "gflops": 2 * n_params * (pt + gt) / 1e9,
                     "kv_cache_mb": kv_bytes_per_token * (pt + gt) / 2 ** 20, "n": len(rows)}

    # CPU side of memory: retrieval over a full K-note memory, and writing a note
    mem = []
    for y in rng.sample(xs, K):
        mem.append({"input": y, "move": "#q1", "truth": truth[y]})
    t0 = time.perf_counter()
    keys = [reason_key(task, r) for r in mem]
    t_key = (time.perf_counter() - t0) / K
    reps = 2000
    t0 = time.perf_counter()
    for i in range(reps):
        recall_key(task, xs[i % len(xs)], mem, keys)
    t_recall = (time.perf_counter() - t0) / reps
    store = len(json.dumps(mem).encode()) + sum(len(note(task, r, True).encode()) for r in mem)

    out = {"run": TIMING_RUN, "model": cfg["model"]["name"], "n_params": n_params,
           "kv_bytes_per_token": kv_bytes_per_token, "paths": per,
           # NOT like-for-like: the frozen model has not learned the format and runs to the
           # 8-token cap. Compare ms per generated token (~34 either way), not raw latency.
           "frozen_instinct_latency_ms": 1000 * st.mean(r[2] for r in frozen),
           "frozen_instinct_gen_tokens": st.mean(r[1] for r in frozen),
           "separate_confidence_pass_ms": 1000 * st.mean(t_belief),
           "retrieval_us": 1e6 * t_recall, "note_write_us": 1e6 * t_key,
           "memory_store_bytes_full": store,
           "delta_pt_bytes": (Path(TIMING_RUN) / "delta.pt").stat().st_size,
           "weights_gb": n_params * 2 / 2 ** 30}
    del model
    torch.cuda.empty_cache()
    return out


def feedback_counts(device="cuda"):
    """For each no-pain run: how many notes came from DEFERRED failures (truth already in hand)
    vs CONFIDENT failures (needs an outcome signal from outside), per tau."""
    rows = []
    for run in BASELINE:
        cfg, task, model, tok, ctrl, norm = load_compiled(Path(run), device)
        enc = Encoder(tok, device)
        rd = Reader(model, enc, device)
        xs = domain(task)
        truth = [task.answer_under(RULE, x) for x in xs]
        direct = [task.make_pair(Puzzle(x, [], t, RULE, {}), "direct")[0] for x, t in zip(xs, truth)]
        inst = Answers(model, enc)(direct)
        idx = {f"#q{i}": i - 1 for i in range(1, 7)}
        bel = rd.belief(direct)
        conf = np.array([bel[i, idx[a]] if a in idx else 0.0 for i, a in enumerate(inst)])
        fail = np.array([a != t for a, t in zip(inst, truth)])
        rows.append({"run": run, "failures": int(fail.sum()), "tau": {
            str(t): {"deferred_failures": int((fail & (conf < t)).sum()),
                     "confident_failures": int((fail & (conf >= t)).sum())} for t in TAUS}})
        del model
        torch.cuda.empty_cache()
    return rows


def systems(paths, pipe):
    """Cost per 192 decisions for each system, at the lowest tau where gate>delib >= 0.99."""
    c_i, c_m, c_d = paths["instinct"], paths["memory (2 notes)"], paths["deliberate, stopped"]
    out = []
    for r in pipe:
        if float(r["pain_lambda"]) != 0.0:
            continue
        tau = next((t for t, c in r["tau"].items() if c["acc"]["gate>delib"]["mean"] >= 0.99), None)
        if tau is None:
            continue
        c = r["tau"][tau]
        n = r["n"]
        rows = {
            "always deliberate": (0, 0, n, r["delib_acc"]),
            "instinct only": (n, 0, 0, c["acc"]["instinct"]["mean"]),
            "gate>delib": (n, 0, c["gate_delib_calls"], c["acc"]["gate>delib"]["mean"]),
            "full (memory)": (n, c["memory_calls"], c["delib_calls"], c["acc"]["full"]["mean"]),
        }
        sysrow = {}
        for name, (ni, nm, nd, acc) in rows.items():
            sysrow[name] = {
                "acc": acc, "calls": (ni, nm, nd),
                "gpu_s": (ni * c_i["latency_ms"] + nm * c_m["latency_ms"] + nd * c_d["latency_ms"]) / 1000,
                "tflops": (ni * c_i["gflops"] + nm * c_m["gflops"] + nd * c_d["gflops"]) / 1000,
                "gen_tokens": ni * c_i["generated_tokens"] + nm * c_m["generated_tokens"]
                              + nd * c_d["generated_tokens"]}
        out.append({"run": r["run"], "tau": tau, "systems": sysrow})
    return out


def show(res):
    p = res["paths"]
    print(f"\nmodel {res['model']}  ({res['n_params'] / 1e9:.2f} B params, "
          f"{res['weights_gb']:.1f} GB bf16)   timed on {res['run']}, one call at a time")
    print(f"\n{'path':<22}{'prompt tok':>11}{'gen tok':>9}{'latency ms':>12}{'GFLOPs':>9}"
          f"{'KV MB':>8}{'x instinct':>12}")
    base = p["instinct"]["latency_ms"]
    for name, v in p.items():
        print(f"{name:<22}{v['prompt_tokens']:>11.0f}{v['generated_tokens']:>9.1f}"
              f"{v['latency_ms']:>12.1f}{v['gflops']:>9.0f}{v['kv_cache_mb']:>8.1f}"
              f"{v['latency_ms'] / base:>12.2f}")
    print(f"\ninstinct on the FROZEN model (delta off): {res['frozen_instinct_latency_ms']:.1f} ms"
          f" for {res.get('frozen_instinct_gen_tokens', float('nan')):.1f} generated tokens"
          f"   separate confidence pass as pipeline.py ran it: {res['separate_confidence_pass_ms']:.1f} ms")
    print(f"memory, CPU side: retrieval {res['retrieval_us']:.1f} us over {K} notes, "
          f"note write {res['note_write_us']:.1f} us;  full store {res['memory_store_bytes_full']} bytes")
    print(f"for scale: the learned delta is {res['delta_pt_bytes'] / 2**20:.0f} MB, "
          f"the weights {res['weights_gb']:.1f} GB")

    print("\nPER 192 DECISIONS (tau = lowest where gate>delib >= 0.99; 'stopped' deliberation)")
    print(f"{'run':<26}{'system':<20}{'acc':>7}{'calls i/m/d':>16}{'GPU s':>8}{'TFLOPs':>8}"
          f"{'x instinct-only':>17}{'x gate>delib':>14}")
    for s in res["systems"]:
        ref_i = s["systems"]["instinct only"]["gpu_s"]
        ref_g = s["systems"]["gate>delib"]["gpu_s"]
        for name, v in s["systems"].items():
            ni, nm, nd = v["calls"]
            print(f"{s['run'][8:] + ' t=' + s['tau']:<26}{name:<20}{v['acc']:>7.3f}"
                  f"{f'{ni}/{nm:.0f}/{nd:.0f}':>16}{v['gpu_s']:>8.1f}{v['tflops']:>8.1f}"
                  f"{v['gpu_s'] / ref_i:>17.2f}{v['gpu_s'] / ref_g:>14.2f}")
    print("\nFEEDBACK: notes need to know the instinct was wrong")
    print(f"{'run':<34}" + "".join(f"{'t=' + str(t):>14}" for t in TAUS))
    for f in res["feedback"]:
        print(f"{f['run'][8:] + ' (' + str(f['failures']) + ' failures)':<34}" + "".join(
            f"{str(v['deferred_failures']) + ' def/' + str(v['confident_failures']) + ' conf':>14}"
            for v in f["tau"].values()))
    print("'def' = failure was deferred; truth already in hand.  'conf' = instinct was confidently")
    print("wrong and shipped; a note needs the environment to report the outcome.")


if __name__ == "__main__":
    res = measure_paths()
    res["feedback"] = feedback_counts()
    res["systems"] = systems(res["paths"], json.load(open("results/pipeline.json", encoding="utf-8")))
    json.dump(res, open("results/cost.json", "w", encoding="utf-8"), indent=1)
    show(res)

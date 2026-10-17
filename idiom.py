"""Use a compiled skill.

This project has compiled 44 skills into 15 GB of delta.pt files and never once used one.
Every consumer on disk is an analysis script - check_bloom, probe_hidden, probe_hidden_xseed -
that measures a compiled state and throws it away. This loads one and does the job it was
compiled for.

    # ask it something, with an empty context window
    .venv/Scripts/python.exe idiom.py "launch analyse_drift.py detached, results/foo"

    # the payoff, side by side: frozen model vs the same model with the skill
    .venv/Scripts/python.exe idiom.py --compare "wait for the seeds stage to finish"

    # keep it resident and ask repeatedly
    .venv/Scripts/python.exe idiom.py --repl

The prompt carries no rules, no examples and no CLAUDE.md - only the task. Whatever the
model gets right, it gets right from its weights. `--compare` toggles the adapter off and
runs the identical prompt through the frozen model, which is the honest control: if the
frozen model already answers correctly there was no skill to compile.

The linter that gated every write during training also scores the output here, so the
verdict is the same standard the skill was trained against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from hdelta.encoding import Encoder
from hdelta.metrics import generate
from hdelta.modelwrap import groups_from_config, install_hdelta, load_model
from hdelta.tasks import make_task


def load_compiled(run: Path, device: str = "cuda"):
    """Rebuild the model the run produced: frozen weights plus its learned dW."""
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    task = make_task(cfg["task"])
    model, tok = load_model(cfg["model"]["name"], cfg["model"].get("dtype", "bfloat16"), device)
    ctrl, _ = install_hdelta(model, groups_from_config(cfg["plastic"]["groups"]),
                             max_fraction=cfg["plastic"].get("max_fraction", 0.05),
                             postsynaptic=cfg["plastic"].get("postsynaptic", "error"),
                             decay_enabled=True, homeo_enabled=True,
                             update_norm=cfg["plastic"].get("update_norm", "adam"))
    snap = torch.load(run / "delta.pt", map_location=device)
    ctrl.restore({k: v.to(device) for k, v in snap.items()})
    ctrl.invalidate()
    norm = sum(float(v.norm()) ** 2 for v in snap.values()) ** 0.5
    return cfg, task, model, tok, ctrl, norm


def which_rule(task, text: str) -> str:
    """Route the request to a rule by the shape of the ask, the way the task phrases it."""
    t = text.lower()
    for r, spec in getattr(task, "spec", {}).items():
        fam = spec.get("family")
        if fam == "launch" and "detach" in t:
            return r
        if fam == "gate" and ("wait" in t or "until" in t or "finish" in t):
            return r
    for r, spec in getattr(task, "spec", {}).items():
        if spec.get("family") == "invoke":
            return r
    return task.rule_names()[0]


def ask(model, enc, task, ctrl, text: str, compiled: bool, max_new: int) -> str:
    """One answer, from a prompt containing nothing but the task."""
    prompt = f"{task.header}{task.input_label}: {text}\n{task.answer_label}:"
    ctrl.set_apply_delta(compiled)
    try:
        out = generate(model, enc, [prompt], max_new)[0]
    finally:
        ctrl.set_apply_delta(True)
    return out.strip()


def verdict(task, rule: str, command: str) -> str:
    ok, why = task.check(rule, command)
    return "PASS" if ok else f"REJECT ({why})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", help="the request, in plain words")
    ap.add_argument("--run", default="results/idiom_v2",
                    help="run directory holding config.json and delta.pt")
    ap.add_argument("--compare", action="store_true",
                    help="also answer with the adapter off (the frozen control)")
    ap.add_argument("--repl", action="store_true")
    ap.add_argument("--max-new", type=int, default=96)
    a = ap.parse_args()

    run = Path(a.run)
    if not (run / "delta.pt").exists():
        sys.exit(f"no compiled skill at {run/'delta.pt'}")
    cfg, task, model, tok, ctrl, norm = load_compiled(run)
    enc = Encoder(tok, cfg["model"].get("device", "cuda"))
    print(f"loaded {cfg['model']['name']} + compiled skill from {run} (|dW| = {norm:.3f})")
    print(f"rules: {', '.join(task.rule_names())}\n")

    def answer(text: str):
        rule = which_rule(task, text)
        comp = ask(model, enc, task, ctrl, text, True, a.max_new)
        print(f"  task      {text}")
        print(f"  compiled  {comp}")
        print(f"            -> {verdict(task, rule, comp)}")
        if a.compare:
            base = ask(model, enc, task, ctrl, text, False, a.max_new)
            print(f"  frozen    {base}")
            print(f"            -> {verdict(task, rule, base)}")
        print()

    if a.task:
        answer(" ".join(a.task))
    if a.repl or not a.task:
        print("empty line to quit\n")
        while True:
            try:
                line = input("task> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            answer(line)


if __name__ == "__main__":
    main()

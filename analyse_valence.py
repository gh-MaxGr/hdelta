"""Compare the valence arms on the thing that actually decides it.

The accuracy alone cannot tell "learned to play" from "collapsed to one move": diplomat's
always-ACCEPT policy scores 0.70, which looks healthy. The baseline run posted 0.44 / 0.15 /
0.52 while emitting REFUSE for all 48 probe situations in all three dispositions.

So this reports, per arm and per disposition:

    cold        compiled accuracy on held-out situations
    floor       the best CONSTANT policy for that disposition - the number to beat
    distinct    how many different moves the compiled reflex emitted over 48 situations
    profile     which move heads it emitted, against what the reference plays

**distinct == 1 means collapse, whatever the accuracy says.**

Loading each arm's delta.pt costs a model load, so this is a few minutes per arm.

    .venv/Scripts/python.exe analyse_valence.py
    .venv/Scripts/python.exe analyse_valence.py results/world results/world_contrast
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

from hdelta.encoding import Encoder
from hdelta.metrics import generate
from hdelta.tasks.world import CONSTANT_FLOOR
from idiom import load_compiled

DEFAULT = ["results/world", "results/world_positive",
           "results/world_contrast", "results/world_contrast_v2",
           "results/world_s1", "results/world_contrast_s1", "results/world_contrast_s1_v2"]
ARM = {"skip": "unsigned", "positive": "reward only", "contrast": "signed"}


def main(runs):
    for rd in runs:
        run = Path(rd)
        if not (run / "delta.pt").exists():
            print(f"\n=== {rd}: no delta.pt (not run yet) ===")
            continue
        cfg, task, model, tok, ctrl, norm = load_compiled(run)
        enc = Encoder(tok, cfg["model"].get("device", "cuda"))
        arm = ARM.get(cfg["stream"].get("on_wrong", "?"), cfg["stream"].get("on_wrong", "?"))
        print(f"\n=== {rd}   arm={arm}  seed={cfg.get('seed')}  |dW|={norm:.3f} ===")
        print(f"{'disposition':>12} {'cold':>6} {'floor':>6} {'distinct':>9}  profile")
        for i, r in enumerate(CONSTANT_FLOOR):
            rng = random.Random(10_000 + 17 * i)
            pz = [task.sample(r, rng) for _ in range(48)]
            ctrl.set_apply_delta(True)
            outs = generate(model, enc, [task.make_pair(p, "direct")[0] for p in pz], 24)
            moves = [task.parse_generation(o.strip())[0] for o in outs]
            acc = sum(task.evaluate(r, m, p.input_str)[0] for m, p in zip(moves, pz)) / len(pz)
            heads = Counter(m.split()[0] if m.split() else "<empty>" for m in moves)
            flag = "  COLLAPSED" if len(set(moves)) == 1 else ""
            beat = "beats floor" if acc > CONSTANT_FLOOR[r] else "below floor"
            print(f"{r:>12} {acc:>6.2f} {CONSTANT_FLOOR[r]:>6.2f} {len(set(moves)):>9}  "
                  f"{dict(heads.most_common(3))}  {beat}{flag}")
        del model
        import torch
        torch.cuda.empty_cache()

    print("\nCollapse is `distinct == 1`. An arm only counts as working if it emits more than")
    print("one move AND holds above the floor - a constant policy can clear neither, or the")
    print("floor alone by luck (diplomat's always-ACCEPT scores 0.70).")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

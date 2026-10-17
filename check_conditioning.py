"""Is the surviving behaviour conditioned on the situation, or just noise?

The signed arms emit more than one move, which clears the collapse test. But `distinct > 1`
has two very different causes:

  DISCRIMINATION  the move depends on the situation - the thing a disposition is
  NOISE           the move varies because a broken training signal shoved the weights around

Accuracy cannot separate them, and neither can the move count. This can: tabulate the emitted
move against the one situation feature with an unambiguous correct response - `standing`. A
reflex that discriminates changes its move distribution as standing changes. A noisy one does
not, however many distinct moves it emits.

Reported per disposition:

    TVD   total variation distance between the move distribution under `hostile` and under
          `friendly`. 0.0 means the reflex ignores standing entirely.

The enumerated reference is included as the ceiling: it is what real conditioning looks like
on this task.

    .venv/Scripts/python.exe check_conditioning.py results/world_contrast results/world_s1
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

from hdelta.encoding import Encoder
from hdelta.metrics import generate
from hdelta.tasks.world import CONSTANT_FLOOR, STANDING, Situation
from idiom import load_compiled

DEFAULT = ["results/world_contrast", "results/world_contrast_s1", "results/world_positive"]
N_PER_STANDING = 32


def head(m: str) -> str:
    p = m.split()
    return p[0].upper() if p else "<none>"


def tvd(a: Counter, b: Counter) -> float:
    na, nb = max(sum(a.values()), 1), max(sum(b.values()), 1)
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def situations_by_standing(task, rule, rng):
    """Matched sets: identical draws except for `standing`, so nothing else co-varies."""
    out = {s: [] for s in STANDING}
    for _ in range(N_PER_STANDING):
        base = Situation.parse(task.sample_input(rule, rng))
        for s in STANDING:
            out[s].append(Situation(dict(base.stock), dict(base.need), s,
                                    base.give_n, base.give_g, base.want_n, base.want_g).text())
    return out


def main(runs):
    for rd in runs:
        run = Path(rd)
        if not (run / "delta.pt").exists():
            print(f"\n=== {rd}: no delta.pt ===")
            continue
        cfg, task, model, tok, ctrl, norm = load_compiled(run)
        enc = Encoder(tok, cfg["model"].get("device", "cuda"))
        print(f"\n=== {rd}  arm={cfg['stream'].get('on_wrong')}  seed={cfg.get('seed')} ===")
        print(f"{'disposition':>12} {'source':>10} {'hostile':>22} {'friendly':>22} {'TVD':>6}")
        for i, r in enumerate(CONSTANT_FLOOR):
            rng = random.Random(4242 + i)
            sets = situations_by_standing(task, r, rng)
            rows = {}
            for src in ("reflex", "reference"):
                cnt = {}
                for s in STANDING:
                    if src == "reference":
                        mv = [task.solve(r, x)[1] for x in sets[s]]
                    else:
                        from hdelta.tasks.base import Puzzle
                        pz = [Puzzle(x, [], "", r, {}) for x in sets[s]]
                        ctrl.set_apply_delta(True)
                        outs = generate(model, enc, [task.make_pair(p, "direct")[0] for p in pz], 24)
                        mv = [task.parse_generation(o.strip())[0] for o in outs]
                    cnt[s] = Counter(head(m) for m in mv)
                rows[src] = cnt
            for src in ("reflex", "reference"):
                c = rows[src]
                d = tvd(c["hostile"], c["friendly"])
                f = lambda x: ",".join(f"{k}:{v}" for k, v in x.most_common(3))
                print(f"{r if src == 'reflex' else '':>12} {src:>10} "
                      f"{f(c['hostile'])[:22]:>22} {f(c['friendly'])[:22]:>22} {d:>6.2f}")
        del model
        import torch
        torch.cuda.empty_cache()

    print("\nTVD 0.00 = the reflex ignores standing, so `distinct > 1` was noise, not a")
    print("disposition. Compare each reflex against the reference row directly beneath it:")
    print("that is what conditioning on this feature actually looks like.")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

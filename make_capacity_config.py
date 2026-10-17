"""Generate a capacity-test config: N interleaved, tagged cipher rules.

    python make_capacity_config.py 8 --block 40 --per-rule 320 --out configs/capacity_n8.yaml

The question is how many skills one global h-delta state can hold at once. Each rule is a
two-step digit procedure with its own tag in the prompt, so the compiled path must learn
(tag, input) -> answer for every rule simultaneously. Rules are interleaved as short
blocks that cycle, rather than long phases, so every rule is repeatedly revisited and
retention can be measured against time-since-last-seen.

Rule design (all over digits 0..9, both steps wrap):
  additive    [shift a, shift b]  compiles to  d -> d + (a+b)
  reflective  [shift a, mirror]   compiles to  d -> (9 - a) - d
Two additive rules never agree (they differ by a non-zero constant); two reflective rules
never agree; an additive and a reflective rule agree on at most 2 of 10 digits, so on a
3-digit answer they coincide with probability <= 0.008. Attribution stays unambiguous.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# (tag, steps, closed form) - ordered so that a prefix of length N is a valid rule set
RULES = [
    ("A7", [["shift", 4], ["shift", 3]], "d -> d + 7"),
    ("M6", [["shift", 3], ["mirror"]], "d -> 6 - d"),
    ("A3", [["shift", 1], ["shift", 2]], "d -> d + 3"),
    ("M2", [["shift", 7], ["mirror"]], "d -> 2 - d"),
    ("A9", [["shift", 5], ["shift", 4]], "d -> d + 9"),
    ("M4", [["shift", 5], ["mirror"]], "d -> 4 - d"),
    ("A1", [["shift", 6], ["shift", 5]], "d -> d + 1"),
    ("M8", [["shift", 1], ["mirror"]], "d -> 8 - d"),
]


def check_distinct(names, steps_by_name, n=10):
    """Every pair of rules must differ on most single digits."""
    def apply(steps, d):
        for op in steps:
            if op[0] == "shift":
                d = (d + int(op[1])) % n
            elif op[0] == "mirror":
                d = n - 1 - d
            else:
                raise ValueError(op[0])
        return d
    worst = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agree = sum(apply(steps_by_name[a], d) == apply(steps_by_name[b], d) for d in range(n))
            worst = max(worst, agree / n)
            assert agree < n, f"rules {a} and {b} are identical"
    return worst


def _merge(a: dict, b: dict) -> dict:
    import copy
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def build(n_rules: int, block: int, per_rule: int, cadence: int, probe_n: int, seed: int) -> dict:
    """A complete, standalone config.

    It is emitted whole rather than as an override because `task.rules` is a dict, and the
    override merger would *add* these rules to the base config's rather than replace them.
    """
    root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    cfg = _merge(cfg, yaml.safe_load((root / "configs/instructed_3b.yaml").read_text(encoding="utf-8")))

    chosen = RULES[:n_rules]
    names = [t for t, _, _ in chosen]
    steps = {t: s for t, s, _ in chosen}
    worst = check_distinct(names, steps)
    cycles = per_rule // block
    phases = [{"rule": names[(i * n_rules + j) % n_rules], "puzzles": block}
              for i in range(cycles) for j in range(n_rules)]
    total = sum(p["puzzles"] for p in phases)

    cfg["_comment"] = (f"capacity test: {n_rules} tagged rules interleaved in {block}-puzzle blocks, "
                       f"{cycles} cycles, {per_rule} puzzles per rule, {total} total; "
                       f"worst pairwise per-digit agreement {worst:.0%}. "
                       f"Standalone config: run with --config, not --override.")
    cfg["seed"] = seed
    cfg["task"] = {"name": "cipher", "alphabet": "digits", "alphabet_size": 10, "length": [3, 3],
                   "tagged": True, "rules": {t: s for t, s, _ in chosen}}
    cfg["stream"].update({
        "phases": phases,
        "probe_every": cadence,
        "probe_n": probe_n,
        "probe_depth": False,     # not the question here; saves 3 scoring passes per rule
        "free_probe": False,      # hand-over already settled; capacity needs retention
        "rule_text": True,
        "self_generate": True,
        "on_wrong": "teach",
        "consolidate": "verified",
        "n_shots": 4,
        "gen_max_new": 40,
    })
    cfg["rem"]["probe_around"] = False   # REM cost already characterised; halves the probe count
    cfg["eval"]["lm_windows"] = 12
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n_rules", type=int)
    ap.add_argument("--block", type=int, default=40, help="puzzles per rule before switching")
    ap.add_argument("--per-rule", type=int, default=320, help="total puzzles per rule")
    ap.add_argument("--cadence", type=int, default=None, help="probe_every (default: block)")
    ap.add_argument("--probe-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cadence = args.cadence or args.block
    cfg = build(args.n_rules, args.block, args.per_rule, cadence, args.probe_n, args.seed)
    out = Path(args.out or f"configs/capacity_n{args.n_rules}.yaml")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    total = sum(p["puzzles"] for p in cfg["stream"]["phases"])
    print(f"{out}: {args.n_rules} rules, {len(cfg['stream']['phases'])} blocks, {total} puzzles, "
          f"probe every {cadence} ({total // cadence} probes)")


if __name__ == "__main__":
    main()

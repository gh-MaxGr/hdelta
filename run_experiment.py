"""Run one self-upgrade sandbox experiment.

    python run_experiment.py --config configs/base.yaml --out results/full
    python run_experiment.py --config configs/base.yaml --override configs/ablations/no_rem.yaml --out results/no_rem
    python run_experiment.py --config configs/base.yaml --set rem.eta=0.5 --set stream.phases[0].puzzles=100
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import yaml

from hdelta.experiment import Experiment


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_set(cfg: dict, expr: str) -> None:
    """--set a.b[0].c=value  (value parsed as YAML)."""
    key, _, raw = expr.partition("=")
    value = yaml.safe_load(raw)
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", key)
    cur = cfg
    for i, p in enumerate(parts):
        last = i == len(parts) - 1
        idx = int(p[1:-1]) if p.startswith("[") else p
        if last:
            cur[idx] = value
        else:
            cur = cur[idx]


def load_config(config: str, overrides: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    for o in overrides:
        cfg = deep_merge(cfg, yaml.safe_load(Path(o).read_text(encoding="utf-8")) or {})
    for s in sets:
        apply_set(cfg, s)
    cfg["_root"] = str(Path(__file__).resolve().parent)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override, args.set)
    exp = Experiment(cfg, Path(args.out), quiet=args.quiet)
    summary = exp.run()
    print(json.dumps({k: v for k, v in summary.items() if k in ("phases", "base_skills", "rem")}, indent=1))


if __name__ == "__main__":
    main()

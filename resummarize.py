"""Recompute summary.json / summary.md for finished runs from their probe logs.

    python resummarize.py results/suite3b/full results/suite3b/none ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hdelta.report import load_jsonl, summarize, to_markdown

for arg in sys.argv[1:]:
    d = Path(arg)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    recs = [r for r in load_jsonl(d / "probes.jsonl") if "metrics" in r]
    s = summarize(recs, cfg)
    old_p = d / "summary.json"
    if old_p.exists():
        old = json.loads(old_p.read_text(encoding="utf-8"))
        s["model_info"] = old.get("model_info")
        s["wall_seconds"] = old.get("wall_seconds")
    else:
        s["model_info"] = json.loads((d / "model_info.json").read_text(encoding="utf-8"))
    old_p.write_text(json.dumps(s, indent=2), encoding="utf-8")
    (d / "summary.md").write_text(to_markdown(s, cfg), encoding="utf-8")
    print("resummarized", d)

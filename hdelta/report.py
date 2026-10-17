"""Turn probe logs into the Phase-3 success measurements."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _first_t(recs: List[dict], key: str, pred) -> Optional[int]:
    for r in recs:
        v = r["metrics"].get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and pred(v):
            return r["t"] - r["phase_start"]
    return None


def _tail_mean(recs: List[dict], key: str, n: int = 3) -> float:
    vals = [r["metrics"][key] for r in recs[-n:] if key in r["metrics"]]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize(records: List[dict], cfg: dict) -> dict:
    theta = float(cfg["eval"].get("compile_threshold", 0.8))
    phases = cfg["stream"]["phases"]
    recs = [r for r in records if "metrics" in r]
    base = next(r for r in recs if r["tag"] == "baseline")
    rules = list(cfg["task"]["rules"])
    out: Dict[str, object] = {
        "threshold": theta,
        "baseline": {k: v for k, v in base["metrics"].items()
                     if k in ("lm_ppl", "cloze_acc") or k.endswith("/zero_direct_acc")
                     or k.endswith("/icl_scratch_acc") or k.endswith("/steps_needed")},
    }
    phase_out = []
    for i, ph in enumerate(phases):
        rule = ph["rule"]
        prs = [r for r in recs if r["phase"] == i and r["tag"] != "baseline"]
        if not prs:
            continue
        prev = None
        for j in range(i - 1, -1, -1):
            if phases[j]["rule"] != rule:
                prev = phases[j]["rule"]
                break
        slow_key = (f"{rule}/instructed_scratch_acc" if f"{rule}/instructed_scratch_acc" in prs[0]["metrics"]
                    else f"{rule}/icl_scratch_acc")
        d: Dict[str, object] = {
            "phase": i, "rule": rule, "prev_rule": prev, "puzzles": ph["puzzles"],
            # cognitive compilation
            "slow_path_metric": slow_key,
            "slow_path_start": prs[0]["metrics"].get(slow_key),
            "slow_path_final": _tail_mean(prs, slow_key),
            "t_slow_path": _first_t(prs, slow_key, lambda v: v >= theta),
            # half-way timings: the ordering of the two paths is often clearer at 0.5
            "t50_slow_path": _first_t(prs, slow_key, lambda v: v >= 0.5),
            "t50_compile": _first_t(prs, f"{rule}/zero_direct_acc", lambda v: v >= 0.5),
            "t_zero_scratch": _first_t(prs, f"{rule}/zero_scratch_acc", lambda v: v >= theta),
            "t_compile": _first_t(prs, f"{rule}/zero_direct_acc", lambda v: v >= theta),
            "final_zero_direct_acc": _tail_mean(prs, f"{rule}/zero_direct_acc"),
            "final_icl_scratch_acc": _tail_mean(prs, f"{rule}/icl_scratch_acc"),
            "final_steps_needed": _tail_mean(prs, f"{rule}/steps_needed", 1),
            "start_steps_needed": prs[0]["metrics"].get(f"{rule}/steps_needed"),
            "final_seen_acc": _tail_mean(prs, "seen/zero_direct_acc"),
            "max_zero_direct_acc": max(r["metrics"][f"{rule}/zero_direct_acc"] for r in prs),
        }
        if d["t_compile"] is not None and d["t_slow_path"] is not None:
            d["compile_lag"] = d["t_compile"] - d["t_slow_path"]
        # open-ended probe: does the model still think out loud, and at what token cost?
        if f"{rule}/free_steps" in prs[0]["metrics"]:
            d.update({
                "free_acc_start": prs[0]["metrics"][f"{rule}/free_acc"],
                "free_acc_final": _tail_mean(prs, f"{rule}/free_acc"),
                "free_steps_start": prs[0]["metrics"][f"{rule}/free_steps"],
                "free_steps_final": _tail_mean(prs, f"{rule}/free_steps"),
                "free_tokens_start": prs[0]["metrics"][f"{rule}/free_tokens"],
                "free_tokens_final": _tail_mean(prs, f"{rule}/free_tokens"),
                "t_instant": _first_t(prs, f"{rule}/free_acc", lambda v: v >= theta),
            })
            if f"{rule}/free_rule_steps" in prs[0]["metrics"]:
                d["free_rule_steps_start"] = prs[0]["metrics"][f"{rule}/free_rule_steps"]
                d["free_rule_steps_final"] = _tail_mean(prs, f"{rule}/free_rule_steps")
                d["free_rule_acc_start"] = prs[0]["metrics"][f"{rule}/free_rule_acc"]
                d["free_rule_acc_final"] = _tail_mean(prs, f"{rule}/free_rule_acc")
        # unhooking: the previous rule's loop must dissolve
        if prev is not None:
            key = f"{rule}/rate_{prev}"
            series = [(r["t"] - r["phase_start"], r["metrics"].get(key, float("nan"))) for r in prs]
            peak_t, peak = max(series, key=lambda x: x[1] if not math.isnan(x[1]) else -1)
            t_dissolve = None
            for t_rel, v in series:
                if t_rel > peak_t and v <= 0.1:
                    t_dissolve = t_rel
                    break
            d.update({
                "oldrule_rate_start": series[0][1],
                "oldrule_rate_peak": peak,
                "t_oldrule_peak": peak_t,
                "t_dissolve": t_dissolve,
                "oldrule_rate_final": _tail_mean(prs, key),
                "prev_rule_acc_start": prs[0]["metrics"].get(f"{prev}/zero_direct_acc"),
                "prev_rule_acc_final": _tail_mean(prs, f"{prev}/zero_direct_acc"),
                "t_unhook": d["t_compile"],
            })
        phase_out.append(d)
    out["phases"] = phase_out

    # base-skill drift (catastrophic forgetting check)
    ppl = [r["metrics"]["lm_ppl"] for r in recs]
    cloze = [r["metrics"]["cloze_acc"] for r in recs]
    out["base_skills"] = {
        "ppl_baseline": ppl[0], "ppl_final": ppl[-1], "ppl_max": max(ppl),
        "ppl_drift_final_pct": 100.0 * (ppl[-1] / ppl[0] - 1.0),
        "ppl_drift_max_pct": 100.0 * (max(ppl) / ppl[0] - 1.0),
        "cloze_baseline": cloze[0], "cloze_final": cloze[-1], "cloze_min": min(cloze),
    }
    # fast-weight health
    groups = list(cfg["plastic"]["groups"])
    out["fast_weights"] = {}
    for g in groups:
        k = f"fw_norm_ratio_group/{g}"
        vals = [r["metrics"][k] for r in recs if k in r["metrics"]]
        nk = f"noise_ratio_group/{g}"
        nvals = [r["metrics"][nk] for r in recs if nk in r["metrics"]]
        out["fast_weights"][g] = {
            "norm_ratio_max": max(vals) if vals else None,
            "norm_ratio_final": vals[-1] if vals else None,
            "noise_ratio_max": max(nvals) if nvals else None,
            "noise_ratio_final": nvals[-1] if nvals else None,
        }
    # REM cost: task accuracy and ppl immediately before vs after each sleep
    costs, ppl_costs, nr_before, nr_after = [], [], [], []
    for idx, r in enumerate(recs):
        if r["tag"] != "post_rem" or idx == 0:
            continue
        before = recs[idx - 1]
        rule = r["rule"]
        costs.append(before["metrics"][f"{rule}/zero_direct_acc"] - r["metrics"][f"{rule}/zero_direct_acc"])
        ppl_costs.append(r["metrics"]["lm_ppl"] - before["metrics"]["lm_ppl"])
        for g in groups:
            nr_before.append(before["metrics"].get(f"noise_ratio_group/{g}", float("nan")))
            nr_after.append(r["metrics"].get(f"noise_ratio_group/{g}", float("nan")))
    out["rem"] = {
        "count": sum(1 for r in recs if r["tag"] == "post_rem"),
        "mean_acc_cost": float(sum(costs) / len(costs)) if costs else None,
        "max_acc_cost": max(costs) if costs else None,
        "mean_ppl_change": float(sum(ppl_costs) / len(ppl_costs)) if ppl_costs else None,
        "noise_ratio_before_mean": float(sum(nr_before) / len(nr_before)) if nr_before else None,
        "noise_ratio_after_mean": float(sum(nr_after) / len(nr_after)) if nr_after else None,
    }
    return out


def _fmt(v) -> str:
    if v is None:
        return "never"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def to_markdown(s: dict, cfg: dict) -> str:
    th = s["threshold"]
    lines = [f"# Run summary", "",
             f"Model `{cfg['model']['name']}`, task `{cfg['task']['name']}`, "
             f"postsynaptic `{cfg['plastic'].get('postsynaptic','error')}`, "
             f"decay {'on' if cfg['plastic'].get('decay_enabled', True) else 'off'}, "
             f"homeostasis {'on' if cfg['plastic'].get('homeo_enabled', True) else 'off'}, "
             f"REM {'on' if cfg['rem'].get('enabled', True) else 'off'}, "
             f"updates {'on' if cfg['plastic'].get('updates_enabled', True) else 'off'}.", ""]
    mi = s.get("model_info")
    if mi:
        lines.append(f"Plastic parameters: {mi['plastic_params']:,} of {mi['total_params']:,} "
                     f"({mi['plastic_fraction']:.2%}); {len(mi['modules'])} h-delta modules.")
        lines.append("")
    lines += ["## Phases (t = puzzles since phase start; threshold = %.2f)" % th, "",
              "| phase | rule | prev | t_slow_path | t_compile | lag | final direct acc | steps needed start->final | old-rule peak | t_dissolve | old-rule final | prev-rule acc final |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in s["phases"]:
        lines.append("| {phase} | {rule} | {prev} | {tslow} | {tc} | {lag} | {fa} | {s0}->{s1} | {op} | {td} | {of} | {pf} |".format(
            phase=p["phase"], rule=p["rule"], prev=p.get("prev_rule") or "-",
            tslow=_fmt(p["t_slow_path"]), tc=_fmt(p["t_compile"]), lag=_fmt(p.get("compile_lag")),
            fa=_fmt(p["final_zero_direct_acc"]), s0=_fmt(p["start_steps_needed"]), s1=_fmt(p["final_steps_needed"]),
            op=_fmt(p.get("oldrule_rate_peak")), td=_fmt(p.get("t_dissolve")) if "t_dissolve" in p else "-",
            of=_fmt(p.get("oldrule_rate_final")), pf=_fmt(p.get("prev_rule_acc_final"))))
    if any("free_steps_start" in p for p in s["phases"]):
        lines += ["", "## Excitation path (open-ended probe: the model chooses whether to think)", "",
                  "| phase | rule | slow path start->final | free acc start->final | Step lines emitted start->final | tokens to answer start->final | t_instant |",
                  "|---|---|---|---|---|---|---|"]
        for p in s["phases"]:
            if "free_steps_start" not in p:
                continue
            lines.append("| {ph} | {r} | {s0}->{s1} | {a0}->{a1} | {st0}->{st1} | {t0}->{t1} | {ti} |".format(
                ph=p["phase"], r=p["rule"], s0=_fmt(p.get("slow_path_start")), s1=_fmt(p.get("slow_path_final")),
                a0=_fmt(p["free_acc_start"]), a1=_fmt(p["free_acc_final"]),
                st0=_fmt(p["free_steps_start"]), st1=_fmt(p["free_steps_final"]),
                t0=_fmt(p["free_tokens_start"]), t1=_fmt(p["free_tokens_final"]), ti=_fmt(p.get("t_instant"))))
    b = s["base_skills"]
    lines += ["", "## Base language skills", "",
              f"- perplexity: baseline {b['ppl_baseline']:.2f}, final {b['ppl_final']:.2f} "
              f"({b['ppl_drift_final_pct']:+.1f}%), worst {b['ppl_max']:.2f} ({b['ppl_drift_max_pct']:+.1f}%)",
              f"- cloze accuracy: baseline {b['cloze_baseline']:.2f}, final {b['cloze_final']:.2f}, min {b['cloze_min']:.2f}",
              "", "## Fast-weight health", ""]
    def _e(x):
        return "n/a" if x is None else f"{x:.2e}"
    for g, v in s["fast_weights"].items():
        # a group whose layer list is empty contributes no modules, so every entry is None
        if v.get("norm_ratio_max") is None:
            lines.append(f"- {g}: no plastic modules in this run")
            continue
        lines.append(f"- {g}: fast-weight ratio max {_fmt(v['norm_ratio_max'])}, final {_fmt(v['norm_ratio_final'])}; "
                     f"noise-response ratio max {_e(v['noise_ratio_max'])}, final {_e(v['noise_ratio_final'])}")
    r = s["rem"]
    lines += ["", "## REM sleep", "",
              f"- phases: {r['count']}; mean accuracy cost {_fmt(r['mean_acc_cost'])}, max {_fmt(r['max_acc_cost'])}; "
              f"mean ppl change {_fmt(r['mean_ppl_change'])}",
              f"- noise-response ratio before/after sleep: {_fmt(r['noise_ratio_before_mean'])} / {_fmt(r['noise_ratio_after_mean'])}",
              ""]
    return "\n".join(lines)

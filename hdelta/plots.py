"""Figures for one run and for an ablation suite (matplotlib, static PNG).

Palette: validated default categorical slots (blue, orange, aqua, yellow, ...),
text in ink tokens, thin 2px lines, recessive grid, direct labels + legend.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    lines = [json.loads(l) for l in (run_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    probes = [r for r in lines if "metrics" in r]
    events = [r for r in lines if "event" in r]
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    return {"probes": probes, "events": events, "cfg": cfg, "summary": summary, "name": run_dir.name}


def _series(probes: List[dict], key: str, tags=("baseline", "stream", "post_rem", "pre_rem", "final")):
    xs, ys = [], []
    for r in probes:
        if r["tag"] in tags and key in r["metrics"]:
            v = r["metrics"][key]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            xs.append(r["t"])
            ys.append(v)
    return xs, ys


def _phase_marks(ax, run: dict, label: bool = True) -> None:
    ymin, ymax = ax.get_ylim()
    for ev in run["events"]:
        if ev["event"] == "phase_start":
            ax.axvline(ev["t"], color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.7)
            if label:
                ax.text(ev["t"] + 3, ymax, f"rule {ev['rule']}", color=INK2, fontsize=8,
                        va="top", ha="left")
        elif ev["event"] == "rem":
            ax.plot([ev["t"]], [ymin], marker="^", color=INK2, markersize=5, clip_on=False, zorder=5)


def _end_label(ax, xs, ys, text, color):
    if xs:
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")


def fig_compilation(run: dict, out: Path) -> None:
    rules = list(run["cfg"]["task"]["rules"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, r in enumerate(rules):
        c = SERIES[i % len(SERIES)]
        xs, ys = _series(run["probes"], f"{r}/zero_direct_acc")
        ax.plot(xs, ys, color=c, label=f"rule {r}: direct, zero-shot (compiled path)")
        _end_label(ax, xs, ys, f"{r} direct", c)
        slow_key = f"{r}/instructed_scratch_acc"
        if not any(slow_key in pr["metrics"] for pr in run["probes"]):
            slow_key = f"{r}/icl_scratch_acc"
        xs, ys = _series(run["probes"], slow_key)
        ax.plot(xs, ys, color=c, linestyle=(0, (4, 2)), linewidth=1.4,
                label=f"rule {r}: scratchpad with {'rule stated' if 'instructed' in slow_key else 'examples'} (slow path)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("puzzles seen")
    ax.set_ylabel("exact-match accuracy")
    fig.suptitle("Cognitive compilation and unhooking", x=0.01, ha="left", color=INK)
    _phase_marks(ax, run)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compilation_unhooking.png", dpi=150)
    plt.close(fig)


def fig_habit(run: dict, out: Path) -> None:
    """During each phase after a switch: do answers follow the new rule or the old one?"""
    phases = run["cfg"]["stream"]["phases"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    plotted = False
    for i, ph in enumerate(phases):
        prev = None
        for j in range(i - 1, -1, -1):
            if phases[j]["rule"] != ph["rule"]:
                prev = phases[j]["rule"]
                break
        if prev is None:
            continue
        recs = [r for r in run["probes"] if r["phase"] == i]
        xs = [r["t"] for r in recs]
        new = [r["metrics"][f"{ph['rule']}/zero_direct_acc"] for r in recs]
        old = [r["metrics"].get(f"{ph['rule']}/rate_{prev}", float('nan')) for r in recs]
        ax.plot(xs, new, color=SERIES[0], label="answer follows the NEW rule" if not plotted else None)
        ax.plot(xs, old, color=SERIES[1], label="answer follows the OLD rule (habit)" if not plotted else None)
        _end_label(ax, xs, new, f"new ({ph['rule']})", SERIES[0])
        _end_label(ax, xs, old, f"old ({prev})", SERIES[1])
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no rule switch in this run", ha="center", transform=ax.transAxes, color=INK2)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("puzzles seen")
    ax.set_ylabel("fraction of probe answers")
    fig.suptitle("Habit persistence after the rules change", x=0.01, ha="left", color=INK)
    _phase_marks(ax, run)
    if plotted:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "habit_persistence.png", dpi=150)
    plt.close(fig)


def fig_depth(run: dict, out: Path) -> None:
    rules = list(run["cfg"]["task"]["rules"])
    if not any(f"{rules[0]}/steps_needed" in pr["metrics"] for pr in run["probes"]):
        return                                  # compilation depth was not probed
    K = max(len(v) for v in run["cfg"]["task"]["rules"].values()) if run["cfg"]["task"]["name"] == "cipher" else run["cfg"]["task"].get("max_hops", 3)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, r in enumerate(rules):
        xs, ys = _series(run["probes"], f"{r}/steps_needed")
        ax.step(xs, ys, where="post", color=SERIES[i % len(SERIES)], label=f"rule {r}")
        _end_label(ax, xs, ys, r, SERIES[i])
    ax.set_yticks(list(range(K + 2)))
    ax.set_yticklabels(["0 = compiled"] + [str(k) for k in range(1, K + 1)] + ["unsolved"])
    ax.set_ylim(-0.3, K + 1.3)
    ax.set_xlabel("puzzles seen")
    ax.set_ylabel("scratchpad steps needed")
    fig.suptitle("Compilation depth: how much thinking-out-loud the answer still needs", x=0.01, ha="left", color=INK)
    _phase_marks(ax, run)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=len(rules), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compilation_depth.png", dpi=150)
    plt.close(fig)


def fig_base_skills(run: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    xs, ys = _series(run["probes"], "lm_ppl")
    base = ys[0] if ys else 1.0
    axes[0].plot(xs, [y / base for y in ys], color=SERIES[0])
    axes[0].axhline(1.0, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    axes[0].set_ylabel("perplexity / baseline")
    axes[0].set_title(f"Language-model perplexity (baseline {base:.2f})", loc="left", fontsize=10)
    if ys and max(ys) / base > 5:
        axes[0].set_yscale("log")
    xs, ys = _series(run["probes"], "cloze_acc")
    axes[1].plot(xs, ys, color=SERIES[0])
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel("cloze accuracy")
    axes[1].set_title("Factual cloze accuracy", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("puzzles seen")
        _phase_marks(ax, run, label=False)
    fig.suptitle("Base language skills (catastrophic-forgetting check)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "base_skills.png", dpi=150)
    plt.close(fig)


def fig_health(run: dict, out: Path) -> None:
    groups = list(run["cfg"]["plastic"]["groups"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, g in enumerate(groups):
        c = SERIES[i % len(SERIES)]
        xs, ys = _series(run["probes"], f"fw_norm_ratio_group/{g}")
        axes[0].plot(xs, ys, color=c, label=g)
        xs, ys = _series(run["probes"], f"noise_ratio_group/{g}")
        axes[1].plot(xs, [max(y, 1e-9) for y in ys], color=c, label=g)
    budget = max(float(v.get("budget", 0)) for v in run["cfg"]["plastic"]["groups"].values())
    if run["cfg"]["plastic"].get("homeo_enabled", True):
        axes[0].axhline(budget, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        axes[0].text(0, budget, " energy budget", color=INK2, fontsize=8, va="bottom")
    axes[0].set_ylabel("||dW|| / ||W||")
    axes[0].set_title("Fast-weight energy", loc="left", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=1e-6)
    axes[1].set_ylabel("||dW x||^2 / ||W x||^2 under noise")
    axes[1].set_title("Noise response (rigidity; triangles = REM sleep)", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("puzzles seen")
        _phase_marks(ax, run, label=False)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Fast-weight health", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fast_weight_health.png", dpi=150)
    plt.close(fig)


def fig_thinking(run: dict, out: Path) -> None:
    """Open-ended probe: accuracy, Step lines emitted, tokens spent before the answer."""
    rules = list(run["cfg"]["task"]["rules"])
    if not any(f"{rules[0]}/free_steps" in pr["metrics"] for pr in run["probes"]):
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, r in enumerate(rules):
        c = SERIES[i % len(SERIES)]
        for ax, key in zip(axes, ("free_acc", "free_steps", "free_tokens")):
            xs, ys = _series(run["probes"], f"{r}/{key}")
            ax.plot(xs, ys, color=c, label=f"rule {r}")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_title("Accuracy, open-ended (no rule text)", loc="left", fontsize=10)
    axes[1].set_title("Step lines emitted before answering", loc="left", fontsize=10)
    axes[2].set_title("Tokens spent to reach the answer", loc="left", fontsize=10)
    for ax in axes:
        ax.set_xlabel("puzzles seen")
        _phase_marks(ax, run, label=False)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Does the model still think out loud?  (excitation cost of the answer)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "thinking.png", dpi=150)
    plt.close(fig)


def plot_run(run_dir: Path) -> None:
    run = load_run(run_dir)
    out = Path(run_dir)
    fig_thinking(run, out)
    fig_compilation(run, out)
    fig_habit(run, out)
    fig_depth(run, out)
    fig_base_skills(run, out)
    fig_health(run, out)


def fig_suite(run_dirs: List[Path], out: Path, metric_rules: Optional[List[str]] = None) -> None:
    runs = [load_run(d) for d in run_dirs if (Path(d) / "probes.jsonl").exists()]
    if not runs:
        return
    n = len(runs)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), sharex=True, sharey=True, squeeze=False)
    for k, run in enumerate(runs):
        ax = axes[k // cols][k % cols]
        rules = metric_rules or list(run["cfg"]["task"]["rules"])
        for i, r in enumerate(rules):
            xs, ys = _series(run["probes"], f"{r}/zero_direct_acc")
            ax.plot(xs, ys, color=SERIES[i % len(SERIES)], label=f"rule {r} direct")
        xs, ys = _series(run["probes"], "lm_ppl")
        b = run["summary"].get("base_skills", {})
        drift = b.get("ppl_drift_max_pct")
        sub = f"ppl drift max {drift:+.0f}%" if drift is not None else ""
        ax.set_title(f"{run['name']}   {sub}", loc="left", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        _phase_marks(ax, run, label=False)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    axes[0][0].legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("puzzles seen")
    for row in axes:
        row[0].set_ylabel("direct accuracy")
    fig.suptitle("Ablations: zero-shot direct accuracy per rule", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "ablation_compare.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        plot_run(Path(d))
        print("plotted", d)

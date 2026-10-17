"""Capacity analysis: how many skills can one global h-delta state hold at once?

Reads a run whose stream interleaves N tagged rules in short blocks and answers:

  simultaneous mastery   how many rules are above threshold at the same probe, over time
                         (the headline: the capacity of one global fast-weight state)
  retention              compiled accuracy as a function of how many puzzles ago that
                         rule was last trained, pooled over rules; the slope is the
                         interference rate in accuracy lost per 100 puzzles away.
                         Recency is binned in puzzles, not blocks, because the
                         matched-cycle ladder uses a different block length per N
  confusion              at the end, for each rule, how often the answer follows a
                         *different* rule's mapping (the rules never agree on any digit,
                         so this is unambiguous)
  deliberate vs compiled both paths per rule, so we can see whether interference destroys
                         the reflex only, or also the ability to follow the stated rule

Usage:  python -m hdelta.capacity results/capacity/n8 [...]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

THETA = 0.8
RECENCY_BIN = 10     # puzzles per retention bin; common across the ladder so the curves
                     # from different N are directly comparable


def load(run_dir: Path):
    run_dir = Path(run_dir)
    lines = [json.loads(l) for l in (run_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    probes = [r for r in lines if "metrics" in r]
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return probes, cfg


def training_timeline(cfg: dict):
    """For each rule, the (start, end) puzzle index of every block it was trained in."""
    blocks: Dict[str, List[tuple]] = {}
    t = 0
    for ph in cfg["stream"]["phases"]:
        r, n = ph["rule"], int(ph["puzzles"])
        blocks.setdefault(r, []).append((t, t + n))
        t += n
    return blocks, t


def analyse(run_dir: Path, theta: float = THETA) -> dict:
    probes, cfg = load(run_dir)
    rules = list(cfg["task"]["rules"])
    blocks, total = training_timeline(cfg)
    block_len = int(cfg["stream"]["phases"][0]["puzzles"])
    recs = [r for r in probes if r["tag"] in ("baseline", "stream", "pre_rem", "post_rem", "final")]
    recs.sort(key=lambda r: (r["t"], r["tag"] != "post_rem"))
    t_axis = [r["t"] for r in recs]

    acc = {r: [rec["metrics"].get(f"{r}/zero_direct_acc", float("nan")) for rec in recs] for r in rules}
    slow_key = "instructed_scratch_acc" if f"{rules[0]}/instructed_scratch_acc" in recs[0]["metrics"] else "icl_scratch_acc"
    slow = {r: [rec["metrics"].get(f"{r}/{slow_key}", float("nan")) for rec in recs] for r in rules}

    # trained-yet, and how long since the last training block ended
    since: Dict[str, List[Optional[float]]] = {r: [] for r in rules}
    for r in rules:
        for t in t_axis:
            ends = [e for (s, e) in blocks[r] if e <= t]
            since[r].append(None if not ends else t - max(ends))

    mastered = [int(sum(1 for r in rules if acc[r][k] >= theta and since[r][k] is not None))
                for k in range(len(recs))]
    trained = [int(sum(1 for r in rules if since[r][k] is not None)) for k in range(len(recs))]

    # retention: accuracy vs blocks since that rule was last trained (only rules already trained)
    ret: Dict[int, List[float]] = {}
    ret_slow: Dict[int, List[float]] = {}
    for r in rules:
        for k in range(len(recs)):
            s = since[r][k]
            if s is None:
                continue
            b = int(round(s / RECENCY_BIN)) * RECENCY_BIN      # puzzles since last trained
            ret.setdefault(b, []).append(acc[r][k])
            ret_slow.setdefault(b, []).append(slow[r][k])
    retention = {b: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                     "n": len(v)} for b, v in sorted(ret.items())}
    retention_slow = {b: {"mean": float(np.nanmean(v)), "n": len(v)} for b, v in sorted(ret_slow.items())}

    # interference slope over the first few recency bins that have real support
    xs = [b for b in sorted(retention) if retention[b]["n"] >= 3]
    slope = None
    if len(xs) >= 3:
        ys = [retention[b]["mean"] for b in xs]
        slope = float(np.polyfit(xs, ys, 1)[0]) * 100.0      # accuracy per 100 puzzles away

    # --- recency dominance -----------------------------------------------------------
    # The signature of a single-slot store: at every probe, is the rule that was trained
    # most recently also the one with the strongest reflex?  Under a multiplexed store
    # (each skill in its own addressable slot) the most-recent rule would rank at chance.
    dom_hits, dom_ranks, totals, tops = 0, [], [], 0
    for k in range(len(recs)):
        live = [r for r in rules if since[r][k] is not None]
        if len(live) < 2:
            continue
        recent = min(live, key=lambda r: since[r][k])
        order = sorted(live, key=lambda r: -acc[r][k])
        rank = order.index(recent) + 1
        dom_ranks.append(rank)
        dom_hits += int(rank == 1)
        tops += 1
        totals.append(float(sum(acc[r][k] for r in live)))
    n_r = len(rules)
    p_chance = 1.0 / n_r
    p_value = None
    if tops:
        p_value = float(sum(math.comb(tops, i) * p_chance ** i * (1 - p_chance) ** (tops - i)
                            for i in range(dom_hits, tops + 1)))
    # do the reflexes of different rules trade off against each other?
    pair_corr = []
    live_from = next((k for k in range(len(recs)) if all(since[r][k] is not None for r in rules)), None)
    if live_from is not None and len(recs) - live_from >= 4:
        M = np.array([[acc[r][k] for k in range(live_from, len(recs))] for r in rules])
        S = np.array([[slow[r][k] for k in range(live_from, len(recs))] for r in rules])
        for i in range(n_r):
            for j in range(i + 1, n_r):
                if np.std(M[i]) > 1e-9 and np.std(M[j]) > 1e-9:
                    pair_corr.append(float(np.corrcoef(M[i], M[j])[0, 1]))
        slow_corr = []
        for i in range(n_r):
            for j in range(i + 1, n_r):
                if np.std(S[i]) > 1e-9 and np.std(S[j]) > 1e-9:
                    slow_corr.append(float(np.corrcoef(S[i], S[j])[0, 1]))
    else:
        slow_corr = []
    dominance = {
        "probes_scored": tops,
        "most_recent_ranked_first": dom_hits,
        "share": (dom_hits / tops) if tops else float("nan"),
        "chance": p_chance,
        "p_value": p_value,
        "mean_rank_of_most_recent": float(np.mean(dom_ranks)) if dom_ranks else float("nan"),
        "total_accuracy_mean": float(np.mean(totals)) if totals else float("nan"),
        "total_accuracy_last": totals[-1] if totals else float("nan"),
        "mean_pairwise_corr_compiled": float(np.mean(pair_corr)) if pair_corr else float("nan"),
        "mean_pairwise_corr_deliberate": float(np.mean(slow_corr)) if slow_corr else float("nan"),
    }

    # final confusion: rule asked -> rule answered
    last = recs[-1]
    confusion = {r: {s: last["metrics"].get(f"{r}/rate_{s}", 0.0) for s in rules if s != r} for r in rules}
    final = {r: {"compiled": acc[r][-1], "deliberate": slow[r][-1],
                 "best": float(np.nanmax(acc[r])),
                 "confusion_total": float(sum(confusion[r].values()))} for r in rules}

    out = {
        "run": Path(run_dir).name, "n_rules": len(rules), "rules": rules, "theta": theta,
        "block_len": block_len, "total_puzzles": total, "t": t_axis,
        "acc": acc, "slow": slow, "slow_key": slow_key, "since": since,
        "mastered": mastered, "trained": trained,
        "max_mastered": int(max(mastered)) if mastered else 0,
        "final_mastered": int(mastered[-1]) if mastered else 0,
        "mean_mastered_last_quarter": float(np.mean(mastered[-max(1, len(mastered) // 4):])),
        "retention": retention, "retention_slow": retention_slow, "interference_slope": slope,
        "dominance": dominance,
        "confusion": confusion, "final": final,
        "lm_ppl": [r["metrics"]["lm_ppl"] for r in recs],
        "cloze": [r["metrics"]["cloze_acc"] for r in recs],
    }
    (Path(run_dir) / "capacity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (Path(run_dir) / "capacity.md").write_text(to_markdown(out), encoding="utf-8")
    try:
        plot(out, Path(run_dir) / "capacity.png")
    except Exception as e:
        out["plot_error"] = repr(e)
    return out


def to_markdown(o: dict) -> str:
    dom = o["dominance"]
    L = [f"# Capacity: {o['run']} ({o['n_rules']} interleaved rules, blocks of {o['block_len']}, "
         f"{o['total_puzzles']} puzzles)", "",
         f"Mastery threshold {o['theta']:.2f} on the compiled path (input + tag only, greedy).", "",
         f"- **rules mastered at the same time: peak {o['max_mastered']} of {o['n_rules']}, "
         f"final {o['final_mastered']}, mean over the last quarter {o['mean_mastered_last_quarter']:.1f}**",
         f"- **the most recently trained rule has the strongest reflex at "
         f"{dom['most_recent_ranked_first']}/{dom['probes_scored']} probes "
         f"({dom['share']:.0%}; chance {dom['chance']:.0%}, p = "
         + ("n/a" if dom["p_value"] is None else f"{dom['p_value']:.2g}") + ")**",
         f"- mean rank of the most recent rule: {dom['mean_rank_of_most_recent']:.2f} of {o['n_rules']}; "
         f"summed compiled accuracy across all rules: mean {dom['total_accuracy_mean']:.2f}, "
         f"last {dom['total_accuracy_last']:.2f} (of {o['n_rules']:.2f} possible)",
         f"- mean pairwise correlation between rules over time: compiled "
         f"{dom['mean_pairwise_corr_compiled']:+.2f}, deliberate {dom['mean_pairwise_corr_deliberate']:+.2f} "
         f"(negative on the compiled path = the reflexes trade off against each other)",
         f"- interference slope: "
         + ("n/a" if o["interference_slope"] is None else f"{o['interference_slope']:+.3f} accuracy per 100 puzzles since that rule was last trained"),
         "", "## Per rule at the end of the run", "",
         "| rule | compiled | deliberate | best compiled ever | answers following another rule |",
         "|---|---|---|---|---|"]
    for r in o["rules"]:
        f = o["final"][r]
        L.append(f"| {r} | {f['compiled']:.2f} | {f['deliberate']:.2f} | {f['best']:.2f} | {f['confusion_total']:.2f} |")
    L += ["", "## Retention: compiled accuracy by how long ago that rule was last trained", "",
          "| puzzles since trained | compiled mean | deliberate mean | observations |", "|---|---|---|---|"]
    for b, v in o["retention"].items():
        s = o["retention_slow"].get(b, {})
        L.append(f"| {b} | {v['mean']:.2f} ± {v['sd']:.2f} | {s.get('mean', float('nan')):.2f} | {v['n']} |")
    L += ["", "## Confusion at the end (row = rule asked, value = share answering that rule's mapping)", "",
          "| asked | " + " | ".join(o["rules"]) + " |", "|---" * (len(o["rules"]) + 1) + "|"]
    for r in o["rules"]:
        cells = []
        for s in o["rules"]:
            cells.append(f"**{o['final'][r]['compiled']:.2f}**" if s == r else f"{o['confusion'][r].get(s, 0.0):.2f}")
        L.append(f"| {r} | " + " | ".join(cells) + " |")
    L += ["", f"Base skills: perplexity {o['lm_ppl'][0]:.2f} -> {o['lm_ppl'][-1]:.2f} "
          f"({100 * (o['lm_ppl'][-1] / o['lm_ppl'][0] - 1):+.1f} %), cloze {o['cloze'][0]:.2f} -> {o['cloze'][-1]:.2f}.", ""]
    return "\n".join(L)


def plot(o: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plots import INK, INK2, SERIES, SURFACE  # noqa: F401

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    t = o["t"]
    ax = axes[0][0]
    for i, r in enumerate(o["rules"]):
        ax.plot(t, o["acc"][r], color=SERIES[i % len(SERIES)], label=r, linewidth=1.6)
    ax.axhline(o["theta"], color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("compiled accuracy")
    ax.set_title("each rule's reflex over the interleaved stream", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax = axes[0][1]
    ax.plot(t, o["trained"], color=INK2, linewidth=1.2, linestyle=(0, (3, 3)), label="rules seen at least once")
    ax.plot(t, o["mastered"], color=SERIES[2], linewidth=2.2, label=f"rules at or above {o['theta']:.1f}")
    ax.set_ylim(-0.2, o["n_rules"] + 0.4)
    ax.set_yticks(range(o["n_rules"] + 1))
    ax.set_ylabel("number of rules")
    ax.set_title("how many skills are held at the same time", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1][0]
    bs = sorted(o["retention"])
    mean = [o["retention"][b]["mean"] for b in bs]
    sd = [o["retention"][b]["sd"] for b in bs]
    ax.errorbar(bs, mean, yerr=sd, color=SERIES[0], marker="o", markersize=4, capsize=3, label="compiled")
    sm = [o["retention_slow"].get(b, {}).get("mean", float("nan")) for b in bs]
    ax.plot(bs, sm, color=SERIES[1], marker="s", markersize=4, linestyle=(0, (4, 2)), label="deliberate")
    ax.axhline(o["theta"], color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("puzzles since that rule was last trained")
    ax.set_ylabel("accuracy")
    ax.set_title("retention: interference as a function of recency", loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1][1]
    n = o["n_rules"]
    M = np.zeros((n, n))
    for i, r in enumerate(o["rules"]):
        for j, s in enumerate(o["rules"]):
            M[i, j] = o["final"][r]["compiled"] if r == s else o["confusion"][r].get(s, 0.0)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(o["rules"], fontsize=7, rotation=45)
    ax.set_yticks(range(n)); ax.set_yticklabels(o["rules"], fontsize=7)
    ax.set_xlabel("answer follows this rule")
    ax.set_ylabel("rule asked")
    ax.set_title("confusion at the end of the run", loc="left", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for a in (axes[0][0], axes[0][1]):
        a.set_xlabel("puzzles seen")
    fig.suptitle(f"Capacity of one global h-delta state: {o['run']}, {n} interleaved rules",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare(run_dirs: List[Path], out_path: Path) -> str:
    rows = ["| run | rules | block | peak held at once | final held | mean over last quarter | most-recent rule ranked first | mean rank of most recent | summed accuracy (of N) | pairwise corr, compiled / deliberate | interference slope (acc / 100 puzzles) | mean confusion | perplexity drift | cloze final |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in run_dirs:
        p = Path(d) / "capacity.json"
        if not p.exists():
            rows.append(f"| {Path(d).name} | (no analysis) |" + " |" * 12)
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        conf = float(np.mean([o["final"][r]["confusion_total"] for r in o["rules"]]))
        dm = o["dominance"]
        rows.append("| {run} | {n} | {bl} | {peak} | {fin} | {mq:.1f} | {dh}/{dt} ({ds:.0%}, chance {dc:.0%}, p={dp}) | {mr:.2f} | {ta:.2f} | {cc:+.2f} / {sc:+.2f} | {sl} | {cf:.2f} | {ppl:+.1f} % | {cz:.2f} |".format(
            run=o["run"], n=o["n_rules"], bl=o["block_len"], peak=o["max_mastered"], fin=o["final_mastered"],
            mq=o["mean_mastered_last_quarter"],
            dh=dm["most_recent_ranked_first"], dt=dm["probes_scored"], ds=dm["share"], dc=dm["chance"],
            dp=("n/a" if dm["p_value"] is None else f"{dm['p_value']:.1g}"),
            mr=dm["mean_rank_of_most_recent"], ta=dm["total_accuracy_last"],
            cc=dm["mean_pairwise_corr_compiled"], sc=dm["mean_pairwise_corr_deliberate"],
            sl="n/a" if o["interference_slope"] is None else f"{o['interference_slope']:+.3f}",
            cf=conf, ppl=100 * (o["lm_ppl"][-1] / o["lm_ppl"][0] - 1), cz=o["cloze"][-1]))
    table = "\n".join(rows)
    Path(out_path).write_text("# Capacity ladder\n\n" + table + "\n", encoding="utf-8")
    return table


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        o = analyse(Path(d))
        print(f"{d}: peak {o['max_mastered']}/{o['n_rules']} held at once, final {o['final_mastered']}")

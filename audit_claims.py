"""Cross-check every load-bearing claim against every run on disk.

Two days of work produced a lot of claims and a lot of runs, and the claims were mostly formed
before most of the runs existed. This re-tests each one against the full dataset and reports
which are SUPPORTED, which are THIN (true of the data but n too small to defend), and which
are CONTRADICTED.

    python audit_claims.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from hdelta.tasks import LogicTask

RESULTS = Path("results")


def load(d: Path):
    if not ((d / "probes.jsonl").exists() and (d / "config.json").exists()):
        return None
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in (d / "probes.jsonl").open(encoding="utf-8")
            if '"metrics"' in l]
    if not recs:
        return None
    return cfg, recs, (d / "summary.json").exists()


def all_runs():
    out = []
    for root in sorted(RESULTS.iterdir()):
        if not root.is_dir():
            continue
        for d in [root] + sorted(x for x in root.iterdir() if x.is_dir()):
            r = load(d)
            if r:
                out.append((d, *r))
    return out


def verdict(label, ok, n, detail):
    tag = "SUPPORTED" if ok and n >= 3 else ("THIN" if ok else "CONTRADICTED")
    print(f"  [{tag:12s}] {label}")
    print(f"                 n={n}  {detail}")


def main():
    runs = all_runs()
    done = [(d, c, r) for d, c, r, fin in runs if fin]
    print(f"{len(runs)} run directories, {len(done)} completed\n")

    # ---- claim 1: student can exceed its teacher -------------------------------
    print("CLAIM: verified self-training denoises - the reflex can beat its own teacher")
    teach = json.loads((RESULTS / "families/teacher.json").read_text(encoding="utf-8")) \
        if (RESULTS / "families/teacher.json").exists() else {}
    beats = []
    for d, cfg, recs in done:
        if cfg["task"]["name"] != "logic":
            continue
        fam = next(iter(cfg["task"]["rules"]))
        spec = cfg["task"]["rules"][fam]
        if spec.get("family") not in teach:
            continue
        base = LogicTask({fam: spec}).balance(fam)["majority_share"]
        acc = [r["metrics"].get(f"{fam}/zero_direct_acc") for r in recs]
        acc = [a for a in acc if a is not None]
        peak = max(acc)
        t = teach[spec["family"]]["taught_mean"]
        beats.append((spec["family"], peak, t, peak - t))
    # peak-vs-mean is biased upward: the peak of a noisy series will beat a mean by chance.
    # Compare the reflex's SECOND-HALF MEAN against the teacher's mean - like against like.
    fair = []
    for d, cfg, recs in done:
        if cfg["task"]["name"] != "logic":
            continue
        fam = next(iter(cfg["task"]["rules"]))
        spec = cfg["task"]["rules"][fam]
        if spec.get("family") not in teach:
            continue
        acc = [r["metrics"].get(f"{fam}/zero_direct_acc") for r in recs]
        acc = [a for a in acc if a is not None]
        late = st.mean(acc[len(acc) // 2:])
        t = teach[spec["family"]]["taught_mean"]
        fair.append((spec["family"], late, t))
    up = [x for x in fair if x[1] > x[2] + 0.02]
    verdict("reflex 2nd-half mean > teacher mean", len(up) > 0, len(fair),
            f"beat teacher in {len(up)}/{len(fair)} runs: "
            + ", ".join(f"{f}({p:.2f}v{t:.2f})" for f, p, t in sorted(set(up))[:5]))

    # ---- claim 2: MLP locus costs more perplexity ------------------------------
    print("\nCLAIM: MLP plasticity costs more base-model drift than attention")
    # ONLY the matched locus pairs: same task, same family, same eta/decay, differing in
    # locus alone. Pooling every run that happens to contain an MLP group sweeps in the old
    # cipher ablations - including a deliberately destructive 4x-energy run - and compares
    # two different experiments as if they were one contrast.
    pairs = {}
    for d, cfg, recs in done:
        if cfg["task"]["name"] != "logic":
            continue
        g = cfg["plastic"]["groups"]
        if len(g) != 1:
            continue
        only = next(iter(g.values()))
        loc = "mlp" if "mlp" in only["patterns"][0] else "attn"
        fam = cfg["task"]["rules"][next(iter(cfg["task"]["rules"]))].get("family")
        sm = json.loads((d / "summary.json").read_text(encoding="utf-8"))["base_skills"]
        pairs.setdefault(fam, {}).setdefault(loc, []).append(abs(sm["ppl_drift_final_pct"]))
    both = {f: v for f, v in pairs.items() if len(v) == 2}
    if both:
        deltas = [(f, st.mean(v["attn"]), st.mean(v["mlp"])) for f, v in both.items()]
        ok = all(m > a for _, a, m in deltas)
        verdict("|ppl drift| mlp > attn, matched family+locus", ok, len(deltas),
                "; ".join(f"{f}: attn {a:.2f}% vs mlp {m:.2f}%" for f, a, m in deltas))
    else:
        print("  [no matched locus pairs found]")

    # ---- claim 3: probe quantisation inflates the fast end ----------------------
    print("\nCLAIM: the fast families' W values are probe artifacts, not measurements")
    fast = {}
    for d, cfg, recs in done:
        if cfg["task"]["name"] != "logic":
            continue
        fam = next(iter(cfg["task"]["rules"]))
        spec = cfg["task"]["rules"][fam]
        f = spec.get("family")
        if f not in ("map", "map_wide", "negation", "disjunction"):
            continue
        pe = cfg["stream"].get("probe_every", 50)
        base = LogicTask({fam: spec}).balance(fam)["majority_share"]
        for r in recs:
            a = r["metrics"].get(f"{fam}/zero_direct_acc")
            if a is not None and (a - base) / (1 - base) >= 0.8:
                fast.setdefault(f, []).append((r["metrics"].get("verifier/writes", r["t"]), pe))
                break
    uniq = {f: v[0][0] for f, v in fast.items()}
    verdict("fast families cluster at probe boundaries", len(set(uniq.values())) < len(uniq),
            len(uniq), f"first-crossing writes: {uniq} (probe_every=50 -> ~50-write grid)")

    # ---- claim 4: contamination stayed at zero throughout -----------------------
    print("\nCLAIM: the honest verifier never wrote a bad example")
    # Filter by verifier_mode. Pooling every run makes this claim meaningless as soon as
    # deliberately-poisoned runs land: `systematic`/`corrupt` runs are SUPPOSED to
    # contaminate, so they either flip an honest-verifier claim to CONTRADICTED for a
    # reason that has nothing to do with honest verifiers, or - worse - report ~0 and let
    # the audit certify a poisoned run as clean. Contamination undercounts `corrupt`
    # badly: bad_writes counts only falsely ACCEPTED answers, while corrupt's dominant
    # channel is reject-then-teach-the-wrong-label, which books as a reject. Never read
    # contamination as an all-clear on a corrupt run; use analyse_poison.py.
    cont, poisoned = [], []
    for d, cfg, recs in done:
        c = recs[-1]["metrics"].get("verifier/contamination")
        if c is None:
            continue
        mode = cfg.get("stream", {}).get("verifier_mode", "honest")
        (cont if mode == "honest" else poisoned).append((d.name, mode, c))
    bad = [x for x in cont if x[2] > 0]
    verdict("contamination == 0 in every HONEST run", not bad, len(cont),
            "all zero" if not bad else f"NONZERO in {bad}")
    if poisoned:
        print("  [INFO        ] non-honest verifiers, excluded above "
              "(contamination is a FLOOR here, not a measurement)")
        for n, m, c in poisoned:
            print(f"                 {n:24s} mode={m:11s} contamination={c:.3f}")

    # ---- claim 5: zero-shot floor really is a floor ------------------------------
    print("\nCLAIM: baseline probes sit at the prior - nothing is guessable from the surface")
    firsts = []
    for d, cfg, recs in done:
        if cfg["task"]["name"] != "logic":
            continue
        fam = next(iter(cfg["task"]["rules"]))
        spec = cfg["task"]["rules"][fam]
        base = LogicTask({fam: spec}).balance(fam)["majority_share"]
        a0 = recs[0]["metrics"].get(f"{fam}/zero_direct_acc")
        if a0 is not None:
            firsts.append((spec.get("family"), a0, base))
    leaks = [x for x in firsts if x[1] > x[2] + 0.1]
    verdict("t=0 accuracy <= prior + 0.1", not leaks, len(firsts),
            f"max t=0 accuracy {max((x[1] for x in firsts), default=0):.3f}"
            + ("" if not leaks else f"  LEAKS: {leaks}"))


if __name__ == "__main__":
    main()

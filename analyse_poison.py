"""Split the poison probe sets into CLEAN and VICTIM inputs.

`map/zero_direct_acc` pools all 64 held-out probes.  The verifier's failure region is a
deterministic sha1 function of the input, so a fixed share of those 64 probes sit INSIDE
the region the verifier mislabels.  Being wrong there is the manipulation working, not
damage, and the pooled number cannot separate the two outcomes that matter:

    graceful    clean 1.00, poison confined to its own slots   -> pooled ~0.73
    cannibal    clean 0.73, poison irrelevant                  -> pooled ~0.73

For `poison_corrupt_03` the pooled metric is capped at 47/64 = 0.734 even when nothing at
all has gone wrong.  Read pooled, that looks like a 27 % capacity loss.

Both the probe set and the mask are reproducible on CPU, and probes.jsonl already stores
per-puzzle correctness in probe order, so this needs no GPU and no re-run.

    .venv/Scripts/python.exe analyse_poison.py [config_name ...]

CAVEAT on the other number in these runs: `contamination` (bad_writes / writes) counts
only falsely ACCEPTED answers.  In `corrupt` mode the dominant channel is the other one -
the model answers correctly, the corrupted verifier rejects it, and `on_wrong: teach`
then writes the verifier's own wrong label.  That is a real bad write booked as a reject,
so contamination understates corrupt runs, most severely early on.  Quote the clean-vs-
victim accuracy below, not contamination, for these four runs.
"""
import json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from hdelta.tasks import make_task
from hdelta.toolcache import ToolVerifier

ROOT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
DEFAULT = ["poison_honest_00", "poison_systematic_03",
           "poison_corrupt_01", "poison_corrupt_03"]


def probe_sets(cfg):
    """Mirror Experiment.__init__: rng = Random(10000 + 17*i + seed), i = rule index."""
    task = make_task(cfg["task"])
    seed = int(cfg.get("seed", 0))
    n = int(cfg["stream"].get("probe_n", 48))
    out = {}
    for i, r in enumerate(cfg["task"]["rules"]):
        rng = random.Random(10_000 + 17 * i + seed)
        out[r] = [task.sample(r, rng) for _ in range(n)]
    return task, out, seed


def victim_mask(v, rule, pz):
    """Which probe inputs fall in this verifier's failure region.

    corrupt     the verifier's own truth is wrong there, so it rejects the right answer
                and teaches its wrong one.  Fires forever.
    systematic  the verifier accepts the model's wrong answer there.  Only fires while
                the model still makes errors, so it goes dormant once a skill compiles.
    """
    if v.mode == "corrupt":
        return [v._corrupted(rule, p.input_str) is not None for p in pz]
    if v.mode == "systematic":
        return [v._fools_verifier(p.input_str) for p in pz]
    return [False] * len(pz)


def _mean(a):
    return (sum(a) / len(a)) if a else float("nan")


def main(names):
    for name in names:
        with open(f"{ROOT}/configs/{name}.yaml") as f:
            cfg = yaml.safe_load(f)
        task, sets, seed = probe_sets(cfg)
        rule = next(iter(sets))
        pz = sets[rule]
        sc = cfg["stream"]
        v = ToolVerifier(task, mode=sc.get("verifier_mode", "honest"),
                         fpr=float(sc.get("verifier_fpr", 0.0)), seed=seed)
        mask = victim_mask(v, rule, pz)
        nv = sum(mask)
        print(f"\n=== {name}   mode={v.mode} fpr={v.fpr} ===")
        # Only `corrupt` puts a ceiling on the pooled number: it makes the verifier's
        # truth wrong there, so those probes stay wrong however well the run goes.
        # `systematic` never corrupts the truth - its victims are learnable, and a
        # healthy systematic run should still reach 1.000 pooled.
        ceil = (f"   pooled ceiling if perfectly graceful = {(len(pz) - nv) / len(pz):.3f}"
                if v.mode == "corrupt" else "   (victims still learnable: no pooled ceiling)")
        print(f"  probe n={len(pz)}  victim={nv}  clean={len(pz) - nv}{ceil}")
        jl = f"{ROOT}/results/poison/{name}/probes.jsonl"
        if not os.path.exists(jl):
            print("  (no probes.jsonl yet - run not finished)")
            continue
        print(f"  {'t':>6} {'pooled':>7} {'CLEAN':>7} {'victim':>7}   contam")
        with open(jl) as f:
            for line in f:
                rec = json.loads(line)
                d = rec.get("per_puzzle", {}).get(rule, {}).get("direct")
                if not d or len(d) != len(mask):
                    continue
                cl = [x for x, b in zip(d, mask) if not b]
                vi = [x for x, b in zip(d, mask) if b]
                mt = rec.get("metrics", {})
                ct = mt.get("verifier/contamination")
                bw = mt.get("verifier/bad_writes")
                rj = mt.get("verifier/rejects")
                ct = (f"{ct:.3f} (bad {bw:.0f}/rej {rj:.0f})"
                      if isinstance(ct, (int, float)) and bw is not None else "-")
                print(f"  {rec.get('t', -1):>6} {_mean(d):>7.3f} {_mean(cl):>7.3f} "
                      f"{_mean(vi):>7.3f}   {ct}")
        print("  CLEAN is the decision: holds at ~1.00 -> lossy but usable;"
              " falls -> the correlated error is eating the valid rule.")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)

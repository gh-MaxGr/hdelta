"""A writability ladder: the same surface task under different logical structures.

Part D found an associative map compiles into the plastic slice (1.00) while a conditional
with precedence does not (0.0000), at matched verified writes in the same run - roughly 50x
harder per update. That is two points on an axis nobody has mapped. This task family fills in
the axis.

Every family shares:
  * the same input surface (``zone/cluster/item``) and the same input distribution,
  * the same answer vocabulary and the same number of distinct answers,
  * the same trace format (one Step per evaluated clause, then the answer).

so that the only thing varying is the SHAPE OF THE RULE. Matched answer entropy matters: a
family whose answers are 90 % one label would look easy for a reason that has nothing to do
with writability, so ``balance()`` reports the majority-class share for every family and the
ladder should only compare families whose baselines are close.

Families, in rough order of expected resistance:

  ``map``          answer depends on ONE feature       (the Part D ordinary branch)
  ``composition``  answer = g(f(x)), two chained lookups
  ``conjunction``  answer depends on TWO features jointly
  ``disjunction``  answer if either feature matches
  ``negation``     one label everywhere EXCEPT a named subset
  ``precedence``   ordered clauses, last match wins     (the Part D exception branch)

``precedence`` and ``map`` are included deliberately as the two known endpoints: any ladder
that does not reproduce 1.00 and ~0.00 for them is measuring something other than what Part D
measured.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .base import Task

_ZONES = ["north", "south", "east", "west"]
_CLUSTERS = ["alpha", "bravo", "delta", "echo", "flint", "gamma"]
_ITEMS = ["rod", "vane", "spar", "hub", "keel", "mast", "prow", "stem"]
# answer vocabulary shares no substring with any input token, so no family can be solved by
# copying part of the input - the same guard as the workspace task's arbitrary labels
_LABELS = ["#q1", "#q2", "#q3", "#q4", "#q5", "#q6"]

FAMILIES = ("map", "map_wide", "composition", "conjunction", "disjunction", "negation",
            "precedence", "joint_k",
            # HARD tier: the seven above saturate the teacher by 3B, so a probe of a larger
            # model needs structures that are not already at ceiling, or it returns all-PASS
            # and measures nothing.
            "conj3", "prec5", "comp3")


class LogicTask(Task):
    name = "logic"
    header = "Resolve the record to its bucket.\n\n"
    input_label = "Record"
    answer_label = "Bucket"
    rule_noun = "ruleset"

    def __init__(self, rules: Dict[str, dict], tagged: bool = False, tag_map=None,
                 exception_fraction=None):
        super().__init__(rules)
        self.tagged = bool(tagged)
        self.tag_map = dict(tag_map) if tag_map else None
        self.exception_fraction = (None if exception_fraction is None
                                   else float(exception_fraction))
        self.spec: Dict[str, dict] = {}
        for r, s in rules.items():
            s = dict(s or {})
            fam = s.get("family", "map")
            if fam not in FAMILIES:
                raise ValueError(f"unknown family {fam}")
            rng = random.Random(int(s.get("seed", 0)))
            self.spec[r] = self._build(fam, rng, int(s.get("k", 0)))

    # ---- rule construction --------------------------------------------------
    def _build(self, family: str, rng: random.Random, k: int = 0) -> dict:
        labels = list(_LABELS)
        rng.shuffle(labels)
        d = {"family": family, "labels": labels}
        if family == "map":
            # bucket determined by cluster alone
            d["by_cluster"] = {c: labels[i % len(labels)] for i, c in enumerate(_CLUSTERS)}
        elif family == "composition":
            # cluster -> group -> bucket, two hops that must both be applied
            groups = ["g0", "g1", "g2"]
            d["to_group"] = {c: groups[i % len(groups)] for i, c in enumerate(_CLUSTERS)}
            d["group_to"] = {g: labels[i] for i, g in enumerate(groups)}
        elif family == "map_wide":
            # CONTROL for conjunction: the rule text lists all 24 (zone, cluster) pairs, so
            # the table and the prompt length match conjunction exactly - but the answer
            # depends on the cluster ALONE. map_wide vs conjunction isolates arity with text
            # length held constant; map vs map_wide isolates text length with structure held
            # constant.
            by_c = {c: labels[i % len(labels)] for i, c in enumerate(_CLUSTERS)}
            d["by_cluster"] = by_c
            d["joint"] = {(z, c): by_c[c] for z in _ZONES for c in _CLUSTERS}
        elif family == "joint_k":
            # Dose-response between map_wide (k=0) and conjunction (k=24): start from a table
            # that factors as f(cluster), then break exactly k of the 24 cells so those depend
            # on the PAIR. k measures how much of the mapping is non-decomposable, which is
            # the axis the map_wide/conjunction contrast only samples at its endpoints.
            by_c = {c: labels[i % len(labels)] for i, c in enumerate(_CLUSTERS)}
            d["by_cluster"] = by_c
            joint = {(z, c): by_c[c] for z in _ZONES for c in _CLUSTERS}
            cells = sorted(joint)
            rng.shuffle(cells)
            for (z, c) in cells[:max(0, min(k, len(cells)))]:
                alts = [a for a in labels if a != joint[(z, c)]]
                joint[(z, c)] = alts[(_ZONES.index(z) + _CLUSTERS.index(c)) % len(alts)]
            d["joint"] = joint
            d["k"] = k
        elif family == "conjunction":
            # answer depends on the PAIR; neither feature alone determines it
            d["joint"] = {(z, c): labels[(i + j) % len(labels)]
                          for i, z in enumerate(_ZONES) for j, c in enumerate(_CLUSTERS)}
        elif family == "disjunction":
            d["zone"] = rng.choice(_ZONES)
            d["cluster"] = rng.choice(_CLUSTERS)
            d["hit"], d["miss"] = labels[0], labels[1]
        elif family == "negation":
            d["except_clusters"] = sorted(rng.sample(_CLUSTERS, 2))
            d["default"], d["other"] = labels[0], labels[1]
        elif family == "conj3":
            # three-way joint: zone x cluster x item, none of the three determines the answer
            zs, cs, its = _ZONES[:2], _CLUSTERS[:3], _ITEMS[:4]     # 2x3x4 = 24 entries
            d["sub"] = (zs, cs, its)
            d["joint3"] = {(z, c, i): labels[(a + b + k) % len(labels)]
                           for a, z in enumerate(zs) for b, c in enumerate(cs)
                           for k, i in enumerate(its)}
        elif family == "comp3":
            # three hops: cluster -> group -> tier -> bucket
            groups, tiers = ["g0", "g1", "g2"], ["t0", "t1", "t2"]
            d["to_group"] = {c: groups[i % len(groups)] for i, c in enumerate(_CLUSTERS)}
            # a bijection here is fine - the structural point of comp3 is CHAIN DEPTH, not
            # information loss at every hop. Forcing a collision instead collapsed the answer
            # space to two labels and pushed the prior to 0.67, leaving no headroom to measure.
            d["group_tier"] = {g: tiers[i] for i, g in enumerate(groups)}
            d["tier_to"] = {t: labels[i] for i, t in enumerate(tiers)}
        elif family == "prec5":
            # five ordered clauses instead of three - depth of the precedence chain
            d["clauses"] = [("zone", rng.choice(_ZONES)), ("cluster", rng.choice(_CLUSTERS)),
                            ("item", rng.choice(_ITEMS)), ("zone", rng.choice(_ZONES)),
                            ("cluster", rng.choice(_CLUSTERS))]
            d["clause_labs"] = labels[:5]
            d["fallback"] = labels[5]
        elif family == "precedence":
            # ordered clauses, last match wins: a broad zone rule, a cluster rule, then an
            # item-level override that fires on a minority of inputs
            d["zone"] = rng.choice(_ZONES)
            d["cluster"] = rng.choice(_CLUSTERS)
            d["item"] = rng.choice(_ITEMS)
            d["zone_lab"], d["cluster_lab"], d["item_lab"] = labels[0], labels[1], labels[2]
        return d

    # ---- Task interface -----------------------------------------------------
    def _draw(self, rng: random.Random, rule: str = None) -> str:
        sub = self.spec.get(rule, {}).get("sub") if rule else None
        zs, cs, its = sub if sub else (_ZONES, _CLUSTERS, _ITEMS)
        return f"{rng.choice(zs)}/{rng.choice(cs)}/{rng.choice(its)}"

    def sample_input(self, rule: str, rng: random.Random) -> str:
        if self.exception_fraction is not None:
            want = rng.random() < self.exception_fraction
            for _ in range(300):
                x = self._draw(rng, rule)
                if self.is_exception(rule, x) == want:
                    return x
        return self._draw(rng, rule)

    def _parts(self, x: str) -> Tuple[str, str, str]:
        z, c, i = x.split("/")
        return z, c, i

    def solve(self, rule: str, input_str: str) -> Tuple[List[str], str]:
        d = self.spec[rule]
        z, c, i = self._parts(input_str)
        f = d["family"]
        if f == "map":
            a = d["by_cluster"][c]
            return [f"cluster {c} -> {a}"], a
        if f == "composition":
            g = d["to_group"][c]
            a = d["group_to"][g]
            return [f"cluster {c} -> {g}", f"{g} -> {a}"], a
        if f in ("conjunction", "map_wide", "joint_k"):
            a = d["joint"][(z, c)]
            return [f"zone {z} with cluster {c} -> {a}"], a
        if f == "disjunction":
            ok = (z == d["zone"] or c == d["cluster"])
            a = d["hit"] if ok else d["miss"]
            return [f"zone is {d['zone']}? {z == d['zone']}",
                    f"cluster is {d['cluster']}? {c == d['cluster']}",
                    f"either -> {a}"], a
        if f == "negation":
            ok = c not in d["except_clusters"]
            a = d["default"] if ok else d["other"]
            return [f"cluster {c} in {', '.join(d['except_clusters'])}? {not ok}",
                    f"-> {a}"], a
        if f == "conj3":
            a = d["joint3"][(z, c, i)]
            return [f"{z}+{c}+{i} -> {a}"], a
        if f == "comp3":
            g = d["to_group"][c]
            t = d["group_tier"][g]
            a = d["tier_to"][t]
            return [f"cluster {c} -> {g}", f"{g} -> {t}", f"{t} -> {a}"], a
        if f == "prec5":
            parts = {"zone": z, "cluster": c, "item": i}
            steps, a = [], None
            for (field, val), lab in zip(d["clauses"], d["clause_labs"]):
                if parts[field] == val:
                    a = lab
                    steps.append(f"{field} {val} matches -> {lab}")
            if a is None:
                a = d["fallback"]
                steps.append(f"no clause matches -> {a}")
            else:
                steps.append(f"last match wins -> {a}")
            return steps, a
        if f == "precedence":
            steps, a = [], None
            if z == d["zone"]:
                a = d["zone_lab"]
                steps.append(f"zone {d['zone']} matches -> {a}")
            if c == d["cluster"]:
                a = d["cluster_lab"]
                steps.append(f"cluster {d['cluster']} matches -> {a}")
            if i == d["item"]:
                a = d["item_lab"]
                steps.append(f"item {d['item']} matches -> {a}")
            if a is None:
                a = d["labels"][3]
                steps.append(f"no clause matches -> {a}")
            else:
                steps.append(f"last match wins -> {a}")
            return steps, a
        raise ValueError(f)

    def _describe_hard(self, d):
        f = d["family"]
        if f == "conj3":
            t = "; ".join(f"{z}+{c}+{i} -> {a}" for (z, c, i), a in d["joint3"].items())
            zs, cs, its = d["sub"]
            return (f"The record is zone/cluster/item, drawn from zones {', '.join(zs)}, "
                    f"clusters {', '.join(cs)}, items {', '.join(its)}. The bucket is set by "
                    f"all three together: {t}.")
        if f == "comp3":
            t1 = "; ".join(f"{c} -> {g}" for c, g in d["to_group"].items())
            t2 = "; ".join(f"{g} -> {t}" for g, t in d["group_tier"].items())
            t3 = "; ".join(f"{t} -> {a}" for t, a in d["tier_to"].items())
            return (f"The record is zone/cluster/item. Map the cluster to a group: {t1}. "
                    f"Then the group to a tier: {t2}. Then the tier to a bucket: {t3}.")
        if f == "prec5":
            t = "; ".join(f"{fl} {v} -> {lab}" for (fl, v), lab in
                          zip(d["clauses"], d["clause_labs"]))
            return (f"The record is zone/cluster/item. Check these five in order: {t}. If none "
                    f"match, the bucket is {d['fallback']}. The bucket is the one from the "
                    f"LAST clause that matches.")
        return None

    def is_valid_answer(self, rule: str, ans: str) -> bool:
        """A wrong answer is punishable only if it is one of the six bucket labels."""
        return (ans or "").strip() in _LABELS

    def is_exception(self, rule: str, input_str: str) -> bool:
        """An input is an 'exception' when a LATER clause overrides an earlier match - the
        structure Part D found unwritable.  Only meaningful for ``precedence``."""
        d = self.spec[rule]
        if d["family"] != "precedence":
            return False
        z, c, i = self._parts(input_str)
        matches = (z == d["zone"]) + (c == d["cluster"]) + (i == d["item"])
        return matches > 1

    def describe(self, rule: str) -> str:
        d = self.spec[rule]
        hard = self._describe_hard(d)
        if hard is not None:
            return hard
        f = d["family"]
        if f == "map":
            t = "; ".join(f"{c} -> {a}" for c, a in d["by_cluster"].items())
            return f"The record is zone/cluster/item. The bucket is set by the cluster: {t}."
        if f == "composition":
            t1 = "; ".join(f"{c} -> {g}" for c, g in d["to_group"].items())
            t2 = "; ".join(f"{g} -> {a}" for g, a in d["group_to"].items())
            return (f"The record is zone/cluster/item. First map the cluster to a group: {t1}. "
                    f"Then map the group to a bucket: {t2}.")
        if f in ("conjunction", "map_wide", "joint_k"):
            t = "; ".join(f"{z}+{c} -> {a}" for (z, c), a in d["joint"].items())
            return (f"The record is zone/cluster/item. The bucket is set by the zone and the "
                    f"cluster together: {t}.")
        if f == "disjunction":
            return (f"The record is zone/cluster/item. If the zone is {d['zone']} OR the "
                    f"cluster is {d['cluster']}, the bucket is {d['hit']}; otherwise it is "
                    f"{d['miss']}.")
        if f == "negation":
            return (f"The record is zone/cluster/item. Every record goes to {d['default']} "
                    f"EXCEPT those whose cluster is {' or '.join(d['except_clusters'])}, "
                    f"which go to {d['other']}.")
        if f == "precedence":
            return (f"The record is zone/cluster/item. Check these in order: zone "
                    f"{d['zone']} -> {d['zone_lab']}; cluster {d['cluster']} -> "
                    f"{d['cluster_lab']}; item {d['item']} -> {d['item_lab']}. If none match, "
                    f"the bucket is {d['labels'][3]}. The bucket is the one from the LAST "
                    f"clause that matches.")
        raise ValueError(f)

    def n_steps(self, rule: str) -> int:
        return 4

    def max_answer_tokens(self) -> int:
        return 8

    def tag(self, rule: str) -> str:
        return self.tag_map.get(rule, rule) if self.tag_map else rule

    # ---- ladder hygiene ------------------------------------------------------
    def balance(self, rule: str, n: int = 4000, seed: int = 0) -> Dict[str, float]:
        """Answer distribution for a family.  Families whose majority-class share differs
        wildly are not comparable: an easy-looking family may just have a skewed prior."""
        rng = random.Random(seed)
        counts: Dict[str, int] = {}
        for _ in range(n):
            a = self.answer_under(rule, self._draw(rng, rule))
            counts[a] = counts.get(a, 0) + 1
        return {"n_answers": len(counts),
                "majority_share": max(counts.values()) / n,
                "exception_share": sum(self.is_exception(rule, self._draw(rng, rule))
                                       for _ in range(n)) / n}

"""The self-upgrade sandbox: exposure stream -> probes -> REM, with rule switches."""
from __future__ import annotations

import difflib
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .encoding import Encoder
from .metrics import (cloze_accuracy, forward_logits, free_generation, generate, generate_text,
                      lm_perplexity, response_loss, score_pairs, trim_to_answer,
                      signed_response_loss, weighted_response_loss)
from .modelwrap import groups_from_config, install_hdelta, load_model
from .rem import embedding_std, measure_noise_response, rem_phase
from .report import summarize, to_markdown
from .tasks import Puzzle, make_task
from .toolcache import (CompiledMonitor, RollbackGuard, ToolVerifier,
                        generate_with_confidence)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Jsonl:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


class Experiment:
    def __init__(self, cfg: dict, out_dir: Path, quiet: bool = False):
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self.seed = int(cfg.get("seed", 0))
        set_seed(self.seed)

        mc = cfg["model"]
        self.device = mc.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok = load_model(mc["name"], mc.get("dtype", "float32"), self.device)
        pc = cfg["plastic"]
        groups = groups_from_config(pc["groups"])
        self.ctrl, self.info = install_hdelta(
            self.model, groups, max_fraction=pc.get("max_fraction", 0.05),
            postsynaptic=pc.get("postsynaptic", "error"),
            decay_enabled=pc.get("decay_enabled", True), homeo_enabled=pc.get("homeo_enabled", True),
            update_norm=pc.get("update_norm", "frob"))
        self.ctrl.updates_enabled = bool(pc.get("updates_enabled", True))
        self.enc = Encoder(self.tok, self.device)

        self.task = make_task(cfg["task"])
        self.rules = self.task.rule_names()
        sc = cfg["stream"]
        self.n_shots = int(sc.get("n_shots", 2))
        self.rule_text = bool(sc.get("rule_text", False))        # state the rule on the slow path
        self.self_generate = bool(sc.get("self_generate", False))  # the model thinks out loud itself
        self.free_probe = bool(sc.get("free_probe", self.self_generate))
        self.gen_max_new = int(sc.get("gen_max_new", 48))
        self.shots_cover_exception = bool(sc.get("shots_cover_exception", False))
        self.probe_minimal = bool(sc.get("probe_minimal", self.rule_text))  # skip icl/zero-scratch formats
        # fraction of WRONG self-generated answers the verifier wrongly accepts
        self.verifier_fpr = float(sc.get("verifier_fpr", 0.0))
        self.verifier_rng = random.Random(self.seed + 4242)
        # The tool is teacher and verifier both.  mode=systematic makes the failures
        # correlated instead of random, which is the untested deployment risk.
        self.verifier = ToolVerifier(self.task, mode=sc.get("verifier_mode", "honest"),
                                     fpr=self.verifier_fpr, seed=self.seed)
        self.poisoned: Dict[str, tuple] = {}     # input -> (wrong answer, true answer, rule)
        self.poison_n = int(sc.get("poison_probe_n", 32))
        self.n_false_accepts = 0
        self.n_unwritable = 0     # outputs skipped as un-decidable, not punished
        # Pain: on the teach path, ALSO write the model's own wrong answer with weight
        # -pain_lambda. 0.0 (the default) is the baseline exactly - no negative item is
        # ever created, so every chunk takes the original, untouched code path.
        self.pain_lambda = float(sc.get("pain_lambda", 0.0))
        self.n_pain = 0
        self.free_n = int(sc.get("free_n", 32))
        # compilation depth costs K+1 extra scoring passes per rule; not needed when the
        # question is capacity rather than hand-over
        self.probe_depth = bool(sc.get("probe_depth", True))
        probe_n = int(sc.get("probe_n", 48))
        self.probe_sets: Dict[str, List[Puzzle]] = {}
        self.probe_shots: Dict[str, List[Puzzle]] = {}
        # Probe items come from the HELD-OUT half of the input pools where the task has
        # them, so cold accuracy cannot be satisfied by recalling a trained string. Shots
        # stay on the training half: they are context, not a thing being generalised to.
        for i, r in enumerate(self.rules):
            rng = random.Random(10_000 + 17 * i + self.seed)
            held = getattr(self.task, "split", None) is not None
            if held:
                self.task.split = "heldout"
            self.probe_sets[r] = [self.task.sample(r, rng) for _ in range(probe_n)]
            if held:
                self.task.split = "train"
            self.probe_shots[r] = [self.task.sample(r, rng) for _ in range(self.n_shots)]
        # generation budget for direct answers, from the tokenizer (digits can be 2 tokens each)
        self.answer_max_tokens = 2 + max(len(self.enc.ids(" " + p.answer + "\n"))
                                         for r in self.rules for p in self.probe_sets[r])
        # sanity: tokenizer boundaries are clean for every format
        pz = self.probe_sets[self.rules[0]][0]
        for fmt in ("scratchpad", "direct"):
            p, r = self.task.make_pair(pz, fmt, shots=self.probe_shots[self.rules[0]])
            assert self.enc.check_boundary(p, r), f"BPE boundary crosses prompt/response in {fmt}"

        ec = cfg["eval"]
        root = Path(cfg.get("_root", "."))
        self.lm_text = (root / ec["lm_reference"]).read_text(encoding="utf-8")
        self.cloze = json.loads((root / ec["cloze"]).read_text(encoding="utf-8"))
        self.lm_windows = int(ec.get("lm_windows", 24))
        self.theta = float(ec.get("compile_threshold", 0.8))
        self.emb_std = embedding_std(self.model)
        self.noise_gen = torch.Generator(device=self.device).manual_seed(self.seed + 999)

        # Deliberate-vs-compiled health check.  The reference runs on the frozen weights
        # so a corrupted dW cannot drag it onto the reflex's own wrong answer.
        mcfg = cfg.get("monitor", {})
        self.monitor_on = bool(mcfg.get("enabled", False))
        self.monitor = CompiledMonitor(
            self.task, self.enc, threshold=float(mcfg.get("threshold", 0.15)),
            max_new_direct=self.answer_max_tokens,
            max_new_deliberate=int(mcfg.get("max_new_deliberate", 64)),
            ctrl=self.ctrl, deliberate_frozen=bool(mcfg.get("deliberate_frozen", True)))
        self.guard = RollbackGuard(self.ctrl, self.monitor) if mcfg.get("rollback", False) else None
        # The monitor calibrates against the tool too, but those checks are instrumentation,
        # not training writes.  Sharing the training verifier would fold them into
        # writes/contamination, which is the number the safety case rests on.
        self.monitor_verifier = ToolVerifier(self.task, mode="honest", seed=self.seed + 1)
        self.monitor_calibrated = False
        self.probe_count = 0
        self.monitor_every = int(mcfg.get("every", 0)) or None   # None -> every probe
        self.capture_confidence = bool(mcfg.get("confidence", False))
        self.save_delta = bool(mcfg.get("save_delta", False))
        # Per-probe spectra + a rolling window of full dW, for dissecting sudden collapses.
        # Full snapshots are ~340 MB each, so only `rolling` of them are kept; the spectra
        # are a few KB and answer "did a singular value spike or a sign flip" on their own.
        self.ckpt_spectra = bool(mcfg.get("checkpoint_spectra", False))
        self.ckpt_rolling = int(mcfg.get("checkpoint_rolling", 0))
        self.spectra_log = Jsonl(self.out / "spectra.jsonl") if self.ckpt_spectra else None

        self.seen: deque = deque(maxlen=int(sc.get("seen_window", 48)))
        self.probes = Jsonl(self.out / "probes.jsonl")
        # One record per DECISION, not per chunk. Chunk aggregates cannot say which
        # situations were punished, so they support neither the re-acquisition
        # prediction nor REM replay - this file IS the replay buffer.
        self.experience = Jsonl(self.out / "experience.jsonl")
        self.adv_scale = None        # running EMA of |advantage|, global to the run
        self.train_log = Jsonl(self.out / "train.jsonl")
        self.records: List[dict] = []
        self.t = 0
        self.phase_idx = -1
        self.phase_rule: Optional[str] = None
        self.prev_rule: Optional[str] = None
        self.phase_start = 0
        self.rem_count = 0
        self.cum_consolidated = 0      # verified loops written since the phase started
        self.last_probe_t = -1
        self.t_start = time.time()

        (self.out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (self.out / "model_info.json").write_text(json.dumps(self.info, indent=2), encoding="utf-8")
        self.say(f"model {mc['name']}: {self.info['total_params']/1e6:.1f}M params, "
                 f"plastic {self.info['plastic_params']/1e6:.2f}M ({self.info['plastic_fraction']:.2%}) "
                 f"in {len(self.info['modules'])} modules; frozen {self.info['frozen_fraction']:.2%}")

    def say(self, msg: str) -> None:
        if not self.quiet:
            print(f"[{time.time()-self.t_start:7.1f}s] {msg}", flush=True)

    # ------------------------------------------------------------------ probes
    @torch.no_grad()
    @torch.no_grad()
    def _dump_spectra(self, tag: str) -> None:
        """Singular-value spectrum of each plastic module, plus a rolling full snapshot.

        A collapse that comes from one direction blowing up shows as a spike in s1/s2; one
        that comes from a routing hyperplane inverting shows in the alignment against the
        previous probe (cosine of the top singular vector), not in the magnitudes.
        """
        rec = {"t": self.t, "tag": tag, "modules": {}}
        for k, mod in self.ctrl.modules.items():
            d = mod.delta.detach().float()
            sv = torch.linalg.svdvals(d)
            tot = float((sv ** 2).sum())
            rec["modules"][k] = {
                "fro": float(d.norm()),
                "top16": [float(x) for x in sv[:16]],
                # participation ratio: how many directions carry the energy
                "eff_rank": float((sv.sum() ** 2) / max((sv ** 2).sum(), 1e-12)),
                "top1_share": float(sv[0] ** 2 / max(tot, 1e-12)),
            }
        self.spectra_log.write(rec)
        if self.ckpt_rolling > 0:
            path = self.out / f"delta_t{self.t}.pt"
            torch.save({k: v.detach().cpu() for k, v in self.ctrl.snapshot().items()}, path)
            olds = sorted(self.out.glob("delta_t*.pt"),
                          key=lambda q: int(q.stem.split("t")[-1]))
            for q in olds[:-self.ckpt_rolling]:
                q.unlink()

    def probe(self, tag: str) -> dict:
        t0 = time.time()
        m: Dict[str, float] = {}
        samples: Dict[str, list] = {}
        per_puzzle: Dict[str, Dict[str, list]] = {}
        task = self.task
        for r in self.rules:
            pz = self.probe_sets[r]
            pp: Dict[str, list] = {}
            per_puzzle[r] = pp
            shots = self.probe_shots[r]
            K = task.n_steps(r)
            # 1) zero-shot direct: the compiled path.  Greedy generation -> accuracy and
            #    the rate at which the answer matches what *another* rule would produce.
            prompts = [task.make_pair(p, "direct")[0] for p in pz]
            if self.capture_confidence:
                # per-query sharpness: a per-rule health signal cannot route one query
                ans = generate_with_confidence(self.model, self.enc, prompts,
                                               self.answer_max_tokens)
                gens = [a.text for a in ans]
                pp["conf"] = [{"probs": a.probs, "margins": a.margins} for a in ans]
            else:
                gens = generate(self.model, self.enc, prompts, self.answer_max_tokens)
            m[f"{r}/zero_direct_acc"] = _mean(task.matches(g, p.answer, r, p.input_str)
                                              for g, p in zip(gens, pz))
            pp["direct"] = [int(task.matches(g, p.answer, r, p.input_str)) for g, p in zip(gens, pz)]
            # split by rule branch: the pooled number hid a 100% failure on the exception
            exc = [task.is_exception(r, p.input_str) for p in pz]
            pp["exception"] = [int(e) for e in exc]
            if any(exc):
                m[f"{r}/zero_direct_acc_exc"] = _mean(
                    task.matches(g, p.answer, r, p.input_str)
                    for g, p, e in zip(gens, pz, exc) if e)
            if not all(exc):
                m[f"{r}/zero_direct_acc_ord"] = _mean(
                    task.matches(g, p.answer, r, p.input_str)
                    for g, p, e in zip(gens, pz, exc) if not e)
            attrib = ["own" if task.matches(g, p.answer, r, p.input_str) else "other"
                      for g, p in zip(gens, pz)]
            for s in self.rules:
                if s == r:
                    continue
                hits = []
                for j, (g, p) in enumerate(zip(gens, pz)):
                    other = task.answer_under(s, p.input_str)
                    if other != p.answer:
                        hits.append(g == other)
                        if g == other:
                            attrib[j] = s
                m[f"{r}/rate_{s}"] = _mean(hits)
            pp["attrib"] = attrib
            pr, rs = zip(*[task.make_pair(p, "direct") for p in pz])
            ex, nll, ta = score_pairs(self.model, self.enc, pr, rs)
            pp["direct_tf"] = [int(e) for e in ex]
            m[f"{r}/zero_direct_tf_acc"] = _mean(ex)
            m[f"{r}/zero_direct_nll"] = _mean(nll)
            m[f"{r}/zero_direct_tokacc"] = _mean(ta)
            if not self.probe_minimal:
                # 2) in-context (excitation-based) direct
                pr, rs = zip(*[task.make_pair(p, "direct", shots=shots) for p in pz])
                ex, nll, ta = score_pairs(self.model, self.enc, pr, rs)
                m[f"{r}/icl_direct_acc"] = _mean(ex)
                m[f"{r}/icl_direct_nll"] = _mean(nll)
                m[f"{r}/icl_direct_tokacc"] = _mean(ta)
                # 3) scratchpad (slow path), zero-shot and in-context; exact match on steps+answer
                pr, rs = zip(*[task.make_pair(p, "scratchpad") for p in pz])
                ex, nll, ta = score_pairs(self.model, self.enc, pr, rs)
                m[f"{r}/zero_scratch_acc"] = _mean(ex)
                m[f"{r}/zero_scratch_nll"] = _mean(nll)
                m[f"{r}/zero_scratch_tokacc"] = _mean(ta)
                pr, rs = zip(*[task.make_pair(p, "scratchpad", shots=shots) for p in pz])
                ex, nll, ta = score_pairs(self.model, self.enc, pr, rs)
                m[f"{r}/icl_scratch_acc"] = _mean(ex)
                m[f"{r}/icl_scratch_nll"] = _mean(nll)
                m[f"{r}/icl_scratch_tokacc"] = _mean(ta)
                if not self.rule_text:
                    pp["slow"] = [int(e) for e in ex]      # few-shot slow path per puzzle
            # 3b) instructed formats: the rule is *stated*; the slow path is execution
            if self.rule_text:
                pr, rs = zip(*[task.make_pair(p, "scratchpad", shots=shots, rule_text=True) for p in pz])
                ex, nll, ta = score_pairs(self.model, self.enc, pr, rs)
                m[f"{r}/instructed_scratch_acc"] = _mean(ex)
                m[f"{r}/instructed_scratch_tokacc"] = _mean(ta)
                pp["slow"] = [int(e) for e in ex]
                pr, rs = zip(*[task.make_pair(p, "direct", shots=shots, rule_text=True) for p in pz])
                ex, nll, ta = score_pairs(self.model, self.enc, pr, rs)
                m[f"{r}/instructed_direct_acc"] = _mean(ex)
            # 3c) open-ended: after "Input: x" the model chooses whether to think out loud.
            #     Tokens spent and Step lines emitted are the cost of the excitation path.
            if self.free_probe:
                fr = free_generation(self.model, self.enc, task, pz[:self.free_n], rule_text=False,
                                     max_new_tokens=self.gen_max_new)
                m[f"{r}/free_acc"], m[f"{r}/free_steps"], m[f"{r}/free_tokens"] = fr["acc"], fr["steps"], fr["tokens"]
                pp["free"] = [int(c) for c in fr["correct"]]
                samples[f"{r}/free"] = [(p.input_str, p.answer, t) for p, t in zip(pz[:3], fr["texts"][:3])]
                if self.rule_text:
                    fr = free_generation(self.model, self.enc, task, pz[:self.free_n], rule_text=True,
                                         max_new_tokens=self.gen_max_new, shots=shots)
                    m[f"{r}/free_rule_acc"], m[f"{r}/free_rule_steps"], m[f"{r}/free_rule_tokens"] = fr["acc"], fr["steps"], fr["tokens"]
                    pp["slow_gen"] = [int(c) for c in fr["correct"]]
                    samples[f"{r}/free_rule"] = [(p.input_str, p.answer, t) for p, t in zip(pz[:3], fr["texts"][:3])]
            # 4) compilation depth: how many scratchpad steps must be supplied before the
            #    answer is right?  k = 0 is the compiled path, k = K is pure copying.
            needed = K + 1
            for k in (range(K + 1) if self.probe_depth else ()):
                pr, rs = zip(*[task.make_pair(p, "partial", k_steps=k) for p in pz])
                ex, _, _ = score_pairs(self.model, self.enc, pr, rs)
                acc = _mean(ex)
                m[f"{r}/partial{k}_acc"] = acc
                if acc >= self.theta and needed == K + 1:
                    needed = k
            if self.probe_depth:
                m[f"{r}/steps_needed"] = needed
        # recently seen stream puzzles (memorisation vs generalisation)
        if self.seen:
            seen = list(self.seen)
            pr, rs = zip(*[task.make_pair(p, "direct") for p in seen])
            ex, _, _ = score_pairs(self.model, self.enc, pr, rs)
            m["seen/zero_direct_acc"] = _mean(ex)
        # do the reinforced errors come back out?  For inputs whose wrong answer was
        # falsely accepted and consolidated, does the compiled path now reproduce it?
        if self.poisoned:
            items = list(self.poisoned.items())[-self.poison_n:]
            pz_in = [k for k, _ in items]
            prompts = [f"{task.header}{task.input_label} [{task.tag(v[2])}]: {k}\n{task.answer_label}:"
                       if getattr(task, "tagged", False) else
                       f"{task.header}{task.input_label}: {k}\n{task.answer_label}:"
                       for k, v in items]
            gens = generate(self.model, self.enc, prompts, self.answer_max_tokens)
            m["poison/n_tracked"] = len(items)
            m["poison/reproduces_error"] = _mean(g == v[0] for g, (_, v) in zip(gens, items))
            m["poison/recovers_truth"] = _mean(g == v[1] for g, (_, v) in zip(gens, items))
            m["poison/false_accepts_total"] = self.n_false_accepts
        # deliberate-vs-compiled health, the signal that survives without ground truth.
        # The deliberate pass is a full generation over every probe set, so it is the most
        # expensive thing here; monitor.every runs it on a subset of probes.
        self.probe_count += 1
        due = (self.monitor_every is None or not self.monitor_calibrated
               or self.probe_count % self.monitor_every == 0)
        if self.monitor_on and due:
            for r in self.rules:
                if not self.monitor_calibrated:
                    # calibrate on the SAME set we monitor: the reference's error rate
                    # swings too far across draws for a borrowed floor to mean anything
                    self.monitor.calibrate(self.model, r, self.probe_sets[r],
                                           self.probe_shots[r], self.monitor_verifier)
                res = self.monitor.check(self.model, r, self.probe_sets[r], self.probe_shots[r])
                m[f"{r}/mon_divergence"] = res.divergence
                m[f"{r}/mon_floor"] = res.divergence_floor
                m[f"{r}/mon_excess"] = res.excess_divergence
                m[f"{r}/mon_agree_wrong"] = res.agree_wrong
                m[f"{r}/mon_gap"] = res.gap
                m[f"{r}/mon_healthy"] = int(res.healthy)
                per_puzzle[r]["agree_wrong_idx"] = res.agree_wrong_idx
                if self.guard is not None and r == self.phase_rule:
                    m[f"{r}/mon_rolled_back"] = int(self.guard.observe(res))
            self.monitor_calibrated = True
        if self.ckpt_spectra:
            self._dump_spectra(tag)
        m.update({f"verifier/{k}": v for k, v in self.verifier.state().items()})
        m["monitor/calibration_calls"] = self.monitor_verifier.calls
        # agent-side calls only; task.tool_calls() also counts dataset construction and the
        # probe's cross-rule attribution loop, which no deployment would pay for
        m["agent_tool_calls"] = self.verifier.calls
        m["tool_calls"] = self.task.tool_calls()
        # base language skills (catastrophic-forgetting check)
        m["lm_ppl"] = lm_perplexity(self.model, self.enc, self.lm_text, max_windows=self.lm_windows)
        m["cloze_acc"] = cloze_accuracy(self.model, self.enc, self.cloze)
        # fast-weight health
        m.update(self.ctrl.state())
        ratios = measure_noise_response(self.model, self.ctrl, self.cfg["rem"], self.emb_std, self.noise_gen)
        per_group: Dict[str, List[float]] = {}
        for k, v in ratios.items():
            m[f"noise_ratio/{k}"] = v
            per_group.setdefault(self.ctrl.modules[k].group, []).append(v)
        for g, vals in per_group.items():
            m[f"noise_ratio_group/{g}"] = _mean(vals)

        rec = {"t": self.t, "phase": self.phase_idx, "rule": self.phase_rule, "prev_rule": self.prev_rule,
               "phase_start": self.phase_start, "tag": tag, "rem_count": self.rem_count,
               "probe_seconds": time.time() - t0, "metrics": m, "samples": samples,
               "per_puzzle": per_puzzle, "cum_consolidated": self.cum_consolidated}
        self.probes.write(rec)
        self.records.append(rec)
        self.last_probe_t = self.t
        parts = [f"t={self.t:5d} {tag:8s}"]
        for r in self.rules:
            slow_key = f"{r}/instructed_scratch_acc" if self.rule_text else f"{r}/icl_scratch_acc"
            extra = ""
            if self.free_probe:
                extra = f" free={m[f'{r}/free_acc']:.2f}/{m[f'{r}/free_steps']:.1f}st/{m[f'{r}/free_tokens']:.0f}tok"
            need = f" need={m[f'{r}/steps_needed']}" if f"{r}/steps_needed" in m else ""
            exc = (f"({m[f'{r}/zero_direct_acc_exc']:.2f}exc/{m.get(f'{r}/zero_direct_acc_ord', float('nan')):.2f}ord)"
                   if f"{r}/zero_direct_acc_exc" in m else "")     # a probe set can be all-exception
            # probe_minimal without rule_text (tool-as-teacher tasks) has no slow-path score
            slow = f" slow={m[slow_key]:.2f}" if slow_key in m else ""
            parts.append(f"{r}: direct={m[f'{r}/zero_direct_acc']:.2f}{exc}{slow}{need}{extra}")
        if self.monitor_on and f"{self.rules[0]}/mon_excess" in m:
            parts.append("mon=" + "/".join(
                f"{m[f'{r}/mon_excess']:+.2f}{'' if m[f'{r}/mon_healthy'] else '!'}"
                for r in self.rules))
        parts.append(f"ppl={m['lm_ppl']:.2f} cloze={m['cloze_acc']:.2f}")
        parts.append("fw=" + "/".join(f"{v:.3f}" for k, v in m.items() if k.startswith("fw_norm_ratio_group/")))
        parts.append("nr=" + "/".join(f"{v:.2e}" for k, v in m.items() if k.startswith("noise_ratio_group/")))
        self.say("  ".join(parts))
        return rec

    # ---------------------------------------------------------------- exposure
    def _valence_weights(self, advantages, clip: float = 2.0, ema: float = 0.99):
        """Signed per-item weights from raw advantages, normalised against a RUNNING scale.

        The scale is global to the run, not local to the chunk, and that is the whole point.
        An earlier version normalised by the chunk's own median and then forced
        sum(|w|) == 1, which gave exact control of the effective learning rate but made an
        item's write depend on its accidental chunk-mates: measured over 2000 draws, an
        identical item (advantage +5.0) received weights spanning **38.5x** depending only on
        which three puzzles shared its batch. That is a batching artifact landing directly in
        the learned policy, and it is larger than the learning-rate difference the
        renormalisation was protecting against.

        With a running scale, `w` is a function of the item and the run's history alone, so
        the same (situation, move, outcome) is written the same way wherever it lands.

        Dividing by N keeps the magnitude comparable to the unsigned case, where the loss is a
        mean over items. Note sum(|w|) is now only approximately 1 - a chunk of decisive
        outcomes writes more than a chunk of neutral ones. That is the "neutral costs less
        capacity" behaviour the design asked for, and it now falls out instead of being
        suppressed. Per-chunk mass is logged rather than enforced.

        Clipping still comes first: an unparseable move scores about -130 where a merely bad
        one scores -3, and without a clip one malformed answer would eat the chunk.
        """
        import torch as _t
        a = _t.tensor([float(x) for x in advantages])
        med = float(a.abs().median())
        if self.adv_scale is None:
            self.adv_scale = max(med, 1e-6)
        elif med > 0:
            self.adv_scale = ema * self.adv_scale + (1.0 - ema) * med
        w = (a / self.adv_scale).clamp(-clip, clip) / max(len(a), 1)
        return w

    def _step(self, pairs, decay_puzzles: float, weights=None, signs=None,
              include=None) -> float:
        prompts, responses = zip(*pairs)
        batch = self.enc.encode_pairs(prompts, responses)
        need_grad = self.ctrl.postsynaptic == "error" and self.ctrl.updates_enabled
        self.ctrl.set_record(self.ctrl.postsynaptic == "hebbian")
        with torch.set_grad_enabled(need_grad):
            logits = forward_logits(self.model, batch)
            if signs is not None:
                inc = None
                if include is not None:
                    R = max(len(r) for r in include)
                    inc = torch.tensor([r + [False] * (R - len(r)) for r in include])
                loss = signed_response_loss(logits, batch, torch.tensor(signs), inc)
            elif weights is not None:
                loss = weighted_response_loss(logits, batch, weights)
            else:
                loss = response_loss(logits, batch)
        if self.ctrl.updates_enabled:
            self.ctrl.update_from_batch(loss if need_grad else None, batch.hidden_mask, decay_puzzles)
        self.ctrl.set_record(False)
        return float(loss.detach())

    def expose_chunk(self, rule: str, puzzles: List[Puzzle], rng: random.Random) -> Dict[str, float]:
        """One chunk of the stream.

        slow path     : few-shot + scratchpad ("thinking out loud"); the h-delta modules
                        update on every step token and the answer.
        consolidation : the same puzzles replayed zero-shot in direct form, so the
                        input->answer loop that the slow path produced is written
                        structurally (the "self-modification" hand-off).
        """
        sc = self.cfg["stream"]
        fmt = sc.get("exposure", "both")
        bs = len(puzzles)
        stats: Dict[str, float] = {}
        to_consolidate: List = list(puzzles)
        if fmt in ("slow", "both"):
            shots = [self.task.sample(rule, rng) for _ in range(self.n_shots)]
            if self.shots_cover_exception and self.n_shots:
                # Force the tail into the context window.  Without this the exception
                # branch is demonstrated almost never, and what the verifier accepts is
                # almost entirely the common case.
                for _ in range(64):
                    cand = self.task.sample(rule, rng)
                    if self.task.is_exception(rule, cand.input_str):
                        shots[0] = cand
                        break
            if self.self_generate:
                # The excitation path proper: the model generates its own steps and answer
                # with the current (plastic) network.  Verified answers are written back as
                # the model's own reasoning; wrong ones get the correct demonstration
                # (on_wrong: teach) or nothing (on_wrong: skip).
                prompts = [self.task.make_pair(p, "scratchpad", shots=shots, rule_text=self.rule_text)[0]
                           for p in puzzles]
                texts = generate_text(self.model, self.enc, prompts, self.gen_max_new)
                pairs, ok, toks, accepted, answers = [], [], [], [], []
                adv_w = []            # signed advantages, parallel to `pairs`
                adv_rows = []         # one dict per write, parallel to `pairs`
                neg_pairs = []        # own WRONG answers to suppress (pain_lambda > 0)
                neg_inc = []          # per negative: which of its tokens were WRONG
                for p, prompt, text in zip(puzzles, prompts, texts):
                    ans, _ = self.task.parse_generation(text)
                    # the tool is the authority and every call it makes is counted
                    verdict = self.verifier.check(rule, p.input_str, ans)
                    correct, passes = verdict.correct, verdict.accepted
                    ok.append(correct)
                    accepted.append(passes)
                    answers.append(ans)
                    toks.append(len(self.enc.ids(trim_to_answer(self.task, text))))
                    if sc.get("on_wrong", "teach") == "positive" and hasattr(self.task, "contrast"):
                        # ABLATION: graded reward, no punishment. Isolates which half of
                        # valence does the work. The unsigned baseline writes every verified
                        # item with the SAME eta, so a move that scraped over the bar and one
                        # that was decisively right are recorded identically; this grades them
                        # but still never pushes anything down. If this alone prevents the
                        # collapse, the missing ingredient was gradation, not punishment.
                        adv, _ = self.task.contrast(rule, ans, p.input_str)
                        if adv is None:
                            self.n_unwritable += 1
                        elif adv > 0:
                            pairs.append((prompt, trim_to_answer(self.task, text)))
                            adv_w.append(adv)
                            adv_rows.append({"rule": rule, "input": p.input_str, "move": ans,
                                             "adv": adv, "kind": "own", "verified": bool(passes)})
                    elif sc.get("on_wrong", "teach") == "contrast" and hasattr(self.task, "contrast"):
                        # Signed, graded write. The move is ALWAYS written - with a positive
                        # weight when it beat the sampled alternatives and a negative one when
                        # it lost to them - so the integrator stops being monotone. A
                        # near-zero advantage writes almost nothing and decay takes it, which
                        # is the "neutral" channel the loop previously had no way to express.
                        adv, better = self.task.contrast(rule, ans, p.input_str)
                        if adv is None:
                            # unwritable output: no decision was made, so nothing to write
                            self.n_unwritable += 1
                            continue
                        pairs.append((prompt, trim_to_answer(self.task, text)))
                        adv_w.append(adv)
                        adv_rows.append({"rule": rule, "input": p.input_str, "move": ans,
                                         "adv": adv, "kind": "own", "verified": bool(passes),
                                         "better": better})
                        if better is not None:
                            # Contrastive, not oracle: `better` is the best of the SAMPLED
                            # alternatives, never the enumerated optimum. Pure punishment says
                            # where not to go and not where to go, so mass pushed off a bad
                            # move lands on whatever was next most likely.
                            pairs.append(self.task.make_pair(
                                Puzzle(p.input_str, [], better, p.rule, dict(p.meta)), "direct"))
                            adv_w.append(abs(adv))
                            adv_rows.append({"rule": rule, "input": p.input_str, "move": better,
                                             "adv": abs(adv), "kind": "alt", "verified": True})
                    elif passes:
                        # written back as the model's own reasoning, right or wrong
                        pairs.append((prompt, trim_to_answer(self.task, text)))
                        if not correct:
                            self.n_false_accepts += 1
                            self.poisoned[p.input_str] = (ans, p.answer, p.rule)
                    elif sc.get("on_wrong", "teach") == "teach":
                        # A corrupted verifier does not merely reject the truth - it supplies
                        # its OWN wrong correction. Teaching the real answer here would make
                        # corruption a no-op (reject, then teach the truth anyway), which is
                        # not what a mislabelling oracle does.
                        tp = p
                        if self.verifier.mode == "corrupt":
                            believed = self.verifier.truth(rule, p.input_str)
                            if believed != p.answer:
                                steps, _ = self.task.solve(rule, p.input_str)
                                tp = Puzzle(p.input_str, steps[:-1] + [f"-> {believed}"],
                                            believed, p.rule, dict(p.meta))
                        pairs.append(self.task.make_pair(tp, "scratchpad", shots=shots,
                                                         rule_text=self.rule_text))
                        # PAIN. The teach demonstration above is unchanged; this only adds a
                        # negative write on what the model actually produced. Legitimate wrong
                        # answers only - never an unparseable one.
                        if (self.pain_lambda > 0 and not correct
                                and self.task.is_valid_answer(rule, ans)):
                            # Point at the mistake. The correct demonstration was produced from
                            # the SAME prompt, so the tokens the two traces share are tokens the
                            # model got right; the negative goes only on where they part ways.
                            own = trim_to_answer(self.task, text)
                            good = self.task.make_pair(p, "scratchpad", shots=shots,
                                                       rule_text=self.rule_text)[1]
                            oi, gi = self.enc.ids(own), self.enc.ids(good)
                            # Punish only tokens the model produced that do NOT align with the
                            # correct demonstration. A sequence alignment, not a positional one:
                            # a positional compare slips as soon as the model phrases its
                            # restatement a few tokens shorter (seen: 15 vs 20), and then
                            # punishes every later token including correct `Bucket:` formatting -
                            # the off-format push that wrecked the world runs. Tokens the model
                            # merely OMITTED are not punishable (nothing to push down); the
                            # positive teach demonstration already covers them.
                            inc = [False] * len(oi)
                            for tag, i1, i2, _, _ in difflib.SequenceMatcher(
                                    a=oi, b=gi, autojunk=False).get_opcodes():
                                if tag in ("replace", "delete"):
                                    for j in range(i1, i2):
                                        inc[j] = True
                            if any(inc):
                                neg_pairs.append((prompt, own))
                                neg_inc.append(inc)
                                self.n_pain += 1
                                self.experience.write({
                                    "t": self.t, "rule": rule, "kind": "pain",
                                    "input": p.input_str, "move": ans, "truth": p.answer,
                                    "weight": -self.pain_lambda,
                                    "n_tokens": len(oi), "n_punished": sum(inc),
                                    "punished": [self.enc.tok.decode([oi[j]])
                                                 for j in range(len(oi)) if inc[j]]})
                stats["slow_acc"] = _mean(ok)
                stats["slow_accepted"] = _mean(accepted)
                stats["false_accepts"] = self.n_false_accepts
                stats["slow_tokens"] = _mean(toks)
                stats.update(self.verifier.state())
                stats["tool_calls"] = self.task.tool_calls()
                if pairs and neg_pairs:
                    signs = [1.0] * len(pairs) + [-self.pain_lambda] * len(neg_pairs)
                    inc = [[True] * len(self.enc.ids(r)) for _, r in pairs] + neg_inc
                    stats["loss_slow"] = self._step(pairs + neg_pairs, decay_puzzles=bs,
                                                    signs=signs, include=inc)
                    stats["pain_writes"] = len(neg_pairs)
                    stats["pain_total"] = self.n_pain
                elif pairs:
                    w = self._valence_weights(adv_w) if adv_w else None
                    stats["loss_slow"] = self._step(pairs, decay_puzzles=bs, weights=w)
                    if adv_w:
                        stats["adv_mean"] = _mean(adv_w)
                        stats["adv_neg_share"] = _mean([a < 0 for a in adv_w])
                        # mass is LOGGED, not enforced: a chunk of decisive outcomes should
                        # write more than a chunk of neutral ones
                        stats["adv_chunk_mass"] = float(w.abs().sum())
                        stats["adv_scale"] = float(self.adv_scale)
                        stats["unwritable_total"] = self.n_unwritable
                        for row, wi in zip(adv_rows, w.tolist()):
                            row.update(t=self.t, weight=wi)
                            self.experience.write(row)
                else:
                    self.ctrl.decay(bs)
                if sc.get("consolidate", "verified") == "verified":
                    # consolidate what the verifier ACCEPTED, using the model's own answer
                    to_consolidate = [(p, a) for p, a, acc in zip(puzzles, answers, accepted) if acc]
            else:
                pairs = [self.task.make_pair(p, "scratchpad", shots=shots, rule_text=self.rule_text)
                         for p in puzzles]
                stats["loss_slow"] = self._step(pairs, decay_puzzles=bs)
        if fmt in ("direct", "both") and to_consolidate:
            # consolidation: the verified input->answer loop replayed zero-shot, so it is
            # written structurally (the hand-off from excitation to the h-delta groups).
            # For a falsely accepted item the answer replayed is the model's wrong one.
            pairs = []
            for item in to_consolidate:
                if isinstance(item, tuple):
                    pz_i, ans_i = item
                    pairs.append((self.task.make_pair(pz_i, "direct")[0], f" {ans_i}\n"))
                else:
                    pairs.append(self.task.make_pair(item, "direct"))
            for rep in range(int(sc.get("consolidation_repeats", 1))):
                stats["loss_direct"] = self._step(
                    pairs, decay_puzzles=bs if (fmt == "direct" and rep == 0) else 0)
            # REPLAY: `replay` extra update steps per chunk, each on a batch drawn at random from
            # every pair consolidated EARLIER in the stream - revisiting old verified data, not
            # repeating the current batch (which is what consolidation_repeats does). No new tool
            # calls: the stored answer is the one the tool already gave. 0, the default, never
            # enters this branch, so every existing config runs its original code path.
            rp = int(sc.get("replay", 0))
            if rp > 0:
                if not hasattr(self, "replay_buf"):
                    self.replay_buf, self.replay_rng = [], random.Random(self.seed + 777)
                if len(self.replay_buf) >= bs:
                    losses = [self._step(self.replay_rng.sample(self.replay_buf, bs), decay_puzzles=0)
                              for _ in range(rp)]
                    stats["loss_replay"] = sum(losses) / len(losses)
                self.replay_buf.extend(pairs)
        stats["n_consolidated"] = len(to_consolidate) if fmt in ("direct", "both") else 0
        return stats

    # --------------------------------------------------------------------- run
    def run(self) -> dict:
        sc = self.cfg["stream"]
        rc = self.cfg["rem"]
        bs = int(sc.get("batch_size", 4))
        probe_every = int(sc.get("probe_every", 20))
        rem_every = int(rc.get("every", 60))
        self.probe("baseline")
        for i, ph in enumerate(sc["phases"]):
            self.phase_idx = i
            self.prev_rule = self.phase_rule if self.phase_rule != ph["rule"] else self.prev_rule
            self.phase_rule = ph["rule"]
            self.phase_start = self.t
            self.cum_consolidated = 0
            n = int(ph["puzzles"])
            rng = random.Random(self.seed * 1000 + 7 * i + 3)
            self.probes.write({"event": "phase_start", "t": self.t, "phase": i, "rule": ph["rule"]})
            self.say(f"=== phase {i}: rule {ph['rule']} for {n} puzzles ===")
            while self.t - self.phase_start < n:
                puzzles = [self.task.sample(ph["rule"], rng) for _ in range(bs)]
                stats = self.expose_chunk(ph["rule"], puzzles, rng)
                self.cum_consolidated += int(stats.get("n_consolidated", 0))
                stats["cum_consolidated"] = self.cum_consolidated
                self.seen.extend(puzzles)
                t_prev, self.t = self.t, self.t + bs
                st = self.ctrl.state()
                self.train_log.write({"t": self.t, "phase": i, "rule": ph["rule"], **stats,
                                      **{k: v for k, v in st.items() if "group" in k}})
                if self.t // probe_every > t_prev // probe_every:
                    self.probe("stream")
                if rc.get("enabled", True) and self.t // rem_every > t_prev // rem_every:
                    # probing either side of every sleep doubles the probe count on long
                    # runs; REM cost is already characterised, so it is optional
                    around = bool(rc.get("probe_around", True))
                    if around and self.last_probe_t != self.t:
                        self.probe("pre_rem")
                    res = rem_phase(self.model, self.ctrl, rc, self.emb_std, self.noise_gen)
                    self.rem_count += 1
                    self.probes.write({"event": "rem", "t": self.t, "phase": i, "before": res["before"],
                                       "after": res["after"], "steps": res["steps"]})
                    if not around:
                        self.probes.write({'event': 'rem_quiet', 't': self.t, 'phase': i})
                    self.say(f"REM #{self.rem_count}: noise ratio "
                             + ", ".join(f"{k.split('.')[-3]}.{k.split('.')[-1]} {res['before'][k]:.2e}->{res['after'][k]:.2e}"
                                         for k in res["before"]))
                    if around:
                        self.probe("post_rem")
        if self.last_probe_t != self.t:
            self.probe("final")
        if self.save_delta:
            # the compiled state itself, so any later gate question is answered offline
            torch.save({k: v.detach().cpu() for k, v in self.ctrl.snapshot().items()},
                       self.out / "delta.pt")
            self.say(f"saved fast weights -> {self.out / 'delta.pt'}")
        summary = summarize(self.records, self.cfg)
        summary["model_info"] = self.info
        summary["wall_seconds"] = time.time() - self.t_start
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.out / "summary.md").write_text(to_markdown(summary, self.cfg), encoding="utf-8")
        try:
            from .handover import analyse_run
            analyse_run(self.out)
        except Exception as e:      # the analysis must never take a finished run down with it
            self.say(f"hand-over analysis failed: {e!r}")
        self.probes.close()
        self.train_log.close()
        self.experience.close()
        if self.spectra_log is not None:
            self.spectra_log.close()
        self.say(f"done in {summary['wall_seconds']:.0f}s -> {self.out}")
        return summary

"""Gate A: does PEFT give `CompiledMonitor` a genuinely frozen reference?

The monitor's ONLY coupling to h-delta is `ctrl.set_apply_delta(bool)`. Everything else
takes a model, an encoder and a task. So the whole architectural bridge to LoRA is a shim
with that one method - and Gate A exists to prove the shim is sound before Gate B spends
hours measuring transfer through it.

What is actually at stake: the monitor is valid only if the deliberate path runs on weights
the reflex cannot touch. If disabling the adapter leaves any residue, the reference drifts
with the thing it is auditing, divergence collapses toward zero and the hook silently stops
working. That is the failure this gate is looking for.

Four checks, in order of what they would kill:

  1. TARGETING   PEFT matched the intended 36 modules. A mistyped pattern matches nothing,
                 and every check below would then pass against an empty adapter.
  2. LIVE        with lora_B perturbed nonzero, outputs MOVE. PEFT zero-initialises lora_B,
                 so an unperturbed adapter is the identity and "disabled == base" is a
                 trivial pass that proves nothing.
  3. EXACT       with the adapter disabled, outputs return to base - within the noise floor
                 measured by running the identical forward pass twice. bf16 matmuls on this
                 card are not bit-reproducible, so "== 0" is the wrong assertion; the right
                 one is "no larger than the model's own repeat noise".
  4. REVERSIBLE  re-enabling restores the perturbed outputs. The monitor toggles this on
                 every check, so a one-way or leaky toggle would degrade run to run.

Then one live `calibrate()` + `check()` through the shim, unmodified.

    .venv/Scripts/python.exe gate_a_lora.py --config configs/gate_a_lora.yaml --out results/gate_a

Nothing is trained. Nothing here measures transfer - that is Gate B.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml

from hdelta.encoding import Encoder
from hdelta.modelwrap import load_model
from hdelta.tasks import make_task
from hdelta.toolcache import CompiledMonitor, ToolVerifier


class LoraCtrl:
    """The entire h-delta -> LoRA bridge.

    `CompiledMonitor` calls `set_apply_delta(False)` to get the frozen reference and
    `True` to restore. PEFT's `enable_adapter_layers` / `disable_adapter_layers` are the
    exact analogue of h-delta's apply-delta toggle.
    """

    def __init__(self, peft_model):
        self.m = peft_model
        self.toggles = 0

    def set_apply_delta(self, on: bool) -> None:
        self.toggles += 1
        if on:
            self.m.base_model.enable_adapter_layers()
        else:
            self.m.base_model.disable_adapter_layers()


@torch.no_grad()
def last_logits(model, enc, prompts, device):
    """Last-position logits only. The full [B, T, 151k] tensor would not fit, and the last
    position is where every answer is decided anyway.

    `device` is passed explicitly rather than read off the model: after `get_peft_model`
    the object is a PeftModel wrapper, and relying on attribute proxying for `.device`
    would make this silently version-dependent.
    """
    out = []
    for p in prompts:
        ids = torch.tensor([enc.ids(p)], device=device)
        out.append(model(input_ids=ids).logits[0, -1, :].float().cpu())
    return torch.stack(out)


def dmax(a, b):
    return (a - b).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gate_a_lora.yaml")
    ap.add_argument("--out", default="results/gate_a")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    ga, lc = cfg["gate_a"], cfg["lora"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"config": a.config}
    t0 = time.time()

    from peft import LoraConfig, get_peft_model      # imported late: clearer failure

    torch.manual_seed(int(cfg.get("seed", 0)))
    task = make_task(cfg["task"])
    rule = next(iter(cfg["task"]["rules"]))

    model, tok = load_model(cfg["model"]["name"], cfg["model"]["dtype"],
                            cfg["model"]["device"])
    dev = cfg["model"]["device"]
    enc = Encoder(tok, dev)
    n_layers = int(model.config.num_hidden_layers)
    print(f"[{time.time() - t0:6.1f}s] base loaded, {n_layers} layers", flush=True)

    rng = random.Random(int(cfg.get("seed", 0)))
    pz = [task.sample(rule, rng) for _ in range(ga["n_logit_prompts"])]
    prompts = [task.make_pair(p, "direct")[0] for p in pz]

    # --- noise floor: the SAME forward pass twice, nothing changed in between ---------
    base1 = last_logits(model, enc, prompts, dev)
    base2 = last_logits(model, enc, prompts, dev)
    floor = dmax(base1, base2)
    print(f"[{time.time() - t0:6.1f}s] repeat-pass noise floor: {floor:.3e}", flush=True)
    rep["noise_floor"] = floor

    # --- 1. TARGETING ----------------------------------------------------------------
    targets = [f"model.layers.{n_layers + L if L < 0 else L}.self_attn.{p}"
               for L in lc["layers"] for p in lc["projections"]]
    peft_cfg = LoraConfig(r=int(lc["r"]), lora_alpha=int(lc["alpha"]),
                          lora_dropout=float(lc["dropout"]),
                          target_modules=targets, bias="none",
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok_target = len(adapted) == int(lc["expect_modules"]) and trainable > 0
    print(f"[{time.time() - t0:6.1f}s] 1 TARGETING  matched {len(adapted)}/"
          f"{lc['expect_modules']} modules, {trainable / 1e6:.2f}M trainable"
          f"  {'ok' if ok_target else 'FAIL'}", flush=True)
    rep.update(adapted_modules=len(adapted), expect_modules=int(lc["expect_modules"]),
               trainable_params=trainable, targeting_ok=bool(ok_target))

    init = last_logits(model, enc, prompts, dev)
    rep["identity_at_init"] = dmax(init, base1)
    print(f"[{time.time() - t0:6.1f}s]   adapter at init (lora_B=0) vs base: "
          f"{rep['identity_at_init']:.3e}", flush=True)

    # --- 2. LIVE: perturb lora_B so the adapter actually does something ---------------
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * float(ga["perturb_scale"]))
    pert = last_logits(model, enc, prompts, dev)
    moved = dmax(pert, base1)
    ok_live = moved > max(floor * 10, 1e-3)
    print(f"[{time.time() - t0:6.1f}s] 2 LIVE       perturbed vs base: {moved:.3e}"
          f"  {'ok' if ok_live else 'FAIL - adapter is not in the forward pass'}",
          flush=True)
    rep.update(perturbed_delta=moved, live_ok=bool(ok_live))

    # --- 3. EXACT: disabled must return to base, within the noise floor --------------
    ctrl = LoraCtrl(model)
    ctrl.set_apply_delta(False)
    disabled = last_logits(model, enc, prompts, dev)
    resid = dmax(disabled, base1)
    ok_exact = resid <= max(floor, 1e-6)
    print(f"[{time.time() - t0:6.1f}s] 3 EXACT      disabled vs base: {resid:.3e}"
          f"  (floor {floor:.3e})  "
          f"{'ok' if ok_exact else 'FAIL - reference is CONTAMINATED'}", flush=True)
    rep.update(disabled_residual=resid, exact_ok=bool(ok_exact))

    # --- 4. REVERSIBLE ---------------------------------------------------------------
    ctrl.set_apply_delta(True)
    back = last_logits(model, enc, prompts, dev)
    rev = dmax(back, pert)
    ok_rev = rev <= max(floor, 1e-6)
    print(f"[{time.time() - t0:6.1f}s] 4 REVERSIBLE re-enabled vs perturbed: {rev:.3e}"
          f"  {'ok' if ok_rev else 'FAIL - toggle leaks state'}", flush=True)
    rep.update(reenabled_residual=rev, reversible_ok=bool(ok_rev))

    # --- the monitor itself, unmodified ----------------------------------------------
    mon = CompiledMonitor(task, enc, threshold=float(cfg["monitor"]["threshold"]),
                          ctrl=ctrl,
                          deliberate_frozen=bool(cfg["monitor"]["deliberate_frozen"]))
    mrng = random.Random(999)
    mpz = [task.sample(rule, mrng) for _ in range(ga["n_monitor_puzzles"])]
    shots = [task.sample(rule, mrng) for _ in range(2)]
    ver = ToolVerifier(task, mode="honest", fpr=0.0, seed=0)
    rel = mon.calibrate(model, rule, mpz, shots, ver)
    res = mon.check(model, rule, mpz, shots)
    print(f"[{time.time() - t0:6.1f}s] MONITOR      reliability {rel:.3f} -> "
          f"floor {res.divergence_floor:.3f}", flush=True)
    print(f"             {res.summary()}", flush=True)
    rep.update(monitor_reliability=rel, monitor_divergence=res.divergence,
               monitor_floor=res.divergence_floor, monitor_excess=res.excess_divergence,
               monitor_agree_wrong=res.agree_wrong, monitor_ran=True,
               ctrl_toggles=ctrl.toggles)

    rep["GATE_A_PASS"] = bool(ok_target and ok_live and ok_exact and ok_rev)
    (out / "gate_a.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\nGATE A: {'PASS' if rep['GATE_A_PASS'] else 'FAIL'}  -> {out / 'gate_a.json'}")
    print("  PASS means the frozen reference is sound and Gate B can measure transfer\n"
          "  through it. It says NOTHING about whether the hook transfers.")


if __name__ == "__main__":
    main()

"""The merged-weight cache must be invisible: every write path to dW must invalidate it.

A missed invalidation would silently serve stale weights, which would look like a real
result. This checks the cached forward against a freshly computed one after every
mutation the controller can perform.

    .venv\\Scripts\\python tests\\test_merge_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdelta.fastweights import GroupConfig, HDeltaLinear, PlasticityController  # noqa: E402


def live_output(m: HDeltaLinear, x: torch.Tensor) -> torch.Tensor:
    """(W + dW) x computed from scratch, bypassing the cache entirely."""
    w = m.weight + m.delta.to(m.weight.dtype)
    return torch.nn.functional.linear(x, w, m.bias)


def check(tag: str, mods, x, tol=0.0):
    for k, m in mods.items():
        with torch.no_grad():
            cached = m(x)
        fresh = live_output(m, x)
        d = (cached - fresh).abs().max().item()
        assert d <= tol, f"{tag}: stale cache in {k}, max deviation {d}"
    return True


def main():
    torch.manual_seed(0)
    dev = "cpu"
    mods = {}
    for i in range(3):
        base = torch.nn.Linear(16, 16, bias=False).to(dev)
        mods[f"m{i}"] = HDeltaLinear(base, f"m{i}", "g").to(dev)
    groups = {"g": GroupConfig(name="g", patterns=[], layers=[], eta=1e-3, decay=0.01,
                               budget=0.05, homeo_mode="global")}
    ctrl = PlasticityController(mods, groups, postsynaptic="hebbian", update_norm="adam")
    x = torch.randn(4, 16, device=dev)

    check("initial (dW = 0)", mods, x)

    # 1) a learning update
    ctrl.set_record(True)
    for m in mods.values():
        m(x)
    hidden = torch.ones(4, dtype=torch.bool)
    ctrl.update_from_batch(None, hidden, n_puzzles_for_decay=0)
    ctrl.set_record(False)
    assert any(m.delta.abs().sum() > 0 for m in mods.values()), "update wrote nothing"
    check("after update_from_batch", mods, x)

    # 2) passive decay
    before = {k: m.delta.clone() for k, m in mods.items()}
    ctrl.decay(5)
    assert any((before[k] - m.delta).abs().sum() > 0 for k, m in mods.items()), "decay did nothing"
    check("after decay", mods, x)

    # 3) homeostatic clamp, forced to bite
    with torch.no_grad():
        for m in mods.values():
            m.delta.mul_(500.0)
            m.invalidate()
    ctrl.clamp_all()
    check("after homeostatic clamp", mods, x)

    # 4) anti-Hebbian REM step
    ctrl.set_record(True)
    for m in mods.values():
        m(x)
    ctrl.anti_hebbian(eta_scale=1.0, default_eta=0.3)
    ctrl.set_record(False)
    check("after anti_hebbian", mods, x)

    # 5) snapshot / restore
    snap = ctrl.snapshot()
    with torch.no_grad():
        for m in mods.values():
            m.delta.add_(torch.randn_like(m.delta))
            m.invalidate()
    _ = [m(x) for m in mods.values()]          # populate the cache with the perturbed state
    ctrl.restore(snap)
    check("after restore", mods, x)

    # 6) reset
    ctrl.reset()
    check("after reset", mods, x)
    for m in mods.values():
        assert m.delta.abs().sum() == 0

    # 7) gradients must still reach delta (the cache must not be used under grad)
    ctrl.reset()
    y = mods["m0"](x)
    loss = y.pow(2).sum()
    g = torch.autograd.grad(loss, mods["m0"].delta, allow_unused=True)[0]
    assert g is not None and g.abs().sum() > 0, "autograd no longer reaches delta"

    print("merge cache ok: identical to live computation on all 7 mutation paths, autograd intact")


if __name__ == "__main__":
    main()

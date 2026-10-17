"""h-delta fast-weight modules and the plasticity controller.

Each ``HDeltaLinear`` wraps a *frozen* ``nn.Linear`` and adds a fast weight:

    y = (W + dW) x + b

``W`` never changes.  ``dW`` (the h-delta module) is updated online while the
stream of experience passes through the network:

    dW_t = (1 - lambda) * dW_{t-1} + eta * (post_t (x) x_t)          (leaky integrator)

``post_t`` is the postsynaptic term and is selectable:

    'error'   : post_t = -dL/dy_t   (error-driven delta rule; obtained with
                autograd restricted to dW, every other parameter is frozen)
    'hebbian' : post_t =  y_t        (pure Hebbian, self-reinforcing, unsupervised)

After every update the controller applies *homeostatic plasticity*: the
Frobenius norm of dW is clamped to a fixed fraction of ||W||_F (global mode) or
each output neuron's row is clamped to a fraction of that row's norm in W
(row mode; "synaptic scaling").

Simulated REM sleep (anti-Hebbian pruning under Gaussian noise) lives in
``rem.py`` and acts on these same modules through ``anti_hebbian``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GroupConfig:
    """Hyper-parameters of one h-delta group (e.g. hdeltaH, hdeltaG)."""
    name: str
    patterns: List[str]                 # module-name patterns with "{L}" placeholder
    layers: List[int]                   # layer indices (negative = from the top)
    eta: float = 0.05                   # learning rate of the fast weight
    decay: float = 0.01                 # lambda: passive decay per *puzzle*
    budget: float = 0.10                # homeostatic cap: ||dW||_F <= budget * ||W||_F
    homeo_mode: str = "global"          # global | row | none
    rem_eta: Optional[float] = None     # anti-Hebbian rate during REM (None -> rem.eta)
    enabled: bool = True                # if False the group never updates (control)


class HDeltaLinear(nn.Module):
    """A frozen linear projection plus an online-updated fast weight."""

    def __init__(self, base: nn.Linear, name: str, group: str):
        super().__init__()
        self.name = name
        self.group = group
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        # the fast weight (h-delta).  Only this tensor ever changes.
        # fp32 master copy of the fast weight even when the frozen weight is bf16
        self.delta = nn.Parameter(torch.zeros(self.weight.shape, dtype=torch.float32,
                                              device=self.weight.device), requires_grad=True)
        self.w_norm = float(self.weight.float().norm())
        self.register_buffer("w_row_norm", self.weight.float().norm(dim=1, keepdim=True).detach().clone())
        self.apply_delta = True     # False -> behaves exactly like the frozen base
        self.record = False         # True -> keep last x / y for Hebbian & REM rules
        self._merged = None         # cached W + dW for no-grad passes; see invalidate()
        self.last_x: Optional[torch.Tensor] = None
        self.last_y: Optional[torch.Tensor] = None

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def invalidate(self) -> None:
        """Drop the cached merged weight.  Must be called after any write to ``delta``."""
        self._merged = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.apply_delta:
            w = self.weight
        elif torch.is_grad_enabled() and self.delta.requires_grad:
            # gradients must reach delta, so build the sum live
            w = self.weight + self.delta.to(self.weight.dtype)
        else:
            # inference: (W + dW) only changes when dW does, so cache it.  Rebuilding it
            # per token made generation cost scale with the size of the plastic slice.
            if self._merged is None:
                self._merged = (self.weight + self.delta.to(self.weight.dtype)).detach()
            w = self._merged
        y = F.linear(x, w, self.bias)
        if self.record:
            self.last_x = x.detach()
            self.last_y = y.detach()
        return y

    # ---- bookkeeping -------------------------------------------------------
    def delta_norm_ratio(self) -> float:
        return float(self.delta.detach().norm() / (self.w_norm + 1e-12))

    @torch.no_grad()
    def homeostatic_clamp(self, budget: float, mode: str) -> float:
        """Clamp the fast weight to the energy budget.  Returns the scale applied."""
        if mode == "none" or budget is None:
            return 1.0
        self.invalidate()
        d = self.delta
        if mode == "global":
            n = d.norm()
            cap = budget * self.w_norm
            if n > cap:
                s = float(cap / (n + 1e-12))
                d.mul_(s)
                return s
            return 1.0
        if mode == "row":
            rn = d.norm(dim=1, keepdim=True)
            cap = budget * self.w_row_norm
            scale = torch.where(rn > cap, cap / rn.clamp_min(1e-12), torch.ones_like(rn))
            d.mul_(scale)
            return float(scale.mean())
        raise ValueError(f"unknown homeo_mode {mode}")


class PlasticityController:
    """Owns all h-delta modules and applies the learning/forgetting rules.

    * ``update_from_batch``  : daytime learning (error-driven or Hebbian) + passive decay
                               + homeostatic clamp.
    * ``decay``              : passive decay alone (lambda per puzzle).
    * ``anti_hebbian``       : used by the REM phase (negative learning rate).
    """

    def __init__(self, modules: Dict[str, HDeltaLinear], groups: Dict[str, GroupConfig],
                 postsynaptic: str = "error", decay_enabled: bool = True,
                 homeo_enabled: bool = True, update_norm: str = "frob"):
        self.modules = modules
        self.groups = groups
        self.postsynaptic = postsynaptic
        self.decay_enabled = decay_enabled
        self.homeo_enabled = homeo_enabled
        # 'none': raw outer-product update (eta in absolute units)
        # 'frob': each experience deposits a fixed quantum of synaptic change:
        #         U <- U / ||U||_F * ||W||_F, so eta is a fraction of ||W||_F per chunk
        # 'adam': per-synapse normalisation with running first/second moments of the
        #         outer-product term (eta in absolute units, e.g. 2e-4)
        self.update_norm = update_norm
        self.updates_enabled = True
        self.n_updates = 0
        self._m: Dict[str, torch.Tensor] = {}
        self._v: Dict[str, torch.Tensor] = {}
        self.betas = (0.9, 0.99)

    @torch.no_grad()
    def _normalise(self, k: str, u: torch.Tensor, m: HDeltaLinear) -> torch.Tensor:
        if self.update_norm == "none":
            return u
        if self.update_norm == "frob":
            return u * (m.w_norm / (float(u.norm()) + 1e-12))
        if self.update_norm == "adam":
            b1, b2 = self.betas
            if k not in self._m:
                self._m[k] = torch.zeros_like(u)
                self._v[k] = torch.zeros_like(u)
            self._m[k].mul_(b1).add_(u, alpha=1 - b1)
            self._v[k].mul_(b2).addcmul_(u, u, value=1 - b2)
            t = self.n_updates + 1
            mh = self._m[k] / (1 - b1 ** t)
            vh = self._v[k] / (1 - b2 ** t)
            return mh / (vh.sqrt() + 1e-8)
        raise ValueError(f"unknown update_norm {self.update_norm}")

    @torch.no_grad()
    def reset_moments(self) -> None:
        self._m.clear()
        self._v.clear()

    # ---- helpers -----------------------------------------------------------
    def deltas(self) -> List[torch.Tensor]:
        return [m.delta for m in self.modules.values()]

    def set_apply_delta(self, flag: bool) -> None:
        for m in self.modules.values():
            m.apply_delta = flag

    def set_record(self, flag: bool) -> None:
        for m in self.modules.values():
            m.record = flag
            if not flag:
                m.last_x = None
                m.last_y = None

    @torch.no_grad()
    def invalidate(self) -> None:
        for m in self.modules.values():
            m.invalidate()

    @torch.no_grad()
    def reset(self) -> None:
        for m in self.modules.values():
            m.delta.zero_()
            m.invalidate()

    @torch.no_grad()
    def snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: m.delta.detach().clone() for k, m in self.modules.items()}

    @torch.no_grad()
    def restore(self, snap: Dict[str, torch.Tensor]) -> None:
        for k, m in self.modules.items():
            m.delta.copy_(snap[k])
            m.invalidate()

    def plastic_param_count(self) -> int:
        return sum(m.delta.numel() for m in self.modules.values())

    # ---- the rules ---------------------------------------------------------
    @torch.no_grad()
    def decay(self, n_puzzles: float) -> None:
        """Passive decay: dW <- (1 - lambda)^n dW  (lambda is per puzzle)."""
        if not self.decay_enabled or n_puzzles <= 0:
            return
        for m in self.modules.values():
            g = self.groups[m.group]
            if g.decay > 0:
                m.delta.mul_((1.0 - g.decay) ** n_puzzles)
                m.invalidate()

    @torch.no_grad()
    def clamp_all(self) -> Dict[str, float]:
        scales = {}
        for k, m in self.modules.items():
            g = self.groups[m.group]
            if self.homeo_enabled:
                scales[k] = m.homeostatic_clamp(g.budget, g.homeo_mode)
            else:
                scales[k] = 1.0
        return scales

    def update_from_batch(self, loss: Optional[torch.Tensor], hidden_mask: torch.Tensor,
                          n_puzzles_for_decay: float) -> Dict[str, float]:
        """One online plasticity step after a forward pass over a chunk of the stream.

        loss        : next-token loss over the response span (needed for 'error').
        hidden_mask : (B, T) bool, True at hidden positions whose output is scored.
        Returns per-module update norms (for logging).
        """
        stats: Dict[str, float] = {}
        if not self.updates_enabled:
            return stats
        # 1) postsynaptic term -> update matrix U = mean_t post_t (x) x_t
        updates: Dict[str, torch.Tensor] = {}
        if self.postsynaptic == "error":
            grads = torch.autograd.grad(loss, self.deltas(), allow_unused=True)
            for (k, m), g in zip(self.modules.items(), grads):
                if g is None:
                    continue
                updates[k] = -g.float()  # delta rule: post = -dL/dy
        elif self.postsynaptic == "hebbian":
            for k, m in self.modules.items():
                if m.last_x is None:
                    continue
                x = m.last_x[hidden_mask]     # (N, d_in)
                y = m.last_y[hidden_mask]     # (N, d_out)
                n = max(1, x.shape[0])
                updates[k] = (y.float().T @ x.float()) / n     # (d_out, d_in)
        else:
            raise ValueError(f"unknown postsynaptic rule {self.postsynaptic}")

        # 2) leaky integration + homeostasis
        with torch.no_grad():
            self.decay(n_puzzles_for_decay)
            for k, u in updates.items():
                m = self.modules[k]
                g = self.groups[m.group]
                if not g.enabled:
                    continue
                stats[f"upd_norm/{k}"] = float(u.norm())
                u = self._normalise(k, u, m)
                m.delta.add_(g.eta * u.to(m.delta.dtype))
                m.invalidate()
            self.clamp_all()
        self.n_updates += 1
        return stats

    @torch.no_grad()
    def anti_hebbian(self, eta_scale: float = 1.0, default_eta: float = 0.2,
                     postsynaptic: str = "delta", threshold: float = 0.0) -> Dict[str, float]:
        """REM rule: dW <- dW - eta_rem * mean_t (post_t (x) x_t) / mean_t ||x_t||^2.

        post_t = dW x_t      ('delta': only the fast weight's own response is penalised)
               = (W+dW) x_t  ('full' : the whole output; aggressive, fights the base model)
        The normalisation by mean ||x||^2 makes eta_rem the *fraction* of the
        noise-responsive component removed per step (in the isotropic limit).
        Modules whose noise-response ratio is below ``threshold`` are left alone.
        Returns the per-module noise-response ratio measured *before* the update
        (with eta_scale=0 this is a pure measurement).
        """
        ratios: Dict[str, float] = {}
        for k, m in self.modules.items():
            if m.last_x is None:
                continue
            g = self.groups[m.group]
            eta = (g.rem_eta if g.rem_eta is not None else default_eta) * eta_scale
            x = m.last_x.reshape(-1, m.in_features).float()
            resp = x @ m.delta.float().T                        # (N, d_out) fast-weight response
            base = x @ m.weight.float().T                       # frozen response
            ratio = float(resp.pow(2).mean() / (base.pow(2).mean() + 1e-12))
            ratios[k] = ratio
            if not g.enabled or eta == 0.0 or ratio < threshold:
                continue
            post = resp if postsynaptic == "delta" else (resp + base)
            n = x.shape[0]
            u = (post.T @ x) / n                                # (d_out, d_in)
            u = u / (x.pow(2).sum(dim=1).mean() + 1e-12)
            m.delta.sub_(eta * u.to(m.delta.dtype))
            m.invalidate()
        if eta_scale != 0.0:
            self.clamp_all()
        return ratios

    # ---- logging -----------------------------------------------------------
    def state(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        per_group: Dict[str, List[float]] = {}
        for k, m in self.modules.items():
            r = m.delta_norm_ratio()
            out[f"fw_norm_ratio/{k}"] = r
            per_group.setdefault(m.group, []).append(r)
        for gname, vals in per_group.items():
            out[f"fw_norm_ratio_group/{gname}"] = sum(vals) / len(vals)
        return out

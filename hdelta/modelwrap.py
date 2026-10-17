"""Load a small open-weights LM, freeze it, and splice in h-delta modules."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fastweights import GroupConfig, HDeltaLinear, PlasticityController


def load_model(name: str, dtype: str = "float32", device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    for p in model.parameters():           # freeze everything
        p.requires_grad_(False)
    return model, tok


def _resolve_layer(idx: int, n_layers: int) -> int:
    return idx if idx >= 0 else n_layers + idx


def resolve_targets(model: nn.Module, groups: Dict[str, GroupConfig]) -> List[Tuple[str, str]]:
    """Expand group patterns into concrete (module_name, group_name) pairs."""
    n_layers = int(model.config.num_hidden_layers)
    named = dict(model.named_modules())
    targets: List[Tuple[str, str]] = []
    for g in groups.values():
        for layer in g.layers:
            li = _resolve_layer(int(layer), n_layers)
            for pat in g.patterns:
                name = pat.format(L=li)
                if name not in named:
                    raise KeyError(f"module '{name}' not found in {type(model).__name__}; "
                                   f"available leaf linears e.g. "
                                   f"{[n for n, m in named.items() if isinstance(m, nn.Linear)][:8]}")
                targets.append((name, g.name))
    return targets


def install_hdelta(model: nn.Module, groups: Dict[str, GroupConfig], max_fraction: float = 0.05,
                   postsynaptic: str = "error", decay_enabled: bool = True,
                   homeo_enabled: bool = True, update_norm: str = "frob") -> Tuple[PlasticityController, dict]:
    """Replace the targeted nn.Linear projections with HDeltaLinear modules."""
    total = sum(p.numel() for p in model.parameters())
    modules: Dict[str, HDeltaLinear] = {}
    for name, gname in resolve_targets(model, groups):
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name} is {type(base).__name__}, expected nn.Linear")
        hd = HDeltaLinear(base, name, gname).to(base.weight.device)
        setattr(parent, attr, hd)
        modules[name] = hd
    plastic = sum(m.delta.numel() for m in modules.values())
    frac = plastic / total
    if frac > max_fraction:
        raise ValueError(f"plastic fraction {frac:.3%} exceeds max_fraction {max_fraction:.1%}")
    ctrl = PlasticityController(modules, groups, postsynaptic=postsynaptic,
                                decay_enabled=decay_enabled, homeo_enabled=homeo_enabled,
                                update_norm=update_norm)
    info = {
        "total_params": total,
        "plastic_params": plastic,
        "plastic_fraction": frac,
        "frozen_fraction": 1.0 - frac,
        "modules": {k: {"group": m.group, "shape": list(m.weight.shape)} for k, m in modules.items()},
    }
    return ctrl, info


def groups_from_config(cfg: dict) -> Dict[str, GroupConfig]:
    out: Dict[str, GroupConfig] = {}
    for name, g in cfg.items():
        out[name] = GroupConfig(name=name, **g)
    return out

"""Simulated REM sleep: anti-Hebbian pruning under Gaussian noise.

The model is taken "offline": instead of token embeddings it receives Gaussian
noise whose per-dimension statistics match the embedding table.  The frozen
network transforms that noise into hidden states; every h-delta module records
its input x and the response of its fast weight dW x.

A loop that has become pathologically rigid fires strongly even for this
structureless input (large ||dW x|| relative to ||W x||).  With the learning rate
flipped negative the loop is penalised in proportion to how strongly it fires:

    dW <- dW - eta_rem * mean_t (dW x_t)(x_t)^T / mean_t ||x_t||^2

Selective loops (which barely respond to noise) are almost untouched.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from .fastweights import PlasticityController


@torch.no_grad()
def embedding_std(model) -> torch.Tensor:
    w = model.get_input_embeddings().weight
    return w.float().std(dim=0)


@torch.no_grad()
def noise_forward(model, ctrl: PlasticityController, batch: int, seq_len: int,
                  emb_std: torch.Tensor, generator: torch.Generator | None = None) -> None:
    d = emb_std.shape[0]
    device = emb_std.device
    noise = torch.randn((batch, seq_len, d), device=device, generator=generator) * emb_std
    noise = noise.to(next(model.parameters()).dtype)
    ctrl.set_record(True)
    model(inputs_embeds=noise)


@torch.no_grad()
def measure_noise_response(model, ctrl: PlasticityController, rem_cfg: dict,
                           emb_std: torch.Tensor, generator=None) -> Dict[str, float]:
    """Noise-response ratio per module without changing anything."""
    noise_forward(model, ctrl, rem_cfg.get("batch", 8), rem_cfg.get("seq_len", 32), emb_std, generator)
    ratios = ctrl.anti_hebbian(eta_scale=0.0)
    ctrl.set_record(False)
    return ratios


@torch.no_grad()
def rem_phase(model, ctrl: PlasticityController, rem_cfg: dict, emb_std: torch.Tensor,
              generator=None) -> Dict[str, object]:
    """Run ``steps`` offline noise batches with anti-Hebbian updates."""
    steps = int(rem_cfg.get("steps", 10))
    hist: List[Dict[str, float]] = []
    for _ in range(steps):
        noise_forward(model, ctrl, rem_cfg.get("batch", 8), rem_cfg.get("seq_len", 32), emb_std, generator)
        ratios = ctrl.anti_hebbian(eta_scale=1.0, default_eta=float(rem_cfg.get("eta", 0.2)),
                                   postsynaptic=rem_cfg.get("postsynaptic", "delta"),
                                   threshold=float(rem_cfg.get("threshold", 0.0)))
        hist.append(ratios)
    ctrl.set_record(False)
    after = measure_noise_response(model, ctrl, rem_cfg, emb_std, generator)
    return {"before": hist[0] if hist else {}, "after": after, "steps": steps}

"""Scoring primitives: teacher-forced exact match, greedy generation, LM perplexity, cloze."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .encoding import Batch, Encoder


def forward_logits(model, batch: Batch) -> torch.Tensor:
    """Logits for the last `batch.keep` positions only (the response span); the vocabulary
    projection over the prompt is skipped, which is the dominant cost for large vocabularies."""
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                position_ids=batch.position_ids, logits_to_keep=batch.keep)
    return out.logits


def _aligned(logits: torch.Tensor, batch: Batch):
    """Next-token alignment for kept logits: position T-k+j predicts token T-k+j+1."""
    k = logits.shape[1]
    return logits[:, :-1], batch.labels[:, -k + 1:]


def response_loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Mean next-token cross-entropy over the response span only."""
    lg, tg = _aligned(logits, batch)
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1), ignore_index=-100)


def signed_response_loss(logits: torch.Tensor, batch: Batch,
                         signs: torch.Tensor, include: torch.Tensor | None = None) -> torch.Tensor:
    """Token-pooled response loss with a per-item SIGN: (sum_pos CE - lam * sum_neg CE) / n_pos.

    Built to add exactly one thing to the baseline - a negative write on the model's own wrong
    answers - and nothing else. The denominator is the token count of the POSITIVE items only,
    which is what `response_loss` divides by when the batch holds only positive items. So:

      - with no negative items this is `response_loss`, arithmetically
      - a negative item's tokens are pushed down with the same per-token strength (times lam)
        that a positive item's are pushed up

    `signs` is (B,): +1.0 for items to reinforce, -lam for items to suppress.

    `include` is (B, R), in response-token order: which tokens of each item count. It is what
    makes pain point at the mistake. A wrong conjunction trace reads
    "Step 1: zone south with cluster delta -> #q1\\nBucket: #q1\\n" against a correct
    "... -> #q6\\nBucket: #q6\\n" - twenty tokens each, differing at exactly two positions, the
    two label digits. Everything else is the model correctly restating its input and correctly
    formatting its answer. Punishing all of it teaches blanket aversion to reasoning and format
    that were right, and pushing a model off its output format is precisely how the world runs
    degenerated. So a negative item includes only the tokens where it differs from the correct
    demonstration. Positive items include everything, so the positive term and the
    denominator are untouched and the no-pain case is still the baseline exactly.
    """
    lg, tg = _aligned(logits, batch)
    per_tok = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                              ignore_index=-100, reduction="none").view(tg.shape)
    keep = (tg != -100).float()
    s = signs.to(per_tok.device, per_tok.dtype)
    den = (keep.sum(1) * (s > 0).float()).sum().clamp_min(1.0)
    if include is not None:
        inc = include.to(per_tok.device).float()
        rank = (keep.cumsum(1) - 1).clamp(0, inc.shape[1] - 1).long()   # 0-based response index
        keep = keep * inc.gather(1, rank)
    num = ((per_tok * keep).sum(1) * s).sum()
    return num / den


def weighted_response_loss(logits: torch.Tensor, batch: Batch,
                           weights: torch.Tensor) -> torch.Tensor:
    """Per-item weighted response loss: sum_i w_i * mean-CE(item i).

    The plasticity step takes ``post = -dL/dy``, so scaling item i's loss by w_i scales its
    contribution to the outer product by w_i - and a NEGATIVE w_i flips that item's gradient,
    which is an anti-Hebbian update on exactly that item. Reinforcement and damage therefore
    come through one path with no second code mode.

    Note this is a single outer-product update, not iterative descent: the loss value is never
    minimised, only differentiated once. So a negative weight does not "diverge to -inf" the
    way it would in a training loop. The risk is magnitude, which ``update_norm: adam``
    normalises per synapse.

    ``weights`` is (B,). Note this normalises PER ITEM and then weights, where
    ``response_loss`` pools every response token in the batch: with ``w = 1/B`` the two agree
    only when items have equal response length. Per-item is the right normalisation here -
    an advantage attaches to a decision, not to each token of its spelling, and pooling would
    silently weight long answers more.
    """
    lg, tg = _aligned(logits, batch)
    per_tok = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                              ignore_index=-100, reduction="none").view(tg.shape)
    keep = (tg != -100).float()
    per_item = (per_tok * keep).sum(1) / keep.sum(1).clamp_min(1.0)
    return (per_item * weights.to(per_item.device, per_item.dtype)).sum()


@torch.no_grad()
def teacher_forced(model, batch: Batch) -> Tuple[List[bool], List[float], List[float]]:
    """Per-sequence exact match (== greedy-decode exact match), mean NLL, token accuracy."""
    logits = forward_logits(model, batch)
    lg, tgt = _aligned(logits, batch)
    pred = lg.argmax(-1)
    m = tgt != -100
    hit = (pred == tgt) & m
    m_ex = m.clone()
    m_ex[:, -1] = False                      # the trailing newline (left padding: last column)
    exact = (hit | ~m_ex).all(dim=1).tolist()
    tokacc = (hit.sum(1) / m.sum(1).clamp_min(1)).tolist()
    lg = lg.float()
    nll_tok = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                              ignore_index=-100, reduction="none").view(tgt.shape)
    nll = (nll_tok * m).sum(1) / m.sum(1).clamp_min(1)
    return exact, nll.tolist(), tokacc


@torch.no_grad()
def score_pairs(model, enc: Encoder, prompts: Sequence[str], responses: Sequence[str],
                chunk: int = 16) -> Tuple[List[bool], List[float], List[float]]:
    ex_all, nll_all, ta_all = [], [], []
    for i in range(0, len(prompts), chunk):
        b = enc.encode_pairs(prompts[i:i + chunk], responses[i:i + chunk])
        ex, nll, ta = teacher_forced(model, b)
        ex_all += ex
        nll_all += nll
        ta_all += ta
    return ex_all, nll_all, ta_all


@torch.no_grad()
def generate(model, enc: Encoder, prompts: Sequence[str], max_new_tokens: int,
             chunk: int = 16) -> List[str]:
    """Greedy generation, stopped at newline, returned as stripped strings."""
    outs: List[str] = []
    for i in range(0, len(prompts), chunk):
        ids, attn = enc.encode_prompts_left(prompts[i:i + chunk])
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1, eos_token_id=enc.newline_id,
                             pad_token_id=enc.pad, temperature=None, top_p=None, top_k=None)
        new = gen[:, ids.shape[1]:]
        for row in new:
            text = enc.tok.decode(row, skip_special_tokens=True)
            outs.append(text.split("\n")[0].strip())
    return outs


@torch.no_grad()
def generate_text(model, enc: Encoder, prompts: Sequence[str], max_new_tokens: int,
                  chunk: int = 16) -> List[str]:
    """Greedy open-ended generation, cut at the first blank line, returned raw."""
    outs: List[str] = []
    for i in range(0, len(prompts), chunk):
        ids, attn = enc.encode_prompts_left(prompts[i:i + chunk])
        gen = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1, pad_token_id=enc.pad,
                             temperature=None, top_p=None, top_k=None)
        for row in gen[:, ids.shape[1]:]:
            outs.append(enc.tok.decode(row, skip_special_tokens=True).split("\n\n")[0])
    return outs


def trim_to_answer(task, text: str) -> str:
    """Keep the generated text up to and including its Answer line (with trailing newline)."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("Step"):
            continue
        # the first non-step line is the answer line, labelled ("Answer: ...") or not
        return "\n".join(lines[:i + 1]) + "\n"
    return text.rstrip("\n") + "\n"


@torch.no_grad()
def free_generation(model, enc: Encoder, task, puzzles, rule_text: bool = False,
                    max_new_tokens: int = 48, shots=()) -> dict:
    """Open-ended probe after 'Input: x': does the model think first, and is it right?
    Returns accuracy, mean Step lines emitted before the answer, mean tokens spent, and
    the raw generations."""
    prompts = [task.make_pair(p, "free", rule_text=rule_text, shots=shots)[0] for p in puzzles]
    texts = generate_text(model, enc, prompts, max_new_tokens)
    accs, steps, toks = [], [], []
    for text, p in zip(texts, puzzles):
        ans, n_steps = task.parse_generation(text)
        accs.append(ans == p.answer)
        steps.append(n_steps)
        toks.append(len(enc.ids(trim_to_answer(task, text))))
    n = max(1, len(puzzles))
    return {"acc": sum(accs) / n, "steps": sum(steps) / n, "tokens": sum(toks) / n,
            "texts": texts, "correct": accs}


@torch.no_grad()
def lm_perplexity(model, enc: Encoder, text: str, window: int = 256, max_windows: int = 24) -> float:
    ids = enc.ids(text)
    windows = [ids[i:i + window] for i in range(0, len(ids) - window + 1, window)][:max_windows]
    if not windows:
        windows = [ids]
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(windows), 8):
        chunk = windows[i:i + 8]
        T = max(len(w) for w in chunk)
        x = torch.full((len(chunk), T), enc.pad, dtype=torch.long)
        a = torch.zeros((len(chunk), T), dtype=torch.long)
        for j, w in enumerate(chunk):
            x[j, :len(w)] = torch.tensor(w)
            a[j, :len(w)] = 1
        x, a = x.to(enc.device), a.to(enc.device)
        logits = model(input_ids=x, attention_mask=a).logits[:, :-1].float()
        tgt = x[:, 1:].clone()
        tgt[a[:, 1:] == 0] = -100
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                              ignore_index=-100, reduction="sum")
        total_nll += float(nll)
        total_tok += int((tgt != -100).sum())
    return math.exp(total_nll / max(1, total_tok))


@torch.no_grad()
def cloze_accuracy(model, enc: Encoder, items: Sequence[dict]) -> float:
    """Top-1 accuracy on the first token of the expected continuation."""
    prompts = [it["prompt"] for it in items]
    ids, attn = enc.encode_prompts_left(prompts)
    logits = model(input_ids=ids, attention_mask=attn, logits_to_keep=1).logits[:, -1]
    pred = logits.argmax(-1).tolist()
    hits = 0
    for p, it in zip(pred, items):
        target = enc.ids(" " + it["answer"].strip())[0]
        hits += int(p == target)
    return hits / max(1, len(items))

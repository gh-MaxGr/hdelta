"""Tokenisation of (prompt, response) pairs into scored batches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch


@dataclass
class Batch:
    input_ids: torch.Tensor        # (B, T), LEFT-padded so every response ends at position T-1
    attention_mask: torch.Tensor   # (B, T)
    labels: torch.Tensor           # (B, T), -100 on prompt + padding
    hidden_mask: torch.Tensor      # (B, T) bool: position t is scored if labels[t+1] != -100
    n_scored: int
    position_ids: torch.Tensor     # (B, T) explicit positions (left padding)
    keep: int                      # only the last `keep` positions need logits (max response + 1)

    def to(self, device):
        return Batch(self.input_ids.to(device), self.attention_mask.to(device),
                     self.labels.to(device), self.hidden_mask.to(device), self.n_scored,
                     self.position_ids.to(device), self.keep)


class Encoder:
    """Tokenises prompt/response pairs. Prompt and response are tokenised separately so
    the response boundary is exact (all task formats break at '\\n' or ':' so BPE
    merges never cross it; verified by ``check_boundary``)."""

    def __init__(self, tok, device: str):
        self.tok = tok
        self.device = device
        self.bos = tok.bos_token_id if tok.bos_token_id is not None else None
        self.pad = tok.pad_token_id
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        self.newline_id = nl[-1]

    def ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def check_boundary(self, prompt: str, response: str) -> bool:
        joint = self.ids(prompt + response)
        sep = self.ids(prompt) + self.ids(response)
        return joint == sep

    def encode_pairs(self, prompts: Sequence[str], responses: Sequence[str]) -> Batch:
        seqs, starts = [], []
        for p, r in zip(prompts, responses):
            p_ids = self.ids(p)
            r_ids = self.ids(r)
            prefix = [self.bos] if self.bos is not None else []
            seqs.append(prefix + p_ids + r_ids)
            starts.append(len(prefix) + len(p_ids))
        T = max(len(s) for s in seqs)
        B = len(seqs)
        input_ids = torch.full((B, T), self.pad, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        keep = 1
        for i, (s, st) in enumerate(zip(seqs, starts)):
            off = T - len(s)
            input_ids[i, off:] = torch.tensor(s)
            attn[i, off:] = 1
            labels[i, off + st:] = torch.tensor(s[st:])
            keep = max(keep, len(s) - st + 1)
        hidden = torch.zeros((B, T), dtype=torch.bool)
        hidden[:, :-1] = labels[:, 1:] != -100
        pos = (attn.cumsum(-1) - 1).clamp_min(0)
        return Batch(input_ids, attn, labels, hidden, int(hidden.sum()), pos, min(keep, T)).to(self.device)

    def encode_prompts_left(self, prompts: Sequence[str]):
        """Left-padded prompts for batched greedy generation."""
        seqs = []
        for p in prompts:
            prefix = [self.bos] if self.bos is not None else []
            seqs.append(prefix + self.ids(p))
        T = max(len(s) for s in seqs)
        B = len(seqs)
        input_ids = torch.full((B, T), self.pad, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, T - len(s):] = torch.tensor(s)
            attn[i, T - len(s):] = 1
        return input_ids.to(self.device), attn.to(self.device)

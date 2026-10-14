"""Frozen reference-LM scoring: total NLL and token counts per text."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class TextScore:
    nll_nats: float
    n_tokens: int

    @property
    def bits(self) -> float:
        return self.nll_nats / math.log(2)

    @property
    def ppl(self) -> float:
        return math.exp(self.nll_nats / self.n_tokens) if self.n_tokens else 1.0


class ReferenceScorer(Protocol):
    def score(self, texts: list[str]) -> list[TextScore]: ...


class RefLM:
    """Scores texts unconditionally under the frozen base model (adapters disabled)."""

    BOS = "<|endoftext|>"

    def __init__(self, model, tokenizer, batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.bos_id = tokenizer.convert_tokens_to_ids(self.BOS)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.bos_id

    def _context(self):
        return self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else contextlib.nullcontext()

    @torch.no_grad()
    def score(self, texts: list[str]) -> list[TextScore]:
        ids = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        out: list[TextScore] = []
        device = next(self.model.parameters()).device
        with self._context():
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                seqs = [[self.bos_id] + list(s) for s in chunk]
                width = max(len(s) for s in seqs)
                input_ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(seqs), width), dtype=torch.long)
                for i, s in enumerate(seqs):
                    input_ids[i, : len(s)] = torch.tensor(s)
                    mask[i, : len(s)] = 1
                input_ids, mask = input_ids.to(device), mask.to(device)
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits.float()
                logp = torch.log_softmax(logits[:, :-1], dim=-1)
                target = input_ids[:, 1:]
                tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
                nll = (-tok_lp.sum(dim=1)).tolist()
                counts = mask[:, 1:].sum(dim=1).tolist()
                out.extend(TextScore(float(n), int(c)) for n, c in zip(nll, counts))
        return out

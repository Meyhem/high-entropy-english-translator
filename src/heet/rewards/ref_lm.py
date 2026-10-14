"""Frozen reference-LM scoring: total NLL and token counts per text."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


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

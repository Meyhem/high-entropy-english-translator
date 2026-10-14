"""Reconstruction-fidelity measures."""

from __future__ import annotations

import re
from typing import Protocol

from sacrebleu.metrics import CHRF

_CHRF = CHRF()


class Embedder(Protocol):
    def similarity(self, a: list[str], b: list[str]) -> list[float]: ...


def normalize(text: str) -> str:
    text = " ".join(text.lower().split())
    return re.sub(r"^[\W_]+|[\W_]+$", "", text)


def exact_match(a: str, b: str) -> float:
    return 1.0 if normalize(a) == normalize(b) else 0.0


def chrf(hyp: str, ref: str) -> float:
    return _CHRF.sentence_score(hyp, [ref]).score / 100.0

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


class SentenceEmbedder:
    def __init__(self, name: str, device: str = "cpu", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.batch_size = batch_size

    def similarity(self, a: list[str], b: list[str]) -> list[float]:
        if not a:
            return []
        ea = self.model.encode(a, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True)
        eb = self.model.encode(b, batch_size=self.batch_size, normalize_embeddings=True, convert_to_tensor=True)
        return (ea * eb).sum(dim=-1).clamp(-1.0, 1.0).tolist()

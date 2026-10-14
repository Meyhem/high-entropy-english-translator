"""Deterministic stand-ins for the embedding model and reference LM."""

from difflib import SequenceMatcher

from heet.rewards.ref_lm import TextScore


class StubEmbedder:
    def similarity(self, a: list[str], b: list[str]) -> list[float]:
        return [SequenceMatcher(None, x.lower(), y.lower()).ratio() for x, y in zip(a, b, strict=True)]


class StubRef:
    """2 nats per whitespace token; texts containing 'zzz' are 'unreadable' (8 nats/token)."""

    def __init__(self):
        self.calls = 0

    def score(self, texts: list[str]) -> list[TextScore]:
        self.calls += 1
        out = []
        for t in texts:
            n = len(t.split())
            per = 8.0 if "zzz" in t else 2.0
            out.append(TextScore(nll_nats=per * n, n_tokens=n))
        return out

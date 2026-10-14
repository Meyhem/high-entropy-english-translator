"""Weighted combination of fidelity, length and readability terms."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from heet.config import RewardConfig
from heet.rewards.fidelity import Embedder, chrf, exact_match
from heet.rewards.length import bits_ratio, char_ratio
from heet.rewards.readability import ppl_log_ratio, readability_penalty
from heet.rewards.ref_lm import ReferenceScorer, TextScore


@dataclass
class RewardBreakdown:
    sim: float = 0.0
    exact: float = 0.0
    chrf: float = 0.0
    char_ratio: float = 0.0
    bits_ratio: float = 0.0
    bits_dense: float = 0.0
    bits_orig: float = 0.0
    ppl_dense: float = 1.0
    ppl_orig: float = 1.0
    ppl_log_ratio: float = 0.0
    read_pen: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


class Scorer:
    def __init__(self, cfg: RewardConfig, embedder: Embedder, ref: ReferenceScorer | None):
        self.cfg = cfg
        self.embedder = embedder
        self.ref = ref

    def score(
        self,
        originals: list[str],
        denses: list[str],
        recons: list[str],
        compute_all: bool = True,
    ) -> list[RewardBreakdown]:
        c = self.cfg
        n = len(originals)
        assert len(denses) == n and len(recons) == n
        sims = self.embedder.similarity(originals, recons)
        need_ref = compute_all or c.w_bits != 0 or c.w_read != 0
        if need_ref and self.ref is None:
            raise ValueError("a reference LM is required for bits/readability terms")
        if need_ref:
            ref_scores = self.ref.score(list(denses) + list(originals))
            dense_s, orig_s = ref_scores[:n], ref_scores[n:]
        else:
            dense_s = orig_s = [TextScore(0.0, 0)] * n

        out = []
        for i in range(n):
            b = RewardBreakdown(sim=float(sims[i]), exact=exact_match(originals[i], recons[i]))
            if compute_all or c.w_chrf != 0:
                b.chrf = chrf(recons[i], originals[i])
            b.char_ratio = char_ratio(denses[i], originals[i])
            if need_ref:
                b.bits_dense, b.bits_orig = dense_s[i].bits, orig_s[i].bits
                b.bits_ratio = bits_ratio(dense_s[i], orig_s[i])
                b.ppl_dense, b.ppl_orig = dense_s[i].ppl, orig_s[i].ppl
                b.ppl_log_ratio = ppl_log_ratio(dense_s[i], orig_s[i])
                b.read_pen = readability_penalty(dense_s[i], orig_s[i], c.margin)
            b.total = (
                c.w_sim * b.sim + c.w_exact * b.exact + c.w_chrf * b.chrf
                - c.w_char * b.char_ratio - c.w_bits * b.bits_ratio - c.w_read * b.read_pen
            )
            out.append(b)
        return out

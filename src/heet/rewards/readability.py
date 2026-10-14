"""Readability penalty: how much less predictable the dense text is than the original."""

import math

from heet.rewards.ref_lm import TextScore


def ppl_log_ratio(dense: TextScore, original: TextScore) -> float:
    return math.log(dense.ppl) - math.log(original.ppl)


def readability_penalty(dense: TextScore, original: TextScore, margin: float) -> float:
    return max(0.0, ppl_log_ratio(dense, original) - margin)

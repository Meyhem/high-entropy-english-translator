"""Length measures: characters and reference-LM bits."""

from heet.rewards.ref_lm import TextScore


def char_ratio(dense: str, original: str) -> float:
    return len(dense) / len(original) if original else 0.0


def bits_ratio(dense: TextScore, original: TextScore) -> float:
    return dense.bits / original.bits if original.bits > 0 else 0.0

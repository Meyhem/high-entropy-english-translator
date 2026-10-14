import math

import pytest

from heet.config import RewardConfig
from heet.rewards.combined import Scorer
from heet.rewards.fidelity import chrf, exact_match, normalize
from heet.rewards.length import bits_ratio, char_ratio
from heet.rewards.readability import ppl_log_ratio, readability_penalty
from heet.rewards.ref_lm import TextScore
from tests.stubs import StubEmbedder, StubRef


def test_textscore_bits_and_ppl():
    s = TextScore(nll_nats=4 * math.log(2), n_tokens=2)
    assert math.isclose(s.bits, 4.0)
    assert math.isclose(s.ppl, 4.0)
    assert TextScore(0.0, 0).ppl == 1.0 and TextScore(0.0, 0).bits == 0.0


def test_normalize_and_exact():
    assert normalize("  The Cat  sat. ") == "the cat sat"
    assert exact_match("The cat sat.", "the cat sat") == 1.0
    assert exact_match("The cat sat.", "the dog sat") == 0.0


def test_chrf_range():
    assert math.isclose(chrf("the cat sat", "the cat sat"), 1.0)
    assert 0.0 <= chrf("xyz", "the cat sat") < 0.3


def test_length_ratios():
    assert char_ratio("abc", "abcdef") == 0.5
    assert char_ratio("abc", "") == 0.0
    assert bits_ratio(TextScore(math.log(2), 1), TextScore(2 * math.log(2), 2)) == 0.5
    assert bits_ratio(TextScore(1.0, 1), TextScore(0.0, 0)) == 0.0


def test_readability_margin():
    orig = TextScore(nll_nats=2.0 * 10, n_tokens=10)       # ppl e^2
    same = TextScore(nll_nats=2.0 * 5, n_tokens=5)          # ppl e^2
    worse = TextScore(nll_nats=4.0 * 5, n_tokens=5)         # ppl e^4
    assert ppl_log_ratio(same, orig) == 0.0
    assert readability_penalty(same, orig, margin=math.log(2)) == 0.0
    assert math.isclose(readability_penalty(worse, orig, margin=math.log(2)), 2.0 - math.log(2))


def test_combined_total_matches_formula():
    cfg = RewardConfig(w_chrf=0.5)
    s = Scorer(cfg, StubEmbedder(), StubRef())
    [b] = s.score(["the big cat sat on the mat"], ["big cat sat on mat"], ["the big cat sat on the mat"])
    assert b.sim == 1.0 and b.exact == 1.0 and math.isclose(b.chrf, 1.0)
    expected = (
        cfg.w_sim * b.sim + cfg.w_exact * b.exact + cfg.w_chrf * b.chrf
        - cfg.w_char * b.char_ratio - cfg.w_bits * b.bits_ratio - cfg.w_read * b.read_pen
    )
    assert math.isclose(b.total, expected)
    assert math.isclose(b.bits_ratio, 5 / 7)
    assert b.read_pen == 0.0


def test_unreadable_dense_is_penalized():
    s = Scorer(RewardConfig(), StubEmbedder(), StubRef())
    good, bad = s.score(["the cat sat"] * 2, ["cat sat", "zzz qq"], ["the cat sat"] * 2)
    assert bad.read_pen > 0 and bad.total < good.total


def test_ref_skipped_when_weights_zero_and_not_compute_all():
    ref = StubRef()
    s = Scorer(RewardConfig(w_bits=0.0, w_read=0.0), StubEmbedder(), ref)
    s.score(["a b"], ["a"], ["a b"], compute_all=False)
    assert ref.calls == 0
    s.score(["a b"], ["a"], ["a b"], compute_all=True)
    assert ref.calls == 1


def test_score_handles_empty_strings():
    s = Scorer(RewardConfig(), StubEmbedder(), StubRef())
    [b] = s.score(["the cat sat"], [""], [""])
    assert b.char_ratio == 0.0 and b.ppl_dense == 1.0
    assert all(math.isfinite(v) for v in b.to_dict().values())


def test_scorer_requires_ref_for_bits():
    with pytest.raises(ValueError, match="reference"):
        Scorer(RewardConfig(), StubEmbedder(), None).score(["a"], ["a"], ["a"])

import math

import pytest

from heet.config import Config
from heet.eval.evaluate import aggregate, run_eval, score_triples
from heet.rewards.combined import RewardBreakdown, Scorer
from tests.stubs import StubEmbedder, StubRef


class WsTok:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


def test_aggregate_values_and_pass():
    cfg = Config()
    bs = [
        RewardBreakdown(sim=1.0, char_ratio=0.5, bits_dense=10, bits_orig=20, bits_ratio=0.5,
                        ppl_dense=4, ppl_orig=2, ppl_log_ratio=math.log(2), total=1.0, exact=1.0),
        RewardBreakdown(sim=0.9, char_ratio=0.4, bits_dense=10, bits_orig=20, bits_ratio=0.5,
                        ppl_dense=2, ppl_orig=2, ppl_log_ratio=0.0, total=0.5),
    ]
    m = aggregate(bs, token_ratios=[0.5, 0.5], orig_chars=[50, 50], cfg=cfg)
    assert math.isclose(m["sim"], 0.95) and math.isclose(m["char_ratio"], 0.45)
    assert math.isclose(m["bpc_dense"], 0.2) and math.isclose(m["bpc_orig"], 0.4)
    assert m["exact_rate"] == 0.5 and m["n"] == 2
    assert m["pass"] is True
    cfg.eval.target_char_ratio = 0.3
    assert aggregate(bs, [0.5, 0.5], [50, 50], cfg)["pass"] is False


def test_score_triples_records():
    cfg = Config()
    scorer = Scorer(cfg.reward, StubEmbedder(), StubRef())
    metrics, recs = score_triples(["the cat sat down"], ["cat sat"], ["the cat sat down"], WsTok(), scorer, cfg)
    assert metrics["token_ratio"] == 0.5 and metrics["sim"] == 1.0
    assert recs[0]["original"] == "the cat sat down" and "total" in recs[0]


def test_eval_missing_adapter(tmp_path):
    cfg = Config()
    cfg.paths.runs_dir = str(tmp_path / "runs")
    cfg.paths.data_dir = str(tmp_path / "data")
    with pytest.raises(FileNotFoundError, match="heet sft --task"):
        run_eval(cfg, run="r1", split="test", limit=2)


def test_load_split_pairs_message_reports_counts(tmp_path):
    from heet.eval.evaluate import load_split_pairs
    from heet.jsonl import write_jsonl

    cfg = Config()
    cfg.paths.data_dir = str(tmp_path)
    write_jsonl(tmp_path / "pairs.jsonl", [{"id": "a", "split": "train"}, {"id": "b", "split": "train"}])
    with pytest.raises(FileNotFoundError, match=r"train=2.*corpus.val_frac"):
        load_split_pairs(cfg, "val", None)

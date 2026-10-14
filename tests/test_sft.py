from heet.config import Config
from heet.eval.logger import MetricsLogger
from heet.jsonl import read_jsonl
from heet.rewards.combined import Scorer
from heet.training.sft import EvalCallback, build_examples
from tests.stubs import StubEmbedder, StubRef
from tests.test_prompts import FakeTok

PAIRS = [{"id": "a", "split": "val", "original": "The cat sat on the mat.", "dense": "cat sat on mat"}]


class WsTok:
    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


class EchoModel:
    """Stands in for the PeftModel; generation is monkeypatched."""

    active_adapter = "compressor"


def test_build_examples_directions():
    [c] = build_examples(PAIRS, "compress", FakeTok())
    assert "The cat sat on the mat." in c["prompt"] and c["completion"] == "cat sat on mat<|im_end|>"
    [d] = build_examples(PAIRS, "decompress", FakeTok())
    assert "cat sat on mat" in d["prompt"] and d["completion"] == "The cat sat on the mat.<|im_end|>"


def test_callback_without_decompressor(tmp_path, monkeypatch, caplog):
    import heet.training.sft as sft

    monkeypatch.setattr(sft, "generate", lambda model, tok, task, texts, gen: [t.lower()[:10] for t in texts])
    cfg = Config()
    cb = EvalCallback(cfg, "compress", WsTok(), PAIRS, Scorer(cfg.reward, StubEmbedder(), StubRef()),
                      MetricsLogger(tmp_path), tmp_path, has_decompressor=False)
    m1 = cb.evaluate(EchoModel(), step=5)
    cb.evaluate(EchoModel(), step=10)
    assert "char_ratio" in m1 and "ppl_log_ratio" in m1 and "sim" not in m1
    assert sum("decompressor" in r.message for r in caplog.records) == 1
    rows = read_jsonl(tmp_path / "metrics.jsonl")
    assert rows[0]["step"] == 5 and "compress/char_ratio" in rows[0]
    assert (tmp_path / "samples" / "compress_step_000005.jsonl").exists()


def test_callback_decompress_task(tmp_path, monkeypatch):
    import heet.training.sft as sft

    monkeypatch.setattr(sft, "generate", lambda model, tok, task, texts, gen: ["The cat sat on the mat."] * len(texts))
    cfg = Config()
    cb = EvalCallback(cfg, "decompress", WsTok(), PAIRS, Scorer(cfg.reward, StubEmbedder(), StubRef()),
                      MetricsLogger(tmp_path), tmp_path, has_decompressor=True)
    m = cb.evaluate(EchoModel(), step=1)
    assert m["sim"] == 1.0 and m["exact_rate"] == 1.0

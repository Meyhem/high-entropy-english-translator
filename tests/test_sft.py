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


def test_select_val_pairs_empty_warns(tmp_path, caplog):
    from heet.jsonl import write_jsonl
    from heet.training.sft import select_val_pairs

    cfg = Config()
    cfg.paths.data_dir = str(tmp_path)
    write_jsonl(tmp_path / "pairs.jsonl", [{"id": "a", "split": "train", "original": "x", "dense": "y"}])
    assert select_val_pairs(cfg) == []
    assert any("no 'val' pairs" in r.message for r in caplog.records)


def test_callback_custom_source_and_tag(tmp_path, monkeypatch):
    import heet.training.sft as sft

    monkeypatch.setattr(sft, "generate", lambda model, tok, task, texts, gen: [t.lower()[:10] for t in texts])
    cfg = Config()
    cb = EvalCallback(cfg, "compress", WsTok(), PAIRS, Scorer(cfg.reward, StubEmbedder(), StubRef()),
                      MetricsLogger(tmp_path), tmp_path, has_decompressor=False, source="ei", sample_tag="ei_val_")
    cb.evaluate(EchoModel(), step=2)
    assert read_jsonl(tmp_path / "metrics.jsonl")[0]["source"] == "ei"
    assert (tmp_path / "samples" / "ei_val_step_000002.jsonl").exists()


def test_train_adapter_uses_given_lr_epochs_and_tag(tmp_path, monkeypatch):
    import trl

    from heet.training.sft import train_adapter

    seen = {}

    class FakeTrainer:
        def __init__(self, model, args, train_dataset, processing_class, callbacks):
            seen.update(args=args, n=len(train_dataset), callbacks=callbacks)

        def train(self):
            seen["trained"] = True

    monkeypatch.setattr(trl, "SFTTrainer", FakeTrainer)
    cfg = Config()
    cfg.model.dtype = "float32"
    train_adapter(object(), FakeTok(), cfg, PAIRS * 3, "compress", tmp_path, lr=5e-5, epochs=1.0, tag="ei_round_1")
    a = seen["args"]
    assert seen["trained"] and seen["n"] == 3 and seen["callbacks"] == []
    assert a.learning_rate == 5e-5 and a.num_train_epochs == 1.0
    assert a.output_dir == str(tmp_path / "checkpoints_ei_round_1")

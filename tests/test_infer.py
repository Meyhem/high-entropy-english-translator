import contextlib

import pytest

import heet.models.loading as loading
from heet.config import Config
from heet.infer import format_results, translate


@pytest.fixture
def fake_models(monkeypatch):
    calls = []

    @contextlib.contextmanager
    def use_adapter(model, name):
        calls.append(name)
        yield model

    def generate(model, tokenizer, task, texts, gen):
        return [f"{task}({t})" for t in texts]

    monkeypatch.setattr(loading, "use_adapter", use_adapter)
    monkeypatch.setattr(loading, "generate", generate)
    return calls


def test_translate_modes(fake_models):
    cfg = Config()
    assert translate(None, None, ["a"], "compress", cfg) == [("compress(a)",)]
    assert translate(None, None, ["d"], "decompress", cfg) == [("decompress(d)",)]
    assert translate(None, None, ["a", "b"], "roundtrip", cfg) == [
        ("compress(a)", "decompress(compress(a))"),
        ("compress(b)", "decompress(compress(b))"),
    ]
    assert fake_models == ["compressor", "decompressor", "compressor", "decompressor"]
    with pytest.raises(ValueError):
        translate(None, None, ["a"], "bogus", cfg)


def test_format_results():
    assert format_results(["x", "y"], [("X",), ("Y",)], "compress") == "X\nY\n"
    out = format_results(["abcd"], [("ab", "abcd!")], "roundtrip")
    assert "dense: ab  (2/4 chars, 0.50)" in out and "back:  abcd!" in out

import math
from pathlib import Path

import pytest

from heet.config import Config, dump_config, load_config


def test_defaults_match_spec():
    c = Config()
    assert c.model.name == "Qwen/Qwen3-0.6B"
    assert c.teacher.model == "qwen3:14b"
    assert c.filter.min_similarity == 0.90
    assert (c.filter.min_ratio, c.filter.max_ratio) == (0.30, 0.80)
    assert c.lora.r == 16 and c.lora.alpha == 32
    assert c.reward.w_sim == 1.0 and c.reward.w_char == 0.5
    assert math.isclose(c.reward.margin, math.log(2))
    assert c.eval.target_char_ratio == 0.55


def test_load_yaml_and_overrides(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("sft:\n  lr: 0.001\nmodel:\n  name: Qwen/Qwen3-1.7B\n")
    c = load_config(p, ["sft.eval_steps=50", "reward.w_chrf=0.3", "corpus.source=/tmp/x.txt"])
    assert c.sft.lr == 0.001
    assert c.model.name == "Qwen/Qwen3-1.7B"
    assert c.sft.eval_steps == 50 and isinstance(c.sft.eval_steps, int)
    assert c.reward.w_chrf == 0.3
    assert c.corpus.source == "/tmp/x.txt"


def test_unknown_key_rejected(tmp_path: Path):
    with pytest.raises(KeyError, match="sft.nope"):
        load_config(None, ["sft.nope=1"])
    p = tmp_path / "c.yaml"
    p.write_text("bogus:\n  a: 1\n")
    with pytest.raises(KeyError, match="bogus"):
        load_config(p)


def test_dump_roundtrip(tmp_path: Path):
    c = load_config(None, ["sft.lr=0.0003"])
    dump_config(c, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml").sft.lr == 0.0003


def test_default_yaml_loads():
    c = load_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    assert c == Config()

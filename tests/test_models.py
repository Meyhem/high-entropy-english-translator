import pytest
import torch

from heet.config import GenerationConfig, LoraConfig, ModelConfig
from heet.models.loading import generate, load_adapters, load_base, load_tokenizer, new_lora, use_adapter
from heet.rewards.ref_lm import RefLM

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(TINY)


def test_generate_returns_one_string_per_input(tok):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    out = generate(base, tok, "compress", ["Hello there.", "A second, longer sentence."],
                   GenerationConfig(batch_size=1, max_new_tokens_factor=0.0, max_new_tokens_extra=4))
    assert len(out) == 2 and all(isinstance(o, str) for o in out)


def test_two_adapters_switch_and_save(tmp_path, tok):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    m = new_lora(base, LoraConfig(r=2, alpha=4, target_modules=["q_proj", "v_proj"]), "decompressor")
    m.save_pretrained(str(tmp_path / "adapters"), selected_adapters=["decompressor"])
    assert (tmp_path / "adapters" / "decompressor" / "adapter_config.json").exists()

    base2 = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    m2 = new_lora(base2, LoraConfig(r=2, alpha=4, target_modules=["q_proj", "v_proj"]), "compressor")
    m2.load_adapter(str(tmp_path / "adapters" / "decompressor"), adapter_name="decompressor", is_trainable=False)
    assert m2.active_adapter == "compressor"
    with use_adapter(m2, "decompressor"):
        assert m2.active_adapter == "decompressor"
    assert m2.active_adapter == "compressor"
    assert any(p.requires_grad for n, p in m2.named_parameters() if ".compressor." in n)
    assert not any(p.requires_grad for n, p in m2.named_parameters() if ".decompressor." in n)


def test_load_adapters_missing_dir(tmp_path):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    with pytest.raises(FileNotFoundError, match="heet sft --task compress"):
        load_adapters(base, {"compressor": tmp_path / "nope"})


def test_ref_lm_on_peft_model_matches_base(tok):
    base = load_base(ModelConfig(name=TINY, dtype="float32", device="cpu"))
    plain = RefLM(base, tok).score(["The cat sat on the mat."])[0]
    m = new_lora(base, LoraConfig(r=2, alpha=4, target_modules=["q_proj", "v_proj"]), "compressor")
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.fill_(0.5)  # make the adapter non-identity
    peft_score = RefLM(m, tok).score(["The cat sat on the mat."])[0]
    assert peft_score.n_tokens == plain.n_tokens
    assert abs(peft_score.nll_nats - plain.nll_nats) < 1e-4

import math

import torch

from heet.rewards.ref_lm import RefLM


class UniformLM(torch.nn.Module):
    """Every next-token distribution is uniform over V tokens -> nll = log V per token."""

    def __init__(self, vocab: int = 8):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        return type("Out", (), {"logits": torch.zeros(b, t, self.vocab)})()


class CharTok:
    """Each character is one token id (mod 7) + 1; id 0 is <|endoftext|>."""

    pad_token_id = 0

    def convert_tokens_to_ids(self, tok):
        assert tok == "<|endoftext|>"
        return 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[(ord(c) % 7) + 1 for c in t] for t in texts]}


def test_ref_lm_uniform_nll():
    ref = RefLM(UniformLM(8), CharTok(), batch_size=2)
    a, b, c = ref.score(["abc", "hello", "x"])
    assert a.n_tokens == 3 and math.isclose(a.nll_nats, 3 * math.log(8), rel_tol=1e-5)
    assert b.n_tokens == 5 and math.isclose(b.ppl, 8.0, rel_tol=1e-5)
    assert c.n_tokens == 1


def test_ref_lm_empty_text():
    [s] = RefLM(UniformLM(8), CharTok()).score([""])
    assert s.n_tokens == 0 and s.nll_nats == 0.0 and s.ppl == 1.0

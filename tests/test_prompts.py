import pytest

from heet.prompts import (
    clean_output,
    render_completion,
    render_prompt,
    task_messages,
    teacher_reconstruct_messages,
    teacher_rewrite_messages,
)


def test_task_messages_shape():
    m = task_messages("compress", "Hello world.")
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[1]["content"] == "Hello world."
    assert task_messages("decompress", "x")[0]["content"] != m[0]["content"]
    with pytest.raises(ValueError):
        task_messages("nope", "x")


def test_teacher_messages_contain_rules_and_text():
    m = teacher_rewrite_messages("The cat sat on the mat.")
    assert m[0]["role"] == "system" and "number" in m[0]["content"].lower()
    assert m[-1]["content"].endswith("The cat sat on the mat.")
    r = teacher_reconstruct_messages("cat sat on mat")
    assert r[-1]["content"].endswith("cat sat on mat")
    assert "The cat sat on the mat." not in str(r)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Dense: cat sat on mat", "cat sat on mat"),
        ('"cat sat on mat"', "cat sat on mat"),
        ("```\ncat sat on mat\n```", "cat sat on mat"),
        ("<think>\nhmm\n</think>\n\ncat sat on mat", "cat sat on mat"),
        ("cat sat on mat\n\nNote: I removed articles.", "cat sat on mat"),
        ("Reconstruction: The cat sat.", "The cat sat."),
        ("   ", ""),
        ("Gov announced bridge.\nNote: I dropped articles.", "Gov announced bridge."),
        ("Gov bridge.\n(Note: abbreviated)", "Gov bridge."),
        ("Here is the rewrite:\nGov bridge.", "Gov bridge."),
        ('"Dense: Gov bridge"', "Gov bridge"),
        ("**Dense:** Gov bridge", "Gov bridge"),
    ],
)
def test_clean_output(raw, expected):
    assert clean_output(raw) == expected


class FakeTok:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return body + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_render_prompt_and_completion():
    p = render_prompt(FakeTok(), "compress", "Hi there.")
    assert p.endswith("</think>\n\n") and "Hi there." in p
    assert render_completion(FakeTok(), "hi") == "hi<|im_end|>"


@pytest.mark.slow
def test_render_prompt_real_qwen3_template():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")
    p = render_prompt(tok, "decompress", "cat sat on mat")
    assert "cat sat on mat" in p and p.rstrip().endswith("</think>")

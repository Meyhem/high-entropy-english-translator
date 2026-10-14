"""Prompt construction shared by data generation, SFT and evaluation."""

from __future__ import annotations

import re

TASK_SYSTEM: dict[str, str] = {
    "compress": (
        "Rewrite the user's English text as dense English: as short as possible while a "
        "careful reader can still recover the full meaning. Abbreviate, drop function words, "
        "use telegraphic phrasing and symbols. Keep names and numbers exact. Output only the rewrite."
    ),
    "decompress": (
        "The user's text is dense English (abbreviated, telegraphic). Expand it back into the "
        "original fluent English sentence(s). Output only the expanded text."
    ),
}

TEACHER_REWRITE_SYSTEM = """You compress English into "dense English": shorter, information-dense text that a careful human reader can still understand and expand back to the original.

Rules:
- Abbreviate common words (e.g. "government" -> "gov", "approximately" -> "~", "with" -> "w/", "because" -> "b/c", "and" -> "&", "leads to"/"becomes" -> "→").
- Drop articles, copulas and auxiliaries when meaning survives ("the", "a", "is", "was", "has been").
- Use telegraphic phrasing; merge clauses with ";" or ",".
- Keep EVERY proper name and EVERY number exactly as written.
- Do not add information. Do not explain. Output only the rewrite on a single line."""

TEACHER_FEWSHOT: list[tuple[str, str]] = [
    (
        "The government announced that the new bridge, which cost approximately 40 million dollars, "
        "would be opened to traffic in March.",
        "Gov announced new bridge (~$40M) opens to traffic in March.",
    ),
    (
        "After the war, Smith returned to London, where he worked as a journalist for the Times "
        "until his retirement in 1952.",
        "Post-war Smith → London, journalist @ Times till retiring 1952.",
    ),
]

TEACHER_RECON_SYSTEM = """The user's text is "dense English": abbreviated, telegraphic English with dropped function words and symbols such as & w/ → ~ b/c.
Expand it back into complete, fluent English sentence(s) with the same meaning. Keep every name and number exactly.
Do not add information. Do not explain. Output only the expanded text."""


def task_messages(task: str, text: str) -> list[dict[str, str]]:
    if task not in TASK_SYSTEM:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASK_SYSTEM)}")
    return [{"role": "system", "content": TASK_SYSTEM[task]}, {"role": "user", "content": text}]


def render_prompt(tokenizer, task: str, text: str) -> str:
    return tokenizer.apply_chat_template(
        task_messages(task, text),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_completion(tokenizer, text: str) -> str:
    return text + tokenizer.eos_token


def teacher_rewrite_messages(text: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": TEACHER_REWRITE_SYSTEM}]
    for orig, dense in TEACHER_FEWSHOT:
        msgs.append({"role": "user", "content": f"Text: {orig}"})
        msgs.append({"role": "assistant", "content": dense})
    msgs.append({"role": "user", "content": f"Text: {text}"})
    return msgs


def teacher_reconstruct_messages(dense: str) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": TEACHER_RECON_SYSTEM}]
    for orig, d in TEACHER_FEWSHOT:
        msgs.append({"role": "user", "content": f"Dense: {d}"})
        msgs.append({"role": "assistant", "content": orig})
    msgs.append({"role": "user", "content": f"Dense: {dense}"})
    return msgs


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*$", re.MULTILINE)
_LABEL = re.compile(
    r"^\**\s*(dense|rewrite|reconstruction|output|original|text|expanded)\s*:\s*\**\s*", re.I
)
_QUOTES = ('"', "'", "“", "”")


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        return text[1:-1].strip()
    return text


def clean_output(text: str) -> str:
    """Keep only the answer line: outputs are single-line by prompt, so later lines are commentary."""
    text = _FENCE.sub("", _THINK.sub("", text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].endswith(":"):
        lines = lines[1:]  # preamble such as "Here is the rewrite:"
    text = " ".join(lines[0].split()) if lines else ""
    return _strip_quotes(_LABEL.sub("", _strip_quotes(text)))

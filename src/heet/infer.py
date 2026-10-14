"""Ad-hoc inference: compress, decompress or round-trip user-supplied text with a run's adapters."""

from __future__ import annotations

import sys
from typing import TextIO

from heet.config import Config

MODES = ("compress", "decompress", "roundtrip")
_ADAPTERS = {
    "compress": ("compressor",),
    "decompress": ("decompressor",),
    "roundtrip": ("compressor", "decompressor"),
}


def translate(model, tokenizer, texts: list[str], mode: str, cfg: Config) -> list[tuple[str, ...]]:
    """Return one tuple per input: (output,) for compress/decompress, (dense, recon) for roundtrip."""
    from heet.models.loading import generate, use_adapter

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if mode == "decompress":
        with use_adapter(model, "decompressor"):
            return [(r,) for r in generate(model, tokenizer, "decompress", texts, cfg.generation)]
    with use_adapter(model, "compressor"):
        denses = generate(model, tokenizer, "compress", texts, cfg.generation)
    if mode == "compress":
        return [(d,) for d in denses]
    with use_adapter(model, "decompressor"):
        recons = generate(model, tokenizer, "decompress", denses, cfg.generation)
    return list(zip(denses, recons, strict=True))


def format_results(texts: list[str], results: list[tuple[str, ...]], mode: str) -> str:
    if mode != "roundtrip":
        return "".join(r[0] + "\n" for r in results)
    blocks = []
    for text, (dense, recon) in zip(texts, results, strict=True):
        ratio = len(dense) / len(text) if text else 0.0
        blocks.append(f"in:    {text}\ndense: {dense}  ({len(dense)}/{len(text)} chars, {ratio:.2f})\nback:  {recon}\n")
    return "\n".join(blocks)


def run_compress(cfg: Config, run: str, texts: list[str], mode: str, out: TextIO = sys.stdout) -> None:
    from heet.models.loading import adapter_dir, load_adapters, load_base, load_tokenizer

    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        raise SystemExit("no input text; pass it as arguments or pipe lines on stdin")
    dirs = {name: adapter_dir(cfg, run, name) for name in _ADAPTERS[mode]}
    tokenizer = load_tokenizer(cfg.model.name)
    model = load_adapters(load_base(cfg.model), dirs)
    out.write(format_results(texts, translate(model, tokenizer, texts, mode, cfg), mode))

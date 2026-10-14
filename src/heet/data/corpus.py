"""Corpus → cleaned 1-3 sentence chunks with deterministic splits."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from heet.config import CorpusConfig
from heet.jsonl import write_jsonl

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "inc", "ltd", "co",
    "gen", "col", "lt", "sgt", "capt", "gov", "sen", "rep", "no", "vol", "fig", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "e.g", "i.e", "u.s",
    "u.k", "a.m", "p.m",
}
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def clean_wikitext(line: str) -> str:
    s = line.strip()
    if not s or (s.startswith("=") and s.endswith("=")):
        return ""
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r"\s+([.,;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\s+('s|n't|'re|'ve|'ll|'d)\b", r"\1", s)
    return " ".join(s.split())


def split_sentences(paragraph: str) -> list[str]:
    parts: list[str] = []
    buf = ""
    last = 0
    for m in _SENT_END.finditer(paragraph):
        candidate = paragraph[last : m.start()].strip()
        last_word = candidate.rsplit(" ", 1)[-1].rstrip(".").lower()
        buf = f"{buf} {candidate}".strip() if buf else candidate
        last = m.end()
        if last_word in _ABBREV or (len(last_word) == 1 and last_word.isalpha()):
            continue
        parts.append(buf)
        buf = ""
    tail = paragraph[last:].strip()
    if tail:
        buf = f"{buf} {tail}".strip() if buf else tail
    if buf:
        parts.append(buf)
    return parts


def chunk_paragraph(paragraph: str, cfg: CorpusConfig) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []

    def flush():
        if cur:
            text = " ".join(cur)
            if cfg.min_chars <= len(text) <= cfg.max_chars:
                chunks.append(text)
            cur.clear()

    for sent in split_sentences(paragraph):
        if len(sent) > cfg.max_chars:
            flush()
            continue
        if cur and (len(cur) >= cfg.max_sentences or len(" ".join(cur + [sent])) > cfg.max_chars):
            flush()
        cur.append(sent)
        if len(" ".join(cur)) >= cfg.min_chars and len(cur) >= cfg.max_sentences:
            flush()
    flush()
    return chunks


def chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def assign_split(cid: str, cfg: CorpusConfig) -> str:
    u = int(cid[:8], 16) / 0xFFFFFFFF
    if u < cfg.test_frac:
        return "test"
    if u < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


def iter_paragraphs(cfg: CorpusConfig) -> Iterator[str]:
    if cfg.source == "wikitext":
        from datasets import load_dataset

        ds = load_dataset(cfg.hf_dataset, cfg.hf_config, split=cfg.hf_split)
        for row in ds:
            text = clean_wikitext(row["text"])
            if text:
                yield text
        return
    raw = Path(cfg.source).read_text(encoding="utf-8")
    for block in re.split(r"\n\s*\n", raw):
        text = clean_wikitext(" ".join(block.split()))
        if text:
            yield text


def build_chunks(cfg: CorpusConfig, out_path: Path) -> int:
    seen: set[str] = set()
    rows: list[dict] = []
    for para in iter_paragraphs(cfg):
        for text in chunk_paragraph(para, cfg):
            cid = chunk_id(text)
            if cid in seen:
                continue
            seen.add(cid)
            rows.append({"id": cid, "text": text, "split": assign_split(cid, cfg)})
            if len(rows) >= cfg.max_chunks:
                write_jsonl(out_path, rows)
                return len(rows)
    write_jsonl(out_path, rows)
    return len(rows)

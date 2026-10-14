"""Stage 1: teacher rewrites → blind reconstruction → fidelity filter → pairs.jsonl."""

from __future__ import annotations

import asyncio
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from heet.config import Config, FilterConfig
from heet.data.corpus import build_chunks
from heet.data.llm_client import ChatClient, OllamaClient
from heet.jsonl import append_jsonl, read_jsonl
from heet.prompts import clean_output, teacher_reconstruct_messages, teacher_rewrite_messages
from heet.rewards.fidelity import Embedder, normalize

log = logging.getLogger(__name__)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def cheap_checks(original: str, dense: str, cfg: FilterConfig) -> str | None:
    if not dense:
        return "empty"
    if "<think>" in dense:
        return "think_tag"
    ratio = len(dense) / len(original)
    if ratio < cfg.min_ratio:
        return "ratio_low"
    if ratio > cfg.max_ratio:
        return "ratio_high"
    for num in _NUMBER.findall(original):
        if num not in dense:
            return "missing_number"
    if SequenceMatcher(None, normalize(original), normalize(dense)).ratio() >= cfg.near_copy:
        return "near_copy"
    return None


async def process_chunk(chunk: dict, client: ChatClient, embedder: Embedder, cfg: Config):
    original = chunk["text"]
    rec = {
        "id": chunk["id"], "split": chunk["split"], "original": original,
        "dense": None, "teacher_recon": None, "similarity": None, "ratio": None,
    }
    try:
        dense = clean_output(await client.chat(teacher_rewrite_messages(original), cfg.teacher.rewrite_temperature))
        rec["dense"] = dense
        rec["ratio"] = len(dense) / len(original)
        reason = cheap_checks(original, dense, cfg.filter)
        if reason:
            return rec, reason
        recon = clean_output(
            await client.chat(teacher_reconstruct_messages(dense), cfg.teacher.recon_temperature)
        )
    except (httpx.HTTPError, KeyError, ValueError) as e:
        log.warning("teacher error on %s: %r", chunk["id"], e)
        return rec, "teacher_error"
    rec["teacher_recon"] = recon
    rec["similarity"] = float(embedder.similarity([original], [recon])[0])
    if rec["similarity"] < cfg.filter.min_similarity:
        return rec, "low_similarity"
    return rec, None


async def generate_pairs(chunks, client: ChatClient, embedder: Embedder, cfg: Config, limit: int | None):
    data = Path(cfg.paths.data_dir)
    pairs_path, rejected_path = data / "pairs.jsonl", data / "rejected.jsonl"
    done = {r["id"] for r in read_jsonl(pairs_path)} | {r["id"] for r in read_jsonl(rejected_path)}
    todo = [c for c in chunks if c["id"] not in done]
    skipped = len(chunks) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    counts = {"accepted": 0, "rejected": 0, "skipped": skipped}
    sem = asyncio.Semaphore(cfg.teacher.concurrency)

    async def one(chunk):
        async with sem:
            rec, reason = await process_chunk(chunk, client, embedder, cfg)
        if reason is None:
            append_jsonl(pairs_path, rec)
            counts["accepted"] += 1
        else:
            append_jsonl(rejected_path, {**rec, "reason": reason})
            counts["rejected"] += 1
        done_n = counts["accepted"] + counts["rejected"]
        if done_n % 100 == 0:
            log.info("processed %d/%d (accepted %d)", done_n, len(todo), counts["accepted"])

    await asyncio.gather(*(one(c) for c in todo))
    return counts


def run_gen_data(cfg: Config, limit: int | None, rechunk: bool) -> None:
    from heet.models.loading import resolve_device
    from heet.rewards.fidelity import SentenceEmbedder

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chunks_path = Path(cfg.paths.data_dir) / "chunks.jsonl"
    if rechunk or not chunks_path.exists():
        n = build_chunks(cfg.corpus, chunks_path)
        log.info("wrote %d chunks to %s", n, chunks_path)
    chunks = read_jsonl(chunks_path)

    async def main():
        client = OllamaClient(cfg.teacher)
        try:
            await client.preflight()
            embedder = SentenceEmbedder(cfg.reward.embedder, device=resolve_device(cfg.model))
            return await generate_pairs(chunks, client, embedder, cfg, limit)
        finally:
            await client.aclose()

    counts = asyncio.run(main())
    log.info("done: %s", counts)

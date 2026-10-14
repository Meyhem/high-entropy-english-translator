"""Dump (input, dense, reconstruction) triples for manual inspection."""

from __future__ import annotations

from pathlib import Path

from heet.jsonl import write_jsonl


def write_samples(run_dir: Path, step: int, records: list[dict], n_samples: int, worst_k: int, tag: str = "") -> Path:
    fixed = [{**r, "worst": False} for r in records[:n_samples]]
    tail = sorted(records[n_samples:], key=lambda r: r["total"])[:worst_k]
    worst = [{**r, "worst": True} for r in tail]
    path = Path(run_dir) / "samples" / f"{tag}step_{step:06d}.jsonl"
    write_jsonl(path, fixed + worst)
    return path

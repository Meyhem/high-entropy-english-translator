"""JSONL helpers that tolerate interrupted writes."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated line from an interrupted run
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # a previous crash may have left a partial line without a trailing newline
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as r:
            r.seek(-1, 2)
            needs_newline = r.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

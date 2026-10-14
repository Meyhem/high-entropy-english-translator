from pathlib import Path

from heet.jsonl import append_jsonl, read_jsonl, write_jsonl


def test_append_and_read(tmp_path: Path):
    p = tmp_path / "sub" / "a.jsonl"
    append_jsonl(p, {"id": "1"})
    append_jsonl(p, {"id": "2"})
    assert [r["id"] for r in read_jsonl(p)] == ["1", "2"]


def test_read_skips_truncated_and_blank(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"id": "1"}\n\n{"id": "2"}\n{"id": "3", "te')
    assert [r["id"] for r in read_jsonl(p)] == ["1", "2"]


def test_missing_file_is_empty(tmp_path: Path):
    assert read_jsonl(tmp_path / "none.jsonl") == []


def test_write_overwrites(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    write_jsonl(p, [{"a": 1}])
    write_jsonl(p, [{"a": 2}])
    assert read_jsonl(p) == [{"a": 2}]

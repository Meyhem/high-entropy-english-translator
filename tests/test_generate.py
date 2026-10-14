import asyncio
import json

import httpx
import pytest

from heet.config import Config, TeacherConfig
from heet.data.generate import cheap_checks, generate_pairs, process_chunk
from heet.data.llm_client import OllamaClient, TeacherUnavailable
from heet.jsonl import read_jsonl
from tests.stubs import StubEmbedder

ORIG = "The government announced that the bridge, costing 40 million dollars, would open in March 2021."


class FakeClient:
    """Returns scripted rewrites; reconstructions return `recon` or the original."""

    def __init__(self, rewrite: str, recon: str | None = None, fail: bool = False):
        self.rewrite, self.recon, self.fail = rewrite, recon, fail
        self.calls: list[str] = []

    async def chat(self, messages, temperature):
        if self.fail:
            raise httpx.ConnectError("boom")
        last = messages[-1]["content"]
        if last.startswith("Text: "):
            self.calls.append("rewrite")
            return self.rewrite
        self.calls.append("recon")
        return self.recon if self.recon is not None else ORIG


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.paths.data_dir = str(tmp_path)
    c.filter.min_similarity = 0.9
    return c


def test_cheap_checks(cfg):
    f = cfg.filter
    assert cheap_checks(ORIG, "Gov: bridge ($40M) opens March 2021.", f) is None
    assert cheap_checks(ORIG, "", f) == "empty"
    assert cheap_checks(ORIG, "Gov: bridge opens March 2021, about 40M.", f) is None
    assert cheap_checks(ORIG, "Gov: bridge (big cost) opens in March.", f) == "missing_number"
    assert cheap_checks(ORIG, "x 40 2021", f) == "ratio_low"
    assert cheap_checks(ORIG, ORIG.replace("The ", ""), f) in {"ratio_high", "near_copy"}


def test_process_chunk_accept(cfg):
    chunk = {"id": "a", "text": ORIG, "split": "train"}
    rec, reason = asyncio.run(
        process_chunk(chunk, FakeClient("Dense: Gov: bridge ($40M) opens March 2021."), StubEmbedder(), cfg)
    )
    assert reason is None
    assert rec["dense"] == "Gov: bridge ($40M) opens March 2021."
    assert rec["similarity"] == 1.0 and 0.3 <= rec["ratio"] <= 0.8


def test_process_chunk_low_similarity(cfg):
    chunk = {"id": "a", "text": ORIG, "split": "train"}
    rec, reason = asyncio.run(
        process_chunk(chunk, FakeClient("Gov: bridge ($40M) opens March 2021.", recon="Cats like milk."),
                      StubEmbedder(), cfg)
    )
    assert reason == "low_similarity" and rec["teacher_recon"] == "Cats like milk."


def test_cheap_reject_skips_reconstruction(cfg):
    client = FakeClient("bridge")
    rec, reason = asyncio.run(process_chunk({"id": "a", "text": ORIG, "split": "train"}, client, StubEmbedder(), cfg))
    assert reason is not None and client.calls == ["rewrite"]


def test_clean_teacher_output(cfg):
    noisy = '```\n"Gov: bridge ($40M) opens March 2021."\n```\n\nNote: removed articles.'
    rec, reason = asyncio.run(
        process_chunk({"id": "a", "text": ORIG, "split": "train"}, FakeClient(noisy), StubEmbedder(), cfg)
    )
    assert reason is None and rec["dense"] == "Gov: bridge ($40M) opens March 2021."


def test_teacher_error_rejected(cfg):
    rec, reason = asyncio.run(
        process_chunk({"id": "a", "text": ORIG, "split": "train"}, FakeClient("", fail=True), StubEmbedder(), cfg)
    )
    assert reason == "teacher_error"


def _chunks(n):
    return [{"id": f"c{i}", "text": ORIG, "split": "train"} for i in range(n)]


def test_generate_pairs_writes_and_resumes(cfg, tmp_path):
    client = FakeClient("Gov: bridge ($40M) opens March 2021.")
    counts = asyncio.run(generate_pairs(_chunks(3), client, StubEmbedder(), cfg, limit=2))
    assert counts == {"accepted": 2, "rejected": 0, "skipped": 0}
    counts = asyncio.run(generate_pairs(_chunks(3), client, StubEmbedder(), cfg, limit=None))
    assert counts == {"accepted": 1, "rejected": 0, "skipped": 2}
    assert [r["id"] for r in read_jsonl(tmp_path / "pairs.jsonl")] == ["c0", "c1", "c2"]


def test_resume_tolerates_truncated_line(cfg, tmp_path):
    good = {"id": "c0", "split": "train", "original": ORIG, "dense": "d", "teacher_recon": ORIG,
            "similarity": 1.0, "ratio": 0.5}
    (tmp_path / "pairs.jsonl").write_text(json.dumps(good) + "\n" + '{"id": "c1", "spl')
    client = FakeClient("Gov: bridge ($40M) opens March 2021.")
    counts = asyncio.run(generate_pairs(_chunks(2), client, StubEmbedder(), cfg, limit=None))
    assert counts["skipped"] == 1 and counts["accepted"] == 1
    assert [r["id"] for r in read_jsonl(tmp_path / "pairs.jsonl")] == ["c0", "c1"]


def test_rejected_written_with_reason(cfg, tmp_path):
    asyncio.run(generate_pairs(_chunks(1), FakeClient("bridge"), StubEmbedder(), cfg, limit=None))
    [r] = read_jsonl(tmp_path / "rejected.jsonl")
    assert r["id"] == "c0" and r["reason"] in {"ratio_low", "missing_number"}


def test_preflight_unreachable():
    client = OllamaClient(TeacherConfig(base_url="http://127.0.0.1:9", timeout_s=2))
    with pytest.raises(TeacherUnavailable, match="127.0.0.1:9"):
        asyncio.run(client.preflight())


def test_preflight_model_missing():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    client = OllamaClient(TeacherConfig(model="qwen3:14b"), transport=httpx.MockTransport(handler))
    with pytest.raises(TeacherUnavailable, match="ollama pull qwen3:14b"):
        asyncio.run(client.preflight())


def test_chat_sends_think_false():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})

    client = OllamaClient(TeacherConfig(), transport=httpx.MockTransport(handler))
    assert asyncio.run(client.chat([{"role": "user", "content": "hi"}], temperature=0.3)) == "ok"
    assert seen["think"] is False and seen["stream"] is False
    assert seen["options"]["temperature"] == 0.3


def test_teacher_error_retried_on_resume(cfg, tmp_path):
    asyncio.run(generate_pairs(_chunks(1), FakeClient("", fail=True), StubEmbedder(), cfg, limit=None))
    assert read_jsonl(tmp_path / "rejected.jsonl")[0]["reason"] == "teacher_error"
    counts = asyncio.run(generate_pairs(_chunks(1), FakeClient("Gov: bridge ($40M) opens March 2021."),
                                        StubEmbedder(), cfg, limit=None))
    assert counts == {"accepted": 1, "rejected": 0, "skipped": 0}


def test_consecutive_teacher_errors_abort(cfg):
    cfg.teacher.concurrency = 1
    with pytest.raises(TeacherUnavailable, match="consecutive"):
        asyncio.run(generate_pairs(_chunks(50), FakeClient("", fail=True), StubEmbedder(), cfg, limit=None))

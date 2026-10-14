from collections import Counter

from heet.config import CorpusConfig
from heet.data.corpus import assign_split, build_chunks, chunk_id, chunk_paragraph, clean_wikitext, split_sentences
from heet.jsonl import read_jsonl


def test_clean_wikitext():
    assert clean_wikitext(" = Valkyria Chronicles = \n") == ""
    assert clean_wikitext(" It cost 1 @,@ 000 dollars , a well @-@ known fact . \n") == (
        "It cost 1,000 dollars, a well-known fact."
    )
    assert clean_wikitext(" Pi is 3 @.@ 14 ( roughly ) . ") == "Pi is 3.14 (roughly)."


def test_split_sentences_guards_abbreviations():
    s = split_sentences("Dr. Smith met Mr. Jones in the U.S. on Monday. They talked. Was it fun? Yes!")
    assert s == ["Dr. Smith met Mr. Jones in the U.S. on Monday.", "They talked.", "Was it fun?", "Yes!"]


def test_chunk_bounds():
    cfg = CorpusConfig(min_chars=40, max_chars=120, max_sentences=3)
    para = " ".join(f"Sentence number {i} is about twenty-five chars." for i in range(10))
    chunks = chunk_paragraph(para, cfg)
    assert chunks
    for c in chunks:
        assert 40 <= len(c) <= 120
        assert 1 <= len(split_sentences(c)) <= 3


def test_long_sentence_dropped():
    cfg = CorpusConfig(min_chars=40, max_chars=60)
    assert chunk_paragraph("x" * 100 + ".", cfg) == []


def test_split_deterministic_and_proportional():
    cfg = CorpusConfig(val_frac=0.1, test_frac=0.1)
    ids = [chunk_id(f"text {i}") for i in range(5000)]
    assert [assign_split(i, cfg) for i in ids] == [assign_split(i, cfg) for i in ids]
    c = Counter(assign_split(i, cfg) for i in ids)
    assert 350 < c["val"] < 650 and 350 < c["test"] < 650


def test_build_chunks_from_text_file(tmp_path):
    src = tmp_path / "corpus.txt"
    src.write_text(
        "The first paragraph has a sentence. And another sentence follows it here.\n\n"
        "Second paragraph sentence one is here. Sentence two is also here today.\n"
    )
    cfg = CorpusConfig(source=str(src), min_chars=20, max_chars=300, max_chunks=10)
    n = build_chunks(cfg, tmp_path / "chunks.jsonl")
    rows = read_jsonl(tmp_path / "chunks.jsonl")
    assert n == len(rows) == 2
    assert {"id", "text", "split"} <= rows[0].keys()
    assert len({r["id"] for r in rows}) == 2

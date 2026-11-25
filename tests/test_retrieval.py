from pathlib import Path

import numpy as np

from forgeai.ingest.models import Chunk
from forgeai.runner.retrieval import HybridIndex, load_corpus_chunks, tokenize
from tests.helpers import FakeEmbedder

SMOKE_DOCS = Path(__file__).parents[1] / "data" / "_smoke" / "docs"


def chunks_of(*texts: str) -> list[Chunk]:
    return [Chunk(chunk_id=f"d.txt#c{i}", doc="d.txt", index=i, text=t) for i, t in enumerate(texts)]


def test_tokenize_keeps_part_codes_whole_and_split() -> None:
    tokens = tokenize("Order MAN-B&W-1123 now")
    assert "man-b&w-1123" in tokens and "1123" in tokens and "order" in tokens


def test_hybrid_search_ranks_the_matching_chunk_first() -> None:
    chunks = chunks_of(
        "annual leave entitlement is twenty four days",
        "hotel stays are capped at one hundred fifty dollars",
        "part MAN-B&W-1123 is a piston ring",
    )
    index = HybridIndex.build(chunks, FakeEmbedder(), "fake")
    hits = index.search("hotel stays capped", top_k=2, dense_weight=0.6, bm25_weight=0.4, min_score=0.0)
    assert hits[0].chunk.chunk_id == "d.txt#c1" and len(hits) == 2


def test_bm25_finds_exact_part_numbers() -> None:
    chunks = chunks_of("alpha beta gamma", "piston ring MAN-B&W-1123 stock", "delta epsilon")
    index = HybridIndex.build(chunks, FakeEmbedder(), "fake")
    hits = index.search("MAN-B&W-1123", top_k=1, dense_weight=0.0, bm25_weight=1.0, min_score=0.0)
    assert hits[0].chunk.chunk_id == "d.txt#c1"


def test_min_score_gates_on_dense_similarity() -> None:
    index = HybridIndex.build(chunks_of("alpha beta", "gamma delta"), FakeEmbedder(), "fake")
    assert index.search("zzz unrelated words", 5, 0.6, 0.4, min_score=0.99) == []


def test_empty_index_returns_nothing() -> None:
    assert HybridIndex.build([], FakeEmbedder(), "fake").search("q", 5, 0.6, 0.4, 0.0) == []


def test_vectors_are_cached_by_content(tmp_path: Path) -> None:
    class Counting(FakeEmbedder):
        calls = 0

        def encode(self, texts):
            Counting.calls += 1
            return super().encode(texts)

    chunks = chunks_of("one two", "three four")
    HybridIndex.build(chunks, Counting(), "m", cache_dir=tmp_path)
    HybridIndex.build(chunks, Counting(), "m", cache_dir=tmp_path)
    assert Counting.calls == 1
    HybridIndex.build(chunks_of("one two", "changed"), Counting(), "m", cache_dir=tmp_path)
    assert Counting.calls == 2
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_corpus_loading_reports_skipped_files(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("1. Title\n\nSome text here.", encoding="utf-8")
    (tmp_path / "bad.xyz").write_text("nope", encoding="utf-8")
    chunks, skipped = load_corpus_chunks(tmp_path, 350, 60)
    assert [c.doc for c in chunks] == ["ok.txt"] and "bad.xyz" in skipped


def test_smoke_corpus_chunk_ids_are_stable() -> None:
    first, _ = load_corpus_chunks(SMOKE_DOCS, 350, 60)
    second, _ = load_corpus_chunks(SMOKE_DOCS, 350, 60)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert any(c.chunk_id.startswith("library_rules.md#") for c in first)
    assert np.all([c.text for c in first])

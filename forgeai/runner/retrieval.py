"""Hybrid retrieval: dense (NumPy cosine) + BM25, fused with reciprocal-rank fusion.

``min_score`` gates on the *dense cosine similarity* of a chunk: BM25 only influences the
ranking. Below the gate a chunk is treated as "no evidence".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from forgeai.embeddings import Embedder
from forgeai.ingest.chunking import chunk_document
from forgeai.ingest.loaders import LoadError, list_doc_files, load_document
from forgeai.ingest.models import Chunk

RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_/.&][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Hyphenated codes (``MAN-B&W-1123``) are kept whole *and* split."""
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        parts = re.split(r"[-_/.&]", match)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    dense: float  # cosine similarity to the query
    bm25: float


def load_corpus_chunks(
    docs_dir: Path, size_tokens: int, overlap_tokens: int, csv_mode: str = "row_as_record"
) -> tuple[list[Chunk], dict[str, str]]:
    """Load and chunk every parseable file. Returns ``(chunks, {skipped_file: reason})``."""
    chunks: list[Chunk] = []
    skipped: dict[str, str] = {}
    for path in list_doc_files(docs_dir):
        name = path.relative_to(docs_dir).as_posix()
        try:
            doc = load_document(path).model_copy(update={"file": name})
        except LoadError as exc:
            skipped[name] = str(exc)
            continue
        chunks.extend(chunk_document(doc, size_tokens, overlap_tokens, csv_mode))
    return chunks, skipped


class HybridIndex:
    def __init__(
        self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder, query_prefix: str = ""
    ) -> None:
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self._vectors = vectors
        self._embedder = embedder
        self._query_prefix = query_prefix
        self._bm25 = BM25Okapi([tokenize(c.retrieval_text) for c in chunks]) if chunks else None

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        model_name: str,
        cache_dir: Path | None = None,
        query_prefix: str = "",
    ) -> HybridIndex:
        """Embed ``chunks`` (or load vectors from the cache keyed by content + model)."""
        texts = [c.retrieval_text for c in chunks]
        key = hashlib.sha256((model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()
        cache_file = cache_dir / f"{key}.npy" if cache_dir else None
        if cache_file is not None and cache_file.is_file():
            vectors = np.load(cache_file)
        else:
            vectors = embedder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float32)
            if cache_file is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
                np.save(cache_file, vectors)
        return cls(chunks, vectors, embedder, query_prefix)

    @property
    def vectors(self) -> np.ndarray:
        """Chunk embeddings (passage side, no query prefix), aligned with ``chunks``."""
        return self._vectors

    def dense_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` (embedded WITH the query prefix) to every chunk."""
        if not self.chunks:
            return np.zeros(0)
        return self._vectors @ self._embedder.encode([self._query_prefix + query])[0]

    def max_dense(self, query: str) -> float:
        """Highest dense similarity of ``query`` to any chunk (0.0 for an empty index)."""
        scores = self.dense_scores(query)
        return float(scores.max()) if len(scores) else 0.0

    def dense_of(self, scores: np.ndarray, chunk_id: str) -> float:
        """The similarity (from ``dense_scores``) of one chunk, by id."""
        position = next((i for i, c in enumerate(self.chunks) if c.chunk_id == chunk_id), None)
        return float(scores[position]) if position is not None and len(scores) else 0.0

    def search(
        self, query: str, top_k: int, dense_weight: float, bm25_weight: float, min_score: float, reserve_prose: int = 0,
    ) -> list[Hit]:
        if not self.chunks or self._bm25 is None:
            return []
        query_vec = self._embedder.encode([self._query_prefix + query])[0]
        dense = self._vectors @ query_vec
        bm25 = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)

        dense_rank = np.empty(len(dense), dtype=int)
        dense_rank[np.argsort(-dense, kind="stable")] = np.arange(len(dense))
        bm25_rank = np.empty(len(bm25), dtype=int)
        bm25_rank[np.argsort(-bm25, kind="stable")] = np.arange(len(bm25))

        fused = dense_weight / (RRF_K + dense_rank + 1) + np.where(
            bm25 > 0, bm25_weight / (RRF_K + bm25_rank + 1), 0.0
        )
        hits = [
            Hit(self.chunks[i], float(fused[i]), float(dense[i]), float(bm25[i]))
            for i in np.argsort(-fused, kind="stable")
            if dense[i] >= min_score
        ]
        chosen = hits[:top_k]
        if reserve_prose > 0:
            chosen = reserve_slots(hits, top_k, reserve_prose)
        return chosen


def reserve_slots(hits: list[Hit], top_k: int, reserve_prose: int) -> list[Hit]:
    """Source diversity: ``reserve_prose`` of the ``top_k`` slots go to prose chunks when the ranking has them (above
    min_score), so many table rows cannot crowd the documents' sentences out. The kept order is the fused ranking."""
    chosen = hits[:top_k]
    need = min(reserve_prose, top_k) - sum(1 for h in chosen if h.chunk.kind == "prose")
    if need <= 0:
        return chosen
    extra = [h for h in hits[top_k:] if h.chunk.kind == "prose"][:need]
    if not extra:
        return chosen
    droppable = [h for h in reversed(chosen) if h.chunk.kind != "prose"][: len(extra)]
    keep = [h for h in chosen if h not in droppable] + extra
    return sorted(keep, key=lambda h: -h.score)

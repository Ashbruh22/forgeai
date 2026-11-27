"""The eval generator's view of the documents: text, chunks, tables and verified seed spans.

It reads only the requirements and the documents. It has no access to how the agent was
configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.ingest.chunking import chunk_document
from forgeai.ingest.loaders import LoadError, list_doc_files, load_document
from forgeai.ingest.models import Chunk, LoadedDoc
from forgeai.ingest.spans import SeedSpan
from forgeai.runner.retrieval import HybridIndex
from forgeai.textmatch import normalize

MIN_SEED_WORDS = 12


@dataclass
class Table:
    doc: str
    columns: list[str]
    rows: list[dict[str, str]]
    row_spans: list[SeedSpan]


@dataclass
class EvalCorpus:
    docs: dict[str, LoadedDoc]
    chunks: list[Chunk]
    index: HybridIndex
    tables: dict[str, Table] = field(default_factory=dict)
    embedder: Embedder | None = None

    @classmethod
    def load(cls, docs_dir: Path, config: ForgeConfig, embedder: Embedder) -> EvalCorpus:
        docs: dict[str, LoadedDoc] = {}
        chunks: list[Chunk] = []
        tables: dict[str, Table] = {}
        for path in list_doc_files(docs_dir):
            name = path.relative_to(docs_dir).as_posix()
            try:
                doc = load_document(path).model_copy(update={"file": name})
            except LoadError:
                continue
            docs[name] = doc
            chunks.extend(chunk_document(doc))
            if doc.file_type == "csv":
                frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                canonical, spans = doc.layout()
                row_spans = [
                    SeedSpan(doc=name, char_start=s[0], char_end=s[1], span_text=canonical[s[0] : s[1]])
                    for s in spans if s is not None
                ]
                tables[name] = Table(name, [str(c) for c in frame.columns], frame.to_dict("records"), row_spans)
        index = HybridIndex.build(
            chunks, embedder, config.embedding_model, cache_dir=config.index_cache_dir,
            query_prefix=config.embedding_query_prefix)
        return cls(docs, chunks, index, tables, embedder)

    # -- spans ------------------------------------------------------------------------------

    def locate(self, doc: str, text: str) -> tuple[int, int] | None:
        """Character range of ``text`` in the document's canonical text (whitespace-flexible)."""
        canonical = self.docs[doc].canonical_text if doc in self.docs else ""
        if not text.strip() or not canonical:
            return None
        found = canonical.find(text)
        if found >= 0:
            return found, found + len(text)
        pattern = r"\s+".join(re.escape(word) for word in text.split())
        match = re.search(pattern, canonical)
        return (match.start(), match.end()) if match else None

    def span_of(self, doc: str, text: str) -> SeedSpan | None:
        """A seed span for ``text`` whose ``span_text`` is read back from the document, never typed in."""
        located = self.locate(doc, text)
        if located is None:
            return None
        canonical = self.docs[doc].canonical_text
        return SeedSpan(doc=doc, char_start=located[0], char_end=located[1], span_text=canonical[located[0] : located[1]])

    def chunk_span(self, chunk: Chunk) -> SeedSpan | None:
        if chunk.char_start is None or chunk.char_end is None or chunk.char_end <= chunk.char_start:
            return None
        text = self.docs[chunk.doc].canonical_text[chunk.char_start : chunk.char_end]
        return SeedSpan(doc=chunk.doc, char_start=chunk.char_start, char_end=chunk.char_end, span_text=text)

    # -- seeds ------------------------------------------------------------------------------

    def seed_chunks(self, kind: str = "prose") -> list[Chunk]:
        """Chunks worth writing a question about (not headings-only, not schema)."""
        enough = MIN_SEED_WORDS if kind == "prose" else 4
        return [c for c in self.chunks if c.kind == kind and len(self._content(c, kind).split()) >= enough]

    @staticmethod
    def _content(chunk: Chunk, kind: str) -> str:
        """The chunk without its page-footer line (a footer is not content, but a long chunk may contain one)."""
        return re.sub(r".{0,120}?(?<!\w)Page \d+(?!\w)", "", chunk.text) if kind == "prose" else chunk.text

    def text_of(self, chunk: Chunk) -> str:
        return normalize(chunk.text)

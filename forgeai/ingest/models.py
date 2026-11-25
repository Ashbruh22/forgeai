"""Data models shared by ingestion, retrieval and the runner."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SEPARATOR = "\n\n"  # joins blocks in a document's canonical text


class Block(BaseModel):
    """One structural unit from a loaded file: a heading, a paragraph or a table row."""

    text: str
    heading_level: int | None = None  # 1..6 for headings, None for body text
    page: int | None = None
    row: int | None = None  # 1-based data row for tabular files


class LoadedDoc(BaseModel):
    file: str
    file_type: str
    blocks: list[Block]
    columns: list[str] = Field(default_factory=list)  # tabular files only
    page_count: int | None = None
    row_count: int | None = None

    def layout(self) -> tuple[str, list[tuple[int, int] | None]]:
        """Canonical text and each block's ``(start, end)`` character range in it.

        The canonical text is every non-empty block joined by a blank line. It depends only on
        the loader, never on chunking parameters, so offsets into it stay valid when the
        optimizer changes chunk size (spec 6.1 seed spans). Empty blocks map to ``None``.
        """
        parts: list[str] = []
        spans: list[tuple[int, int] | None] = []
        position = 0
        for block in self.blocks:
            text = block.text.strip()
            if not text:
                spans.append(None)
                continue
            if parts:
                position += len(SEPARATOR)
            spans.append((position, position + len(text)))
            parts.append(text)
            position += len(text)
        return SEPARATOR.join(parts), spans

    @property
    def canonical_text(self) -> str:
        return self.layout()[0]


class Chunk(BaseModel):
    """A retrievable, citable span. Citations are verified against ``text`` verbatim."""

    chunk_id: str  # "<file>#c<N>", stable for identical input and chunking params
    doc: str
    index: int
    text: str
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    row: int | None = None
    kind: Literal["prose", "row", "schema"] = "prose"
    # Character range in the document's canonical text (LoadedDoc.canonical_text);
    # None for synthetic chunks such as a table's schema chunk.
    char_start: int | None = None
    char_end: int | None = None

    @property
    def retrieval_text(self) -> str:
        """Text used for embedding/BM25: headings give context, citations ignore them."""
        prefix = " > ".join(self.heading_path)
        return f"{prefix}\n{self.text}" if prefix else self.text


class StructureProfile(BaseModel):
    """Computed by code (no LLM): how the document is shaped."""

    kind: Literal["prose", "tabular"]
    total_words: int
    n_blocks: int
    n_headings: int
    max_heading_depth: int
    avg_section_words: float


class DocRecord(BaseModel):
    """One entry of ``doc_inventory.json``."""

    file: str
    status: Literal["ok", "skipped"]
    reason: str = ""
    file_type: str = ""
    page_count: int | None = None
    row_count: int | None = None
    title: str = ""
    summary: str = ""
    structure: StructureProfile | None = None
    n_chunks: int = 0

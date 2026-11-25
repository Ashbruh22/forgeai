"""Data models shared by ingestion, retrieval and the runner."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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

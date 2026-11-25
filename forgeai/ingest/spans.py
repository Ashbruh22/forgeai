"""Seed spans: where in a source document an eval case's facts live (spec §6.1).

Offsets are character offsets into the document's canonical text
(``LoadedDoc.canonical_text``), which does not depend on chunking parameters. A retrieved
chunk is a gold hit when its range *overlaps* the seed range, so a smaller chunk size that
splits the span across two chunks still counts, and changing chunk size never invalidates
an eval set.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from forgeai.ingest.models import Chunk, LoadedDoc


class SeedSpan(BaseModel):
    doc: str
    char_start: int
    char_end: int
    span_text: str

    @model_validator(mode="after")
    def _ordered(self) -> SeedSpan:
        if not 0 <= self.char_start < self.char_end:
            raise ValueError("seed span needs 0 <= char_start < char_end")
        return self


def make_seed_span(doc: LoadedDoc, char_start: int, char_end: int) -> SeedSpan:
    """Build a seed span whose ``span_text`` is read from the canonical text (never typed in)."""
    return SeedSpan(
        doc=doc.file, char_start=char_start, char_end=char_end,
        span_text=doc.canonical_text[char_start:char_end],
    )


def span_matches_document(seed: SeedSpan, doc: LoadedDoc) -> bool:
    """True if the recorded ``span_text`` really is the text at those offsets."""
    return seed.doc == doc.file and doc.canonical_text[seed.char_start : seed.char_end] == seed.span_text


def chunk_overlaps_seed(chunk: Chunk, seed: SeedSpan) -> bool:
    """Gold-chunk test: same document and overlapping character ranges."""
    if chunk.doc != seed.doc or chunk.char_start is None or chunk.char_end is None:
        return False
    return chunk.char_start < seed.char_end and seed.char_start < chunk.char_end

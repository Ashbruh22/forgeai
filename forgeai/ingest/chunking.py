"""Chunking: heading-aware recursive splitting for prose, row-as-record for tables.

Chunk ids are ``<file>#c<N>`` and are stable for identical input and parameters. Every
chunk also records its character range in the document's canonical text, which is what
eval seed spans are measured in, so gold-chunk matching survives a change of chunk size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from forgeai.ingest.models import Chunk, LoadedDoc

TOKENS_PER_WORD = 1.3  # rough, tokenizer-free estimate; good enough for sizing


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD) + 1


@dataclass(frozen=True)
class Unit:
    """A piece of body text with its place in the canonical text."""

    text: str
    page: int | None
    start: int
    end: int


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def _split_oversized(paragraph: str, size_tokens: int) -> list[str]:
    """Split a paragraph longer than ``size_tokens`` by sentence, then by word window."""
    units: list[str] = []
    for sentence in _SENTENCE_END.split(paragraph):
        if estimate_tokens(sentence) <= size_tokens:
            units.append(sentence)
            continue
        words = sentence.split()
        step = max(1, int(size_tokens / TOKENS_PER_WORD))
        units.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    return units


def _block_units(text: str, page: int | None, start: int, size_tokens: int) -> list[Unit]:
    """Turn one block into units, locating each sub-span inside the block for offsets."""
    if estimate_tokens(text) <= size_tokens:
        return [Unit(text, page, start, start + len(text))]
    units: list[Unit] = []
    cursor = 0
    for piece in _split_oversized(text, size_tokens):
        found = text.find(piece, cursor)
        offset = found if found >= 0 else cursor
        units.append(Unit(piece, page, start + offset, start + min(offset + len(piece), len(text))))
        cursor = offset + len(piece)
    return units


def _tail_start(units: list[Unit], overlap_tokens: int) -> tuple[str, int | None]:
    """The trailing words (about ``overlap_tokens``) of a finished chunk and where they start."""
    wanted = max(0, int(overlap_tokens / TOKENS_PER_WORD))
    if not wanted:
        return "", None
    remaining = wanted
    words: list[str] = []
    start: int | None = None
    for unit in reversed(units):
        unit_words = unit.text.split()
        if len(unit_words) >= remaining:
            taken = unit_words[-remaining:]
            words = taken + words
            start = unit.end - len(" ".join(taken))
            break
        words = unit_words + words
        remaining -= len(unit_words)
        start = unit.start
    return " ".join(words), start


def _pack_section(
    units: list[Unit], size_tokens: int, overlap_tokens: int
) -> list[tuple[str, int | None, int, int]]:
    """Greedily pack units into ``(text, page, start, end)`` chunks of <= size_tokens + overlap."""
    chunks: list[tuple[str, int | None, int, int]] = []
    current: list[Unit] = []
    carry = ""
    carry_start: int | None = None
    tokens = 0

    def body() -> str:
        parts = ([carry] if carry else []) + [u.text for u in current]
        return " ".join(parts)

    def close() -> None:
        first = min([u.start for u in current] + ([carry_start] if carry_start is not None else []))
        chunks.append((body(), current[0].page, first, max(u.end for u in current)))

    for unit in units:
        unit_tokens = estimate_tokens(unit.text)
        if current and tokens + unit_tokens > size_tokens:
            close()
            carry, carry_start = _tail_start(current, overlap_tokens)
            current = []
            tokens = estimate_tokens(carry) if carry else 0
        current.append(unit)
        tokens += unit_tokens
    if current:
        close()
    return chunks


def chunk_document(
    doc: LoadedDoc,
    size_tokens: int = 350,
    overlap_tokens: int = 60,
    csv_mode: str = "row_as_record",
) -> list[Chunk]:
    """Chunk one loaded document according to its kind."""
    if doc.file_type == "csv":
        return _chunk_table(doc)
    return _chunk_prose(doc, size_tokens, overlap_tokens)


def _chunk_table(doc: LoadedDoc) -> list[Chunk]:
    schema_text = (
        f"Table {doc.file} has {doc.row_count} rows. Columns: {', '.join(doc.columns)}."
    )
    chunks = [Chunk(chunk_id=f"{doc.file}#c0", doc=doc.file, index=0, text=schema_text, kind="schema")]
    _, spans = doc.layout()
    for block, span in zip(doc.blocks, spans, strict=True):
        row = block.row or len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.file}#c{row}",
                doc=doc.file,
                index=row,
                text=block.text,
                row=row,
                kind="row",
                char_start=span[0] if span else None,
                char_end=span[1] if span else None,
            )
        )
    return chunks


def _chunk_prose(doc: LoadedDoc, size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    section: list[Unit] = []
    section_path: list[str] = []
    _, spans = doc.layout()

    def flush() -> None:
        for text, page, start, end in _pack_section(section, size_tokens, overlap_tokens):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.file}#c{index}",
                    doc=doc.file,
                    index=index,
                    text=text,
                    heading_path=list(section_path),
                    page=page,
                    char_start=start,
                    char_end=end,
                )
            )
        section.clear()

    for block, span in zip(doc.blocks, spans, strict=True):
        if span is None:
            continue
        text = block.text.strip()
        if block.heading_level is not None:
            flush()
            level = block.heading_level
            heading_stack[:] = heading_stack[: level - 1] + [text]
            section_path = list(heading_stack)
            continue
        section.extend(_block_units(text, block.page, span[0], size_tokens))
    flush()
    return chunks

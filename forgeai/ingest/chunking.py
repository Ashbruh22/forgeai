"""Chunking: heading-aware recursive splitting for prose, row-as-record for tables.

Chunk ids are ``<file>#c<N>`` and are stable for identical input and parameters,
so eval cases and citations can refer to them across runs.
"""

from __future__ import annotations

import re

from forgeai.ingest.models import Chunk, LoadedDoc

TOKENS_PER_WORD = 1.3  # rough, tokenizer-free estimate; good enough for sizing


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD) + 1


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


def _tail_words(text: str, overlap_tokens: int) -> str:
    n_words = max(0, int(overlap_tokens / TOKENS_PER_WORD))
    return " ".join(text.split()[-n_words:]) if n_words else ""


def _pack_section(
    units: list[tuple[str, int | None]], size_tokens: int, overlap_tokens: int
) -> list[tuple[str, int | None]]:
    """Greedily pack ``(text, page)`` units into chunks of <= size_tokens (plus overlap)."""
    chunks: list[tuple[str, int | None]] = []
    current: list[str] = []
    current_tokens = 0
    current_page: int | None = None
    carry = ""
    for text, page in units:
        tokens = estimate_tokens(text)
        if current and current_tokens + tokens > size_tokens:
            body = " ".join(current)
            chunks.append((body, current_page))
            carry = _tail_words(body, overlap_tokens)
            current = [carry] if carry else []
            current_tokens = estimate_tokens(carry) if carry else 0
            current_page = None
        if not current or current_page is None:
            current_page = current_page if current_page is not None else page
        current.append(text)
        current_tokens += tokens
    if current and (not carry or " ".join(current) != carry):
        chunks.append((" ".join(current), current_page))
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
    for block in doc.blocks:
        row = block.row or len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.file}#c{row}",
                doc=doc.file,
                index=row,
                text=block.text,
                row=row,
                kind="row",
            )
        )
    return chunks


def _chunk_prose(doc: LoadedDoc, size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    section: list[tuple[str, int | None]] = []
    section_path: list[str] = []

    def flush() -> None:
        packed = _pack_section(section, size_tokens, overlap_tokens)
        for text, page in packed:
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.file}#c{index}",
                    doc=doc.file,
                    index=index,
                    text=text,
                    heading_path=list(section_path),
                    page=page,
                )
            )
        section.clear()

    for block in doc.blocks:
        text = block.text.strip()
        if not text:
            continue
        if block.heading_level is not None:
            flush()
            level = block.heading_level
            heading_stack[:] = heading_stack[: level - 1] + [text]
            section_path = list(heading_stack)
            continue
        if estimate_tokens(text) > size_tokens:
            section.extend((unit, block.page) for unit in _split_oversized(text, size_tokens))
        else:
            section.append((text, block.page))
    flush()
    return chunks

"""Stage 1: ingest a folder, profile every document, build the topic map."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.ingest.chunking import chunk_document
from forgeai.ingest.loaders import LoadError, list_doc_files, load_document
from forgeai.ingest.models import Chunk, DocRecord
from forgeai.ingest.profile import structure_profile, summarize
from forgeai.ingest.topic_map import TopicMap, build_topic_map


@dataclass
class IngestResult:
    records: list[DocRecord]
    chunks: list[Chunk]
    topic_map: TopicMap


def ingest_folder(
    docs_dir: Path,
    config: ForgeConfig,
    embedder: Embedder,
    out_dir: Path | None = None,
    size_tokens: int = 350,
    overlap_tokens: int = 60,
) -> IngestResult:
    """Load every file under ``docs_dir``. Unparseable files are recorded as skipped."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")
    records: list[DocRecord] = []
    chunks: list[Chunk] = []
    for path in list_doc_files(docs_dir):
        name = path.relative_to(docs_dir).as_posix()
        try:
            doc = load_document(path).model_copy(update={"file": name})
        except LoadError as exc:
            records.append(DocRecord(file=name, status="skipped", reason=str(exc)))
            continue
        doc_chunks = chunk_document(doc, size_tokens, overlap_tokens)
        summary = summarize(doc, config)
        records.append(
            DocRecord(
                file=name,
                status="ok",
                file_type=doc.file_type,
                page_count=doc.page_count,
                row_count=doc.row_count,
                title=summary.title,
                summary=summary.summary,
                structure=structure_profile(doc),
                n_chunks=len(doc_chunks),
            )
        )
        chunks.extend(doc_chunks)
    topic_map = build_topic_map(chunks, embedder, config)
    result = IngestResult(records=records, chunks=chunks, topic_map=topic_map)
    if out_dir is not None:
        write_ingest_artifacts(result, out_dir)
    return result


def write_ingest_artifacts(result: IngestResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "doc_inventory.json").write_text(
        json.dumps([r.model_dump() for r in result.records], indent=2), encoding="utf-8"
    )
    (out_dir / "topic_map.json").write_text(
        result.topic_map.model_dump_json(indent=2), encoding="utf-8"
    )
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in result.chunks:
            handle.write(chunk.model_dump_json() + "\n")

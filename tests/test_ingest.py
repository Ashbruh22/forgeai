"""P1 exit test: all file types parse; a corrupt file is skipped without a crash."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forgeai.config import ForgeConfig, ModelSpec
from forgeai.ingest.chunking import chunk_document, estimate_tokens
from forgeai.ingest.loaders import LoadError, load_document
from forgeai.ingest.pipeline import ingest_folder
from forgeai.ingest.profile import structure_profile
from forgeai.llm import ChatResult, LLMClient, set_client
from tests.helpers import FakeEmbedder

PROSE = (
    "1. Annual Leave\n\n"
    "1.1 Full-time employees receive 24 days of paid annual leave per calendar year.\n\n"
    "1.2 Leave accrues monthly at 2 days per month and may be carried over up to 5 days.\n\n"
    "2. Sick Leave\n\n"
    "2.1 Employees receive 10 days of paid sick leave per year with a medical certificate."
)


class JsonTransport:
    """Answers any call with schema-valid JSON, chosen by the schema's fields."""

    def chat(self, *, model: str, messages: list[dict[str, str]], schema: dict[str, Any], options: dict[str, Any]) -> ChatResult:
        props = schema.get("properties", {})
        body = (
            {"title": "A Title", "summary": "A short summary."}
            if "title" in props
            else {"label": "A Topic", "description": "About a topic."}
        )
        return ChatResult(content=json.dumps(body))


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    import docx
    import pymupdf as fitz

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "policy.txt").write_text(PROSE, encoding="utf-8")
    (folder / "notes.md").write_text(
        "# Travel\n\n## Flights\n\nEconomy class is required for flights under 6 hours.\n\n"
        "## Hotels\n\nHotel stays are capped at USD 150 per night.\n",
        encoding="utf-8",
    )
    (folder / "page.html").write_text(
        "<html><body><script>x=1</script><h1>Conduct</h1><p>Be respectful.</p>"
        "<h2>IT use</h2><p>No personal software installs.</p></body></html>",
        encoding="utf-8",
    )
    document = docx.Document()
    document.add_heading("Expenses", level=1)
    document.add_paragraph("Claims must be filed within 30 days of travel.")
    document.save(str(folder / "expenses.docx"))
    (folder / "parts.csv").write_text(
        "part_no,description,unit_price_usd\nP-1,Gasket,12.50\nP-2,Piston ring,40.00\n",
        encoding="utf-8",
    )
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Leave Policy", fontsize=22)
    page.insert_text((72, 120), "Employees receive 24 days of annual leave each year.", fontsize=11)
    page.insert_text((72, 140), "Unused leave expires on 31 March of the following year.", fontsize=11)
    pdf.save(str(folder / "leave.pdf"))
    pdf.close()
    # Failure cases that must not crash the run:
    (folder / "corrupt.pdf").write_bytes(b"%PDF-1.7 this is not really a pdf \x00\x01\x02")
    (folder / "empty.txt").write_text("   \n\n", encoding="utf-8")
    (folder / "weird.xyz").write_text("unsupported", encoding="utf-8")
    return folder


@pytest.fixture
def config(tmp_path: Path) -> ForgeConfig:
    return ForgeConfig(
        agent_model=ModelSpec(name="qwen3:8b"),
        judge_model=ModelSpec(name="gemma3:4b"),
        cache_dir=tmp_path / "cache",
    )


@pytest.mark.parametrize(
    ("name", "file_type", "expect"),
    [
        ("policy.txt", "txt", "24 days of paid annual leave"),
        ("notes.md", "md", "Economy class"),
        ("page.html", "html", "Be respectful."),
        ("expenses.docx", "docx", "within 30 days"),
        ("leave.pdf", "pdf", "24 days of annual leave"),
        ("parts.csv", "csv", "part_no: P-2 | description: Piston ring"),
    ],
)
def test_every_supported_type_parses(docs: Path, name: str, file_type: str, expect: str) -> None:
    loaded = load_document(docs / name)
    assert loaded.file_type == file_type
    assert expect in " ".join(b.text for b in loaded.blocks)


def test_script_content_is_not_extracted_from_html(docs: Path) -> None:
    assert "x=1" not in " ".join(b.text for b in load_document(docs / "page.html").blocks)


def test_pdf_heading_detected_by_font_size(docs: Path) -> None:
    blocks = load_document(docs / "leave.pdf").blocks
    assert any(b.heading_level and "Leave Policy" in b.text for b in blocks)


@pytest.mark.parametrize(
    ("name", "reason"),
    [("corrupt.pdf", "PDF"), ("empty.txt", "no extractable text"), ("weird.xyz", "unsupported")],
)
def test_bad_files_raise_load_error(docs: Path, name: str, reason: str) -> None:
    with pytest.raises(LoadError, match=reason):
        load_document(docs / name)


def test_prose_chunks_follow_headings_and_have_stable_ids(docs: Path) -> None:
    doc = load_document(docs / "policy.txt")
    first = chunk_document(doc)
    second = chunk_document(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_id for c in first] == ["policy.txt#c0", "policy.txt#c1"]
    assert first[0].heading_path == ["1. Annual Leave"]
    assert "24 days" in first[0].text and "sick" not in first[0].text.lower()
    assert first[1].heading_path == ["2. Sick Leave"]


def test_long_section_is_split_with_overlap(tmp_path: Path) -> None:
    sentences = [f"Sentence number {i} says something about policy item {i}." for i in range(80)]
    path = tmp_path / "long.txt"
    path.write_text("1. Big Section\n\n" + " ".join(sentences), encoding="utf-8")
    chunks = chunk_document(load_document(path), size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    assert all(estimate_tokens(c.text) <= 100 + 20 + 5 for c in chunks)
    # Consecutive chunks share trailing words (overlap), so no sentence is orphaned.
    tail = chunks[0].text.split()[-3:]
    assert " ".join(tail) in chunks[1].text


def test_csv_is_row_as_record_with_schema_chunk(docs: Path) -> None:
    chunks = chunk_document(load_document(docs / "parts.csv"))
    assert [c.kind for c in chunks] == ["schema", "row", "row"]
    assert "Columns: part_no, description, unit_price_usd" in chunks[0].text
    assert chunks[2].chunk_id == "parts.csv#c2" and "Piston ring" in chunks[2].text


def test_structure_profile(docs: Path) -> None:
    profile = structure_profile(load_document(docs / "policy.txt"))
    assert profile.kind == "prose" and profile.n_headings == 2
    assert structure_profile(load_document(docs / "parts.csv")).kind == "tabular"


def test_ingest_folder_skips_bad_files_and_writes_artifacts(
    docs: Path, config: ForgeConfig, tmp_path: Path
) -> None:
    set_client(LLMClient(config, transport=JsonTransport(), trace_path=tmp_path / "trace.jsonl"))
    out = tmp_path / "out"
    result = ingest_folder(docs, config, FakeEmbedder(), out_dir=out)

    by_file = {r.file: r for r in result.records}
    assert {f for f, r in by_file.items() if r.status == "ok"} == {
        "policy.txt", "notes.md", "page.html", "expenses.docx", "parts.csv", "leave.pdf",
    }
    for bad in ("corrupt.pdf", "empty.txt", "weird.xyz"):
        assert by_file[bad].status == "skipped" and by_file[bad].reason
    assert by_file["policy.txt"].title == "A Title" and by_file["policy.txt"].n_chunks == 2

    inventory = json.loads((out / "doc_inventory.json").read_text(encoding="utf-8"))
    assert len(inventory) == 9
    topic_map = json.loads((out / "topic_map.json").read_text(encoding="utf-8"))
    assert topic_map["topics"] and all(t["chunk_ids"] for t in topic_map["topics"])
    chunk_lines = (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(chunk_lines) == len(result.chunks)
    set_client(None)


def test_missing_folder_is_an_error(config: ForgeConfig, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_folder(tmp_path / "nope", config, FakeEmbedder())

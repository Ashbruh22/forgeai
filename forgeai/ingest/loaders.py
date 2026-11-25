"""File loaders: PDF, DOCX, MD, TXT, HTML, CSV -> structured ``Block`` lists.

Loaders never silently drop a file: a failure raises ``LoadError`` with a reason,
and the caller records the file as ``skipped`` in ``doc_inventory.json``.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from pathlib import Path

from forgeai.ingest.models import Block, LoadedDoc

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".html", ".htm", ".csv"}

_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")


class LoadError(Exception):
    """A file could not be parsed. The message is shown in the inventory."""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _looks_like_heading(line: str) -> bool:
    """Heuristic for plain text: short, unpunctuated, numbered or all-caps lines."""
    if not line or len(line) > 90:
        return False
    if _NUMBERED_HEADING.match(line):
        return len(line.split()) <= 12 and not line.endswith((".", ",", ";"))
    return line.isupper() and len(line.split()) <= 10


def _heading_level_from_number(line: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)", line)
    return min(match.group(1).count(".") + 1, 6) if match else 1


def _paragraphs(text: str, page: int | None = None) -> list[Block]:
    blocks: list[Block] = []
    for raw in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [_clean(line) for line in raw.split("\n") if _clean(line)]
        if not lines:
            continue
        if len(lines) == 1 and _looks_like_heading(lines[0]):
            level = _heading_level_from_number(lines[0]) if _NUMBERED_HEADING.match(lines[0]) else 1
            blocks.append(Block(text=lines[0], heading_level=level, page=page))
        else:
            blocks.append(Block(text=" ".join(lines), page=page))
    return blocks


def load_txt(path: Path) -> LoadedDoc:
    text = path.read_text(encoding="utf-8", errors="replace")
    return LoadedDoc(file=path.name, file_type="txt", blocks=_paragraphs(text))


def load_md(path: Path) -> LoadedDoc:
    blocks: list[Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(Block(text=_clean(" ".join(paragraph))))
            paragraph.clear()

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if heading:
            flush()
            blocks.append(Block(text=_clean(heading.group(2)), heading_level=len(heading.group(1))))
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.strip())
    flush()
    return LoadedDoc(file=path.name, file_type="md", blocks=blocks)


def load_html(path: Path) -> LoadedDoc:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks: list[Block] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre"]):
        text = _clean(element.get_text(" "))
        if not text:
            continue
        if element.name.startswith("h") and len(element.name) == 2:
            blocks.append(Block(text=text, heading_level=int(element.name[1])))
        else:
            blocks.append(Block(text=text))
    return LoadedDoc(file=path.name, file_type="html", blocks=blocks)


def load_docx(path: Path) -> LoadedDoc:
    import docx

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx raises many types for bad zips/xml
        raise LoadError(f"not a readable DOCX: {exc}") from exc
    blocks: list[Block] = []
    for paragraph in document.paragraphs:
        text = _clean(paragraph.text)
        if not text:
            continue
        style = (paragraph.style.name or "") if paragraph.style is not None else ""
        heading = re.match(r"^Heading (\d)$", style)
        if heading:
            blocks.append(Block(text=text, heading_level=int(heading.group(1))))
        elif style == "Title":
            blocks.append(Block(text=text, heading_level=1))
        else:
            blocks.append(Block(text=text))
    for table in document.tables:
        for row in table.rows:
            cells = [_clean(cell.text) for cell in row.cells]
            if any(cells):
                blocks.append(Block(text=" | ".join(cells)))
    return LoadedDoc(file=path.name, file_type="docx", blocks=blocks)


def load_pdf(path: Path) -> LoadedDoc:
    import pymupdf as fitz  # PyMuPDF

    try:
        document = fitz.open(str(path))
    except Exception as exc:
        raise LoadError(f"not a readable PDF: {exc}") from exc
    blocks: list[Block] = []
    try:
        sizes: list[float] = []
        pages = []
        for page in document:
            data = page.get_text("dict")
            pages.append(data)
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["text"].strip():
                            sizes.append(span["size"])
        body_size = statistics.median(sizes) if sizes else 0.0
        for page_no, data in enumerate(pages, start=1):
            for block in data.get("blocks", []):
                lines = block.get("lines", [])
                if not lines:
                    continue
                text = _clean(
                    " ".join(span["text"] for line in lines for span in line.get("spans", []))
                )
                if not text:
                    continue
                spans = [s for line in lines for s in line.get("spans", []) if s["text"].strip()]
                max_size = max((s["size"] for s in spans), default=0.0)
                all_bold = bool(spans) and all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
                is_heading = len(text) <= 100 and (
                    (body_size and max_size >= body_size * 1.15)
                    or (all_bold and len(lines) == 1 and not text.endswith("."))
                )
                if is_heading:
                    level = 1 if body_size and max_size >= body_size * 1.5 else 2
                    if _NUMBERED_HEADING.match(text):
                        level = max(level, _heading_level_from_number(text))
                    blocks.append(Block(text=text, heading_level=level, page=page_no))
                else:
                    blocks.append(Block(text=text, page=page_no))
        page_count = document.page_count
    finally:
        document.close()
    return LoadedDoc(file=path.name, file_type="pdf", blocks=blocks, page_count=page_count)


def load_csv(path: Path) -> LoadedDoc:
    import pandas as pd

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise LoadError(f"not a readable CSV: {exc}") from exc
    columns = [str(c) for c in frame.columns]
    blocks = [
        Block(
            text=" | ".join(f"{col}: {str(value).strip()}" for col, value in row.items()),
            row=i,
        )
        for i, (_, row) in enumerate(frame.iterrows(), start=1)
    ]
    return LoadedDoc(
        file=path.name, file_type="csv", blocks=blocks, columns=columns, row_count=len(frame)
    )


_LOADERS: dict[str, Callable[[Path], LoadedDoc]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".md": load_md,
    ".markdown": load_md,
    ".txt": load_txt,
    ".html": load_html,
    ".htm": load_html,
    ".csv": load_csv,
}


def load_document(path: Path) -> LoadedDoc:
    """Load one file or raise ``LoadError`` with a human-readable reason."""
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise LoadError(f"unsupported file type {suffix or '(none)'!r}")
    try:
        doc = loader(path)
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"{type(exc).__name__}: {exc}") from exc
    if not any(block.text.strip() for block in doc.blocks):
        raise LoadError("no extractable text")
    return doc


def list_doc_files(docs_dir: Path) -> list[Path]:
    """All files under ``docs_dir`` (recursive, sorted, hidden files and folders skipped)."""
    return sorted(
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(docs_dir).parts)
    )

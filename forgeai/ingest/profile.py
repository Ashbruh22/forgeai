"""Document profiling: structure profile (code) and short summary (LLM + fallback)."""

from __future__ import annotations

import re

from pydantic import BaseModel

from forgeai.config import ForgeConfig
from forgeai.ingest.models import LoadedDoc, StructureProfile
from forgeai.llm import call_structured

EXCERPT_CHARS = 3500


class DocSummary(BaseModel):
    title: str
    summary: str


def structure_profile(doc: LoadedDoc) -> StructureProfile:
    """Compute how the document is shaped, with no LLM involved."""
    headings = [b for b in doc.blocks if b.heading_level is not None]
    body_words = sum(len(b.text.split()) for b in doc.blocks if b.heading_level is None)
    n_sections = max(len(headings), 1)
    return StructureProfile(
        kind="tabular" if doc.file_type == "csv" else "prose",
        total_words=sum(len(b.text.split()) for b in doc.blocks),
        n_blocks=len(doc.blocks),
        n_headings=len(headings),
        max_heading_depth=max((b.heading_level or 0) for b in headings) if headings else 0,
        avg_section_words=round(body_words / n_sections, 1),
    )


def _excerpt(doc: LoadedDoc) -> str:
    if doc.file_type == "csv":
        head = "\n".join(b.text for b in doc.blocks[:6])
        return f"Columns: {', '.join(doc.columns)}\nRows: {doc.row_count}\nFirst rows:\n{head}"[
            :EXCERPT_CHARS
        ]
    return "\n".join(b.text for b in doc.blocks)[:EXCERPT_CHARS]


def _fallback_summary(doc: LoadedDoc) -> DocSummary:
    """Deterministic default used when the model's output stays invalid."""
    heading = next((b.text for b in doc.blocks if b.heading_level is not None), "")
    title = heading or re.sub(r"[_-]+", " ", doc.file.rsplit(".", 1)[0]).strip().title()
    body = " ".join(b.text for b in doc.blocks if b.heading_level is None)
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return DocSummary(title=title, summary=" ".join(sentences[:3])[:500])


def summarize(doc: LoadedDoc, config: ForgeConfig) -> DocSummary:
    model = config.agent_model
    return call_structured(
        role="ingest.summary",
        prompt_id="ingest/summary",
        variables={"file_name": doc.file, "file_type": doc.file_type, "excerpt": _excerpt(doc)},
        schema=DocSummary,
        model=model.name,
        temperature=model.temperature,
        seed=model.seed,
        fallback=lambda: _fallback_summary(doc),
    )

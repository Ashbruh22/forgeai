"""The answer output contract (``contracts/answer_v1.json``) and its checks.

``status`` makes refusal machine-readable. Each citation carries a verbatim ``quote``
that is verified against the cited chunk (deterministic, no LLM).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, Field

from forgeai.ingest.models import Chunk
from forgeai.textmatch import is_verbatim

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "contracts" / "answer_v1.json"

Status = Literal["answered", "refused", "needs_clarification"]


class Citation(BaseModel):
    doc: str
    chunk_id: str
    quote: str


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class Answer(BaseModel):
    status: Status
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    clarifying_question: str | None = None


def load_contract_schema(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_issues(answer: Answer, schema: dict[str, Any] | None = None) -> list[str]:
    """Validate the answer against the committed JSON schema."""
    validator = jsonschema.Draft202012Validator(schema or load_contract_schema())
    return [
        f"schema: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
        for e in validator.iter_errors(answer.model_dump(mode="json"))
    ]


def citation_issues(answer: Answer, chunks_by_id: dict[str, Chunk]) -> list[str]:
    """Every citation must name a real chunk of that document and quote it verbatim."""
    issues: list[str] = []
    for citation in answer.citations:
        chunk = chunks_by_id.get(citation.chunk_id)
        if chunk is None:
            issues.append(f"citation: chunk_id {citation.chunk_id!r} does not exist")
        elif chunk.doc != citation.doc:
            issues.append(
                f"citation: chunk {citation.chunk_id!r} belongs to {chunk.doc!r}, not {citation.doc!r}"
            )
        elif not is_verbatim(citation.quote, chunk.text):
            issues.append(
                f"citation: quote is not a verbatim span of chunk {citation.chunk_id!r}: "
                f"{citation.quote[:80]!r}"
            )
    return issues


def contract_issues(answer: Answer, chunks_by_id: dict[str, Chunk]) -> list[str]:
    """All deterministic contract problems; an empty list means the answer is valid."""
    issues = schema_issues(answer) + citation_issues(answer, chunks_by_id)
    if answer.status == "needs_clarification" and not (answer.clarifying_question or "").strip():
        issues.append("status needs_clarification requires a clarifying_question")
    if answer.status != "needs_clarification" and not answer.answer.strip():
        issues.append("answer text is empty")
    return issues

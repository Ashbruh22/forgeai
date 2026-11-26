"""Eval case schema (spec §6.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from forgeai.ingest.spans import SeedSpan

CaseType = Literal["answerable", "multidoc", "ambiguous", "out_of_scope", "tool_required"]
Status = Literal["answered", "refused", "needs_clarification"]
FactKind = Literal["number", "phrase", "date", "snapshot_date"]


class KeyFact(BaseModel):
    text: str
    kind: FactKind = "phrase"


class Interpretation(BaseModel):
    """One valid reading of an ambiguous question, anchored in the corpus."""

    reading: str
    discriminators: list[str]  # words that would select this reading; the question must not contain them
    key_facts: list[KeyFact]
    seed_span: SeedSpan


class EvalCase(BaseModel):
    id: str
    type: CaseType
    split: Literal["dev", "test", ""] = ""
    question: str
    expected_status: Status
    expected_sources: list[str] = Field(default_factory=list)
    key_facts: list[KeyFact] = Field(default_factory=list)
    seed_spans: list[SeedSpan] = Field(default_factory=list)  # character offsets, not chunk ids
    interpretations: list[Interpretation] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    # Where a refusal / answer expectation comes from, for the consistency check.
    origin: str = ""  # must_refuse | gap | near_miss | negative_tool | topic | snapshot | computed | lookup
    topic_ref: str = ""
    tags: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)


class RejectedCase(BaseModel):
    case_type: str
    question: str
    reason: str
    detail: str = ""

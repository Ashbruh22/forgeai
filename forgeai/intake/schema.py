"""Requirements schema (spec §3.3) and the interview/guard records that feed it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["resolved_from_brief", "resolved_from_docs", "unresolved"]
Coverage = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
GapKind = Literal["unsupported_topic", "live_data", "capability", "freshness", "partial_coverage"]
GapAction = Literal["refuse_with_redirect", "request_documents", "warn"]


class Topic(BaseModel):
    id: str
    topic: str
    priority: Literal["high", "medium", "low"] = "medium"
    coverage: Coverage | None = None  # filled by the groundability guard
    evidence: list[str] = Field(default_factory=list)  # chunk ids with a verified supporting span
    # Set when the topic was split: only these aspects are supported; the rest became gaps.
    supported_aspects: list[str] = Field(default_factory=list)
    # ISO date of the static snapshot the answer comes from, if the evidence carries one.
    snapshot_as_of: str | None = None


class RefuseTopic(BaseModel):
    id: str
    topic: str
    reason: str
    redirect: str


class Scope(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    in_: list[str] = Field(default_factory=list, alias="in")
    out: list[str] = Field(default_factory=list)


class Assumption(BaseModel):
    field: str
    value: str
    source: str  # always starts with "ASSUMPTION"


class Gap(BaseModel):
    id: str
    kind: GapKind
    description: str
    recommended_action: GapAction
    topic_id: str | None = None
    aspect: str | None = None  # the unsupported part of a split topic
    evidence: list[str] = Field(default_factory=list)  # chunk ids, or the brief sentence that triggered it
    redirect: str = ""


class RejectedAssumption(BaseModel):
    """A self-answered assumption dropped because the brief or the corpus contradicts it."""

    field: str
    item: str
    conflict_with: Literal["brief", "corpus"]
    detail: str
    evidence: list[str] = Field(default_factory=list)


class Requirements(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    agent_name: str
    purpose: str
    audience: str
    tone: str
    scope: Scope
    must_answer_topics: list[Topic] = Field(default_factory=list)
    must_refuse_topics: list[RefuseTopic] = Field(default_factory=list)
    capabilities_needed: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    assumptions_rejected: list[RejectedAssumption] = Field(default_factory=list)
    interview_log: str = "questions.json"


# ---- LLM draft (stage 2, step 1) -----------------------------------------------------------


class ScalarField(BaseModel):
    value: str = ""
    confidence: Confidence = "unresolved"


class ListField(BaseModel):
    values: list[str] = Field(default_factory=list)
    confidence: Confidence = "unresolved"


class RequirementsDraft(BaseModel):
    """What the model drafts from the brief + topic map, with a confidence per field."""

    agent_name: ScalarField = Field(default_factory=ScalarField)
    purpose: ScalarField = Field(default_factory=ScalarField)
    audience: ScalarField = Field(default_factory=ScalarField)
    tone: ScalarField = Field(default_factory=ScalarField)
    scope_in: ListField = Field(default_factory=ListField)
    scope_out: ListField = Field(default_factory=ListField)
    must_answer: ListField = Field(default_factory=ListField)
    must_refuse: ListField = Field(default_factory=ListField)
    capabilities: ListField = Field(default_factory=ListField)


# ---- Interview records ---------------------------------------------------------------------


class Question(BaseModel):
    id: str
    field: str  # the requirement field this question resolves
    text: str
    kind: Literal["interview", "gap"] = "interview"


class Answer(BaseModel):
    question_id: str
    field: str
    text: str
    answered_by: Literal["user", "ASSUMPTION"]

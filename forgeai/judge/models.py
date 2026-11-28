"""Records the judge works on: one agent run per eval case, and the verdicts written for it."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forgeai.runner.contract import Answer


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc: str
    char_start: int | None = None
    char_end: int | None = None
    text: str
    dense: float = 0.0


class CitedChunk(BaseModel):
    """Where a citation points (resolved by the runner from its index, never taken from the model)."""

    chunk_id: str
    doc: str
    char_start: int | None = None
    char_end: int | None = None


class CaseRun(BaseModel):
    """What the agent did for one eval case: the answer, what it retrieved and what it cited."""

    case_id: str
    answer: Answer
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    cited: list[CitedChunk] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)  # deterministic contract problems from the runner
    latency_ms: float = 0.0
    search_query: str = ""


class FactCheck(BaseModel):
    fact: str
    kind: str
    matched: bool
    method: str = ""  # exact | number | date | quantity | fuzzy | tool_result | judge_upgrade | none
    upgraded_by_judge: bool = False
    quote: str = ""  # the verified span of the answer a judge upgrade rests on
    note: str = ""


class Layer1(BaseModel):
    contract_valid: bool
    contract_issues: list[str] = Field(default_factory=list)
    status_ok: bool
    status: str
    citations_total: int = 0
    citations_valid: int = 0
    citation_issues: list[str] = Field(default_factory=list)
    source_ok: bool | None = None  # None: the case has no source requirement
    sources_missing: list[str] = Field(default_factory=list)
    facts: list[FactCheck] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list)
    tool_ok: bool | None = None
    grounded_ok: bool | None = None  # an optional-tool lookup: right value grounded in a cited row or in a tool result
    tool_notes: list[str] = Field(default_factory=list)
    retrieval_hit: bool | None = None
    retrieval_rank: int | None = None  # 1-based rank of the first retrieved chunk overlapping a seed span
    retrieval_docs_hit: list[str] = Field(default_factory=list)
    interpretation_facts: list[list[FactCheck]] = Field(default_factory=list)
    interpretations_covered: int | None = None
    interpretations_total: int | None = None
    snapshot_date_ok: bool | None = None
    refusal_has_figures: bool = False


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    quote: str = ""
    reason: str = ""
    quote_verified: bool = False
    verdict: str = "supported"  # supported | unsupported | contradicted
    missing: str = ""  # what the evidence lacks, or what it says instead
    kind: str = "text"  # numeric: decided by code; text: decided by the judge model
    answer_bearing: bool = True
    nli: dict[str, float] = Field(default_factory=dict)


class Layer2(BaseModel):
    ran: bool = False
    claims: list[ClaimVerdict] = Field(default_factory=list)
    unsupported: int = 0  # answer-bearing claims that are unsupported or contradicted: these fail the case
    minor_unsupported: list[str] = Field(default_factory=list)  # unsupported claims that bear no answer: logged, low confidence
    duplicates_dropped: int = 0
    refusal_claims: list[str] = Field(default_factory=list)
    upgrades: list[dict[str, Any]] = Field(default_factory=list)  # fact-equivalence attempts and their outcome
    notes: list[str] = Field(default_factory=list)


class CaseScore(BaseModel):
    case_id: str
    type: str
    split: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)  # which checks failed (facts, not a failure classification)
    layer1: Layer1
    layer2: Layer2
    passed_without_layer2_upgrades: bool = True
    expected_status: str = ""
    disagreements: list[dict[str, Any]] = Field(default_factory=list)
    low_confidence: bool = False
    question: str = ""
    answer_status: str = ""
    answer_text: str = ""

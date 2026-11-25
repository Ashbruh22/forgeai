"""Stages 1-3 end to end: ingest, interview, groundability guard -> ``requirements.yaml``.

Writes ``00_intake/`` and ``01_guard/`` exactly as laid out in the spec (§10).

Precedence when sources disagree: **brief > corpus evidence > assumptions**. A self-answered
assumption may add refusals for things the corpus does not cover, but may not exclude a topic
the brief asks for or the corpus supports; such assumptions are dropped and recorded under
``assumptions_rejected`` with the evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from forgeai.config import ForgeConfig
from forgeai.embeddings import Embedder
from forgeai.guard.guard import GuardResult, TopicCoverage, probe_topics, run_guard
from forgeai.guard.snapshot import find_snapshot_date
from forgeai.ingest.models import Chunk
from forgeai.ingest.pipeline import IngestResult, ingest_folder
from forgeai.intake import interviewer as iv
from forgeai.intake.render import render_gaps, render_requirements
from forgeai.intake.schema import (
    Answer, Assumption, Gap, GapAction, Question, RefuseTopic, RejectedAssumption, Requirements,
    RequirementsDraft, Scope, Topic,
)
from forgeai.intake.structural import detect_structural_gaps
from forgeai.runner.retrieval import HybridIndex
from forgeai.textmatch import content_terms, shares_content_term
from forgeai.tools import REGISTRY

AskFn = Callable[[Question], str]
DEFAULTS = iv.DEFAULTS
NON_INTERACTIVE = "ASSUMPTION (non-interactive mode)"
BUDGET_EXHAUSTED = "ASSUMPTION (not asked: question budget spent, default used)"
NOT_STATED = "ASSUMPTION (not stated by the requester, default used)"


def parse_gap_action(answer: str) -> GapAction:
    """Deterministic reading of the requester's reply to the gap question."""
    return "request_documents" if re.search(r"\b(documents?|add|provide|upload|will supply)\b", answer.lower()) else "refuse_with_redirect"


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", text.strip())[0][:300]


def _interview(
    brief: str, draft: RequirementsDraft, ingest: IngestResult, config: ForgeConfig,
    ask: AskFn | None, budget: iv.QuestionBudget,
) -> tuple[RequirementsDraft, list[Answer], list[Assumption], dict[str, list[str]]]:
    """Ask about unresolved fields. Also returns the list items that came from self-answers."""
    answers: list[Answer] = []
    assumptions: list[Assumption] = []
    assumed_items: dict[str, list[str]] = {}
    documents = iv.documents_summary(ingest.records) + "\n" + iv.topics_summary(ingest.topic_map)
    for field in iv.unresolved_fields(draft)[: budget.interview_slots]:
        question = iv.build_question(field, len(budget.asked) + 1, brief, config)
        budget.register(question)
        text = ask(question) if ask else iv.self_answer(question, brief, documents, config)
        by = "user" if ask else "ASSUMPTION"
        answers.append(Answer(question_id=question.id, field=field, text=text, answered_by=by))
        if not text.strip():
            continue  # no answer: falls through to the default below
        target = getattr(draft, field)
        if field in iv.LIST_FIELDS:
            target.values, target.confidence = iv.answer_to_items(question, text, config), "resolved_from_brief"
            if by == "ASSUMPTION":
                assumed_items[field] = list(target.values)
        else:
            target.value, target.confidence = text.strip(), "resolved_from_brief"
        if by == "ASSUMPTION":
            assumptions.append(Assumption(field=field, value=text.strip(), source=f"{NON_INTERACTIVE}: {question.id} {question.text}"))
    return draft, answers, assumptions, assumed_items


def _fill_defaults(
    draft: RequirementsDraft, brief: str, ingest: IngestResult, assumptions: list[Assumption],
    asked_fields: set[str],
) -> None:
    """Every field still unresolved gets a default AND an ASSUMPTION record."""

    def note(field: str, value: str, why: str) -> None:
        assumptions.append(Assumption(field=field, value=value or "(empty)", source=why))

    for name in ("audience", "tone"):
        scalar = getattr(draft, name)
        if scalar.confidence == "unresolved":
            scalar.value, scalar.confidence = DEFAULTS["fields"][name], "resolved_from_docs"
            note(name, scalar.value, NOT_STATED if name in asked_fields else BUDGET_EXHAUSTED)
    for name in iv.LIST_FIELDS:
        listed = getattr(draft, name)
        if listed.confidence == "unresolved":
            listed.values, listed.confidence = list(DEFAULTS["fields"][name]), "resolved_from_docs"
            note(name, "; ".join(listed.values) or "(nothing)", NOT_STATED if name in asked_fields else BUDGET_EXHAUSTED)
    if not draft.agent_name.value.strip():
        draft.agent_name.value = DEFAULTS["fields"]["agent_name"]
        note("agent_name", draft.agent_name.value, "ASSUMPTION (model gave no name, default used)")
    if not draft.purpose.value.strip():
        draft.purpose.value = _first_sentence(brief)
        note("purpose", draft.purpose.value, "ASSUMPTION (purpose taken from the first sentence of the brief)")
    if not draft.must_answer.values:
        draft.must_answer.values = [t.label for t in ingest.topic_map.topics]
        note("must_answer_topics", "; ".join(draft.must_answer.values), "ASSUMPTION (topics taken from the document topic map)")


def _assemble(draft: RequirementsDraft, assumptions: list[Assumption]) -> Requirements:
    allowed = {"document_qa"} | {spec.capability for spec in REGISTRY.values()}
    capabilities = ["document_qa"] + [c for c in dict.fromkeys(draft.capabilities.values) if c in allowed and c != "document_qa"]
    return Requirements(
        agent_name=draft.agent_name.value.strip(), purpose=draft.purpose.value.strip(),
        audience=draft.audience.value.strip(), tone=draft.tone.value.strip(),
        scope=Scope(**{"in": draft.scope_in.values, "out": draft.scope_out.values}),
        must_answer_topics=[Topic(id=f"T{i}", topic=t, priority="high") for i, t in enumerate(draft.must_answer.values, 1)],
        must_refuse_topics=[
            RefuseTopic(id=f"R{i}", topic=t, reason=DEFAULTS["refuse_reason"], redirect=DEFAULTS["gap_redirect"])
            for i, t in enumerate(draft.must_refuse.values, 1)
        ],
        capabilities_needed=capabilities, assumptions=assumptions,
    )


# ---- precedence: brief > corpus > assumptions ----------------------------------------------


def reject_conflicting_assumptions(
    req: Requirements, assumed_items: dict[str, list[str]], index: HybridIndex, config: ForgeConfig,
) -> None:
    """Drop self-answered exclusions that contradict the brief or the corpus, and log why.

    The brief's own topics (must-answer topics and scope) always win. For the rest, an
    assumed exclusion is probed like a topic: if the corpus can answer questions about it, the
    corpus wins and the assumption is rejected. Exclusions about things the corpus does not
    cover are exactly what assumptions are allowed to add.
    """
    brief_topics = [t.topic for t in req.must_answer_topics]
    if "scope_in" not in assumed_items:
        brief_topics += req.scope.in_
    candidates = [(field, item) for field in ("scope_out", "must_refuse") for item in assumed_items.get(field, [])]
    if not candidates:
        return
    rejected: list[RejectedAssumption] = []
    to_probe: list[tuple[str, str]] = []
    for field, item in candidates:
        clash = next((t for t in brief_topics if shares_content_term(item, t)), None)
        if clash:
            rejected.append(RejectedAssumption(
                field=field, item=item, conflict_with="brief", evidence=[],
                detail=f"overlaps the topic the brief asks for: {clash!r}"))
        else:
            to_probe.append((field, item))
    if to_probe:
        coverage, _, _, _ = probe_topics(
            [(f"X{i}", item) for i, (_, item) in enumerate(to_probe)], req.purpose, index, config,
            with_near_miss=False)
        for (field, item), cov in zip(to_probe, coverage, strict=True):
            if cov.verdict != "UNSUPPORTED":
                spans = [f"{p.chunk_id}: {p.span}" for p in cov.probes if p.supported]
                rejected.append(RejectedAssumption(
                    field=field, item=item, conflict_with="corpus", evidence=spans,
                    detail=f"the documents answer questions about it ({cov.verdict}); excluding it would "
                           "refuse something the corpus supports"))
    dropped = {(r.field, r.item) for r in rejected}
    req.scope.out = [i for i in req.scope.out if ("scope_out", i) not in dropped]
    kept = [t for t in req.must_refuse_topics if ("must_refuse", t.topic) not in dropped]
    req.must_refuse_topics = [t.model_copy(update={"id": f"R{n}"}) for n, t in enumerate(kept, 1)]
    req.assumptions_rejected = rejected


# ---- applying the guard --------------------------------------------------------------------


_REQUEST_PREFIX = re.compile(r"^(?:it|the assistant|they|this)\s+(?:should|must|needs? to|has to|will)\s+(?:also\s+)?", re.IGNORECASE)
TIE_OVERLAP = 0.5  # share of a topic's content terms that the live clause must contain


def tie_live_gaps(topics: list[Topic], structural: list[Gap]) -> frozenset[str]:
    """Tie each live-data gap to the must-answer topic it is about; returns those topic ids.

    A live clause shares single words ("book", "port") with unrelated topics, and sentence
    embeddings proved unreliable on such short texts (a fines topic beat the shelf topic). So a
    topic is tied only if the clause contains at least half of the topic's own content terms.
    If none qualifies, the clause itself becomes a must-answer topic: the requester asked for it,
    so it gets probed and split like any other request, and no unrelated topic is blamed for it.
    ``topics`` is extended in place.
    """
    tied: set[str] = set()
    for gap in (g for g in structural if g.kind == "live_data" and g.evidence):
        clause_terms = content_terms(gap.evidence[0])
        scored = [(len(content_terms(t.topic) & clause_terms) / max(len(content_terms(t.topic)), 1), t) for t in topics]
        score, best = max(scored, key=lambda pair: pair[0], default=(0.0, None))
        if best is None or score < TIE_OVERLAP:
            text = _REQUEST_PREFIX.sub("", gap.evidence[0]).strip().rstrip(".")
            best = Topic(id=f"T{len(topics) + 1}", topic=text, priority="high")
            topics.append(best)
        gap.topic_id = best.id
        tied.add(best.id)
    return frozenset(tied)


def _live_gap_of(topic: Topic, structural: list[Gap]) -> Gap | None:
    return next((g for g in structural if g.kind == "live_data" and g.topic_id == topic.id), None)


def _apply_guard(
    req: Requirements, coverage: list[TopicCoverage], structural: list[Gap], chunks_by_id: dict[str, Chunk],
) -> None:
    by_id = {c.topic_id: c for c in coverage}
    kept: list[Topic] = []
    gaps: list[Gap] = []
    for topic in req.must_answer_topics:
        cov = by_id.get(topic.id)
        if cov is None:
            kept.append(topic)
            continue
        verdict = cov.effective_verdict
        topic.coverage = verdict
        topic.evidence = list(dict.fromkeys(cov.evidence + [e for a in cov.supported_aspects for e in a.evidence]))
        if verdict == "UNSUPPORTED":
            gaps.append(Gap(
                id=f"G-{topic.id}", kind="unsupported_topic", topic_id=topic.id,
                description=f"The brief asks for '{topic.topic}', but none of the {len(cov.probes)} probe questions could be answered from the documents.",
                recommended_action="refuse_with_redirect", redirect=DEFAULTS["gap_redirect"],
                evidence=[f"probe: {p.question} ({p.reason})" for p in cov.probes],
            ))
            continue
        if cov.supported_aspects:  # split: keep the supported part, refuse only the rest
            topic.supported_aspects = [a.aspect for a in cov.supported_aspects]
            for aspect in (a for a in cov.aspects if a not in cov.supported_aspects):
                live = _live_gap_of(topic, structural) if aspect.needs_live_values else None
                if live is not None:
                    live.aspect = aspect.aspect
                elif aspect.needs_live_values:
                    gaps.append(Gap(
                        id=f"G-{topic.id}-live", kind="live_data", topic_id=topic.id, aspect=aspect.aspect,
                        description=f"'{aspect.aspect}' needs a live value, but the documents are a static snapshot and no tool provides it.",
                        recommended_action="refuse_with_redirect", redirect=DEFAULTS["gap_redirect"]))
                else:
                    gaps.append(Gap(
                        id=f"G-{topic.id}-{len(gaps)}", kind="unsupported_topic", topic_id=topic.id, aspect=aspect.aspect,
                        description=f"Part of '{topic.topic}' is not covered by the documents: '{aspect.aspect}'.",
                        recommended_action="refuse_with_redirect", redirect=DEFAULTS["gap_redirect"]))
        topic.snapshot_as_of = find_snapshot_date([chunks_by_id[c] for c in topic.evidence if c in chunks_by_id])
        if cov.verdict == "PARTIAL" and not cov.supported_aspects:
            gaps.append(Gap(
                id=f"G-{topic.id}", kind="partial_coverage", topic_id=topic.id, recommended_action="warn",
                description=f"Only part of '{topic.topic}' is supported by the documents ({len(cov.evidence)} of {len(cov.probes)} probes verified).",
                evidence=cov.evidence))
        kept.append(topic)
    for gap in structural:  # a live gap on a topic whose recorded part IS supported says so
        topic = next((t for t in kept if t.id == gap.topic_id), None)
        if gap.kind == "live_data" and topic is not None and topic.supported_aspects:
            suffix = f" as of {topic.snapshot_as_of}" if topic.snapshot_as_of else ""
            gap.description += (f" The recorded version ({'; '.join(topic.supported_aspects)}) IS answerable from "
                                f"the documents{suffix}; only the live value is refused.")
        if gap.recommended_action == "refuse_with_redirect" and not gap.redirect:
            gap.redirect = DEFAULTS["gap_redirect"]
    req.must_answer_topics = kept
    req.gaps = gaps + structural


def _gap_question(req: Requirements, number: int) -> Question | None:
    actionable = [g for g in req.gaps if g.recommended_action != "warn"]
    if not actionable:
        return None
    items = "; ".join(
        (g.aspect or (g.description.split("'")[1] if g.kind == "unsupported_topic" and "'" in g.description else g.evidence[0] if g.evidence else g.description)).rstrip(". ")
        for g in actionable
    )
    text = DEFAULTS["gap_question"].format(items=items, options=DEFAULTS["gap_question_options"])
    return Question(id=f"Q{number}", field="gaps", text=text, kind="gap")


def run_intake_and_guard(
    brief: str, docs_dir: Path, out_dir: Path, config: ForgeConfig, embedder: Embedder,
    ask: AskFn | None = None, today: date | None = None,
) -> Requirements:
    """Run stages 1-3. ``ask=None`` means ``--non-interactive`` (self-answer, label ASSUMPTION)."""
    intake_dir, guard_dir = out_dir / "00_intake", out_dir / "01_guard"
    intake_dir.mkdir(parents=True, exist_ok=True)
    guard_dir.mkdir(parents=True, exist_ok=True)
    (intake_dir / "brief.txt").write_text(brief, encoding="utf-8")

    ingest = ingest_folder(docs_dir, config, embedder, out_dir=intake_dir)
    docs_text = iv.documents_summary(ingest.records) + "\n" + iv.topics_summary(ingest.topic_map)

    dropped: list[str] = []
    raw_draft = iv.draft_requirements(brief, ingest.records, ingest.topic_map, config)
    draft = iv.verify_confidence(raw_draft, brief, docs_text, log=dropped)
    (intake_dir / "draft_verification.json").write_text(
        json.dumps({"raw_draft": raw_draft.model_dump(), "dropped_unsupported": dropped}, indent=2), encoding="utf-8")
    budget = iv.QuestionBudget(config.limits.max_clarifying_questions)
    asked_fields = set(iv.unresolved_fields(draft)[: budget.interview_slots])
    draft, answers, assumptions, assumed_items = _interview(brief, draft, ingest, config, ask, budget)
    _fill_defaults(draft, brief, ingest, assumptions, asked_fields)
    req = _assemble(draft, assumptions)
    (intake_dir / "requirements.draft.yaml").write_text(_dump(req), encoding="utf-8")

    chunks = ingest.chunks
    index = HybridIndex.build(
        chunks, embedder, config.embedding_model,
        cache_dir=config.index_cache_dir, query_prefix=config.embedding_query_prefix,
    )
    reject_conflicting_assumptions(req, assumed_items, index, config)

    # Stage 3: groundability guard.
    documents = {r.file: " ".join(c.text for c in chunks if c.doc == r.file)[:3000] for r in ingest.records if r.status == "ok"}
    structural = detect_structural_gaps(brief, documents, today or date.today())
    live_topics = tie_live_gaps(req.must_answer_topics, structural)
    guard = run_guard([(t.id, t.topic) for t in req.must_answer_topics], req.purpose, index, config, live_topics)
    _apply_guard(req, guard.coverage, structural, index.by_id)

    gap_question = _gap_question(req, len(budget.asked) + 1)
    if gap_question is not None:
        budget.register(gap_question)
        reply = ask(gap_question) if ask else "refuse"
        by = "user" if ask else "ASSUMPTION"
        answers.append(Answer(question_id=gap_question.id, field="gaps", text=reply, answered_by=by))
        action = parse_gap_action(reply)
        for gap in req.gaps:
            if gap.kind == "unsupported_topic":
                gap.recommended_action = action
        if by == "ASSUMPTION":
            req.assumptions.append(Assumption(
                field="gaps", value="refuse_with_redirect for every gap",
                source=f"{NON_INTERACTIVE}: {gap_question.id} {gap_question.text}"))
    assert len(budget.asked) <= config.limits.max_clarifying_questions  # hard cap, enforced in code

    _write_outputs(req, budget.asked, answers, guard, intake_dir, guard_dir)
    return req


def _dump(req: Requirements) -> str:
    return yaml.safe_dump(req.model_dump(mode="json", by_alias=True), sort_keys=False, allow_unicode=True, width=100)


def _write_outputs(
    req: Requirements, questions: list[Question], answers: list[Answer], guard: GuardResult,
    intake_dir: Path, guard_dir: Path,
) -> None:
    (intake_dir / "questions.json").write_text(json.dumps([q.model_dump() for q in questions], indent=2), encoding="utf-8")
    (intake_dir / "answers.json").write_text(json.dumps([a.model_dump() for a in answers], indent=2), encoding="utf-8")
    (intake_dir / "requirements.yaml").write_text(_dump(req), encoding="utf-8")
    (intake_dir / "requirements.md").write_text(render_requirements(req), encoding="utf-8")
    coverage_json: list[dict[str, Any]] = [asdict(c) | {"effective_verdict": c.effective_verdict} for c in guard.coverage]
    payload = {"topics": coverage_json, "calibration": guard.calibration.to_dict(),
               "check_agreement": guard.agreement, "gaps": [g.model_dump() for g in req.gaps]}
    (guard_dir / "coverage.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (guard_dir / "gaps.md").write_text(
        render_gaps(req, coverage_json, guard.calibration.to_dict(), guard.agreement), encoding="utf-8")

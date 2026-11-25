"""Intake interviewer (spec §3): draft requirements, find what is unresolved, ask (<=5).

The model drafts every field with a confidence. The confidence claim is then *verified in
code*: a value only counts as resolved if it shares content terms with the source it claims
(the brief, or the document summaries). Anything unverified becomes ``unresolved`` and is
eligible for a question. Questions are only ever asked about unresolved fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from forgeai.config import ForgeConfig
from forgeai.ingest.models import DocRecord
from forgeai.ingest.topic_map import TopicMap
from forgeai.intake.schema import ListField, Question, RequirementsDraft, ScalarField
from forgeai.llm import call_structured
from forgeai.textmatch import shares_content_term
from forgeai.tools import REGISTRY

DEFAULTS = yaml.safe_load((Path(__file__).parent / "defaults.yaml").read_text(encoding="utf-8"))

# Spec §3.2: scope boundary > must-refuse topics > audience > tone > format.
QUESTION_PRIORITY = ("scope_out", "scope_in", "must_refuse", "audience", "tone")
FIELD_LABELS = {
    "scope_out": "topics the assistant must stay away from",
    "scope_in": "topics the assistant should cover",
    "must_refuse": "kinds of question the assistant must always refuse",
    "audience": "who will use the assistant",
    "tone": "the tone of the assistant's answers",
}
LIST_FIELDS = {"scope_in", "scope_out", "must_refuse"}


class QuestionLimitError(Exception):
    """Raised if code ever tries to exceed the hard cap on clarifying questions."""


class QuestionBudget:
    """Enforces the assignment's hard cap (<=5) with one slot reserved for the guard."""

    def __init__(self, total: int, reserved_for_guard: int = 1) -> None:
        self.total = total
        self.reserved = reserved_for_guard
        self.asked: list[Question] = []

    @property
    def interview_slots(self) -> int:
        return max(0, self.total - self.reserved)

    def register(self, question: Question) -> None:
        interview = sum(1 for q in self.asked if q.kind == "interview")
        if len(self.asked) >= self.total:
            raise QuestionLimitError(f"question cap of {self.total} exceeded")
        if question.kind == "interview" and interview >= self.interview_slots:
            raise QuestionLimitError(f"interview slots ({self.interview_slots}) exceeded")
        self.asked.append(question)


class QuestionText(BaseModel):
    text: str


class SelfAnswer(BaseModel):
    answer: str


class AnswerItems(BaseModel):
    values: list[str] = []


def documents_summary(records: list[DocRecord]) -> str:
    return "\n".join(f"- {r.file}: {r.title}. {r.summary}" for r in records if r.status == "ok") or "(none)"


def topics_summary(topic_map: TopicMap) -> str:
    return "\n".join(f"- {t.label}: {t.description}" for t in topic_map.topics) or "(none)"


def capability_tags() -> str:
    lines = ["- document_qa: answer from the documents (always needed)"]
    lines += [f"- {spec.capability}: provided by the {spec.name} tool" for spec in REGISTRY.values()]
    return "\n".join(lines)


def draft_requirements(
    brief: str, records: list[DocRecord], topic_map: TopicMap, config: ForgeConfig
) -> RequirementsDraft:
    model = config.agent_model
    return call_structured(
        "intake.draft", "intake/draft",
        {"brief": brief, "documents": documents_summary(records), "topics": topics_summary(topic_map),
         "capabilities": capability_tags()},
        RequirementsDraft, model.name, model.temperature, seed=model.seed,
        fallback=RequirementsDraft(),  # everything unresolved: the safe direction
    )


def _supported_by(value: str, brief: str, docs_text: str) -> Literal["resolved_from_brief", "resolved_from_docs"] | None:
    if shares_content_term(value, brief):
        return "resolved_from_brief"
    if shares_content_term(value, docs_text):
        return "resolved_from_docs"
    return None


def verify_confidence(
    draft: RequirementsDraft, brief: str, docs_text: str, log: list[str] | None = None
) -> RequirementsDraft:
    """Downgrade any resolved value that neither the brief nor the documents support.

    Support from EITHER source is accepted whatever source the model claimed: models often
    say "from the docs" for something the brief asked for, and dropping such a value would
    silently lose a requirement (this happened on the first try, see docs/DECISIONS.md). Anything
    removed is appended to ``log`` so it is visible in the run artifacts.
    """
    notes = log if log is not None else []
    checked = draft.model_copy(deep=True)
    for name in ("audience", "tone"):
        scalar: ScalarField = getattr(checked, name)
        if scalar.confidence == "unresolved" or not scalar.value.strip():
            scalar.confidence, scalar.value = "unresolved", ""
            continue
        source = _supported_by(scalar.value, brief, docs_text)
        if source is None:
            notes.append(f"{name}: dropped {scalar.value!r} (no support in brief or documents)")
            scalar.confidence, scalar.value = "unresolved", ""
        else:
            scalar.confidence = source
    for name in ("scope_in", "scope_out", "must_answer", "must_refuse"):
        listed: ListField = getattr(checked, name)
        keep = []
        for item in listed.values:
            if item.strip() and _supported_by(item, brief, docs_text):
                keep.append(item.strip())
            elif item.strip():
                notes.append(f"{name}: dropped {item!r} (no support in brief or documents)")
        listed.values = keep
        if not keep:
            listed.confidence = "unresolved"
    return checked


def unresolved_fields(draft: RequirementsDraft) -> list[str]:
    """Question-worthy unresolved fields, highest priority first."""
    result = []
    for name in QUESTION_PRIORITY:
        field = getattr(draft, name)
        if field.confidence == "unresolved":
            result.append(name)
    return result


def build_question(field: str, number: int, brief: str, config: ForgeConfig) -> Question:
    model = config.agent_model
    generic = DEFAULTS["question_templates"][field]
    text = call_structured(
        "intake.question", "intake/question",
        {"brief": brief, "field_label": FIELD_LABELS[field], "generic": generic},
        QuestionText, model.name, model.temperature, seed=model.seed,
        fallback=QuestionText(text=generic),
    ).text.strip()
    return Question(id=f"Q{number}", field=field, text=text or generic)


def self_answer(question: Question, brief: str, documents: str, config: ForgeConfig) -> str:
    """Non-interactive mode: answer our own question with the most defensible default."""
    model = config.agent_model
    fallback_value = DEFAULTS["fields"].get(question.field, "")
    fallback = SelfAnswer(answer=fallback_value if isinstance(fallback_value, str) else "")
    return call_structured(
        "intake.self_answer", "intake/self_answer",
        {"brief": brief, "documents": documents, "question": question.text},
        SelfAnswer, model.name, model.temperature, seed=model.seed, fallback=fallback,
    ).answer.strip()


def answer_to_items(question: Question, answer: str, config: ForgeConfig) -> list[str]:
    model = config.agent_model
    items = call_structured(
        "intake.apply_answer", "intake/apply_answer",
        {"field_label": FIELD_LABELS[question.field], "question": question.text, "answer": answer},
        AnswerItems, model.name, 0.0, seed=model.seed,
        fallback=AnswerItems(values=[answer] if answer.strip() else []),
    )
    return clean_items(items.values)


_SENTENCE_LIKE = re.compile(r"\b(should|must|shall|will|is|are)\b|^the assistant\b", re.IGNORECASE)
MAX_ITEM_WORDS = 12


def clean_items(values: list[str]) -> list[str]:
    """Keep topic-like items; drop sentences and instructions the model slipped in.

    A scope or refusal list is data that later steps compile into an agent prompt, so a
    stray sentence such as "the assistant should not avoid X" would invert a rule.
    """
    seen: dict[str, None] = {}
    for value in values:
        item = value.strip().rstrip(".")
        if item and len(item.split()) <= MAX_ITEM_WORDS and not _SENTENCE_LIKE.search(item):
            seen.setdefault(item, None)
    return list(seen)

"""Computed and lookup cases: the expected value is a deterministic function of anchored facts.

The model only PROPOSES a plan: a question, the facts it uses (each bound to a table cell, a
number in a quoted passage, or a number written in the question) and a formula. Code then
verifies every binding against the documents, checks the stated conditions, and computes the
answer with the safe calculator. The model never supplies the answer, and nothing here knows
what the numbers mean, so price x quantity x (1 - discount tier) is just one formula among many.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from forgeai.evals.corpus import EvalCorpus
from forgeai.evals.matching import numbers_in
from forgeai.ingest.spans import SeedSpan
from forgeai.textmatch import is_verbatim
from forgeai.tools.base import ToolError
from forgeai.tools.calculator import evaluate, evaluate_condition, format_number

IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class KV(BaseModel):
    column: str
    value: str


class PlanVar(BaseModel):
    name: str
    source: str  # csv | prose | question
    doc: str = ""
    row_key: list[KV] = Field(default_factory=list)
    column: str = ""
    span_text: str = ""
    number_text: str = ""


class ComputePlan(BaseModel):
    question: str = ""
    variables: list[PlanVar] = Field(default_factory=list)
    formula: str = ""
    conditions: list[str] = Field(default_factory=list)


@dataclass
class VerifiedPlan:
    question: str
    answer: float
    answer_text: str
    values: dict[str, float]
    spans: list[SeedSpan]
    docs: list[str]
    uses_table: bool


class PlanRejected(Exception):
    """The plan failed a deterministic check; the message is the rejection reason."""


def _number(text: str) -> float:
    found = numbers_in(text)
    if not found:
        raise PlanRejected(f"no number in {text!r}")
    return found[0]


def _bind(var: PlanVar, corpus: EvalCorpus, question: str, spans: list[SeedSpan], docs: list[str]) -> float:
    if var.source == "question":
        value = _number(var.number_text)
        if not any(abs(value - n) < 1e-9 for n in numbers_in(question)):
            raise PlanRejected(f"question variable {var.name} ({var.number_text}) is not a number in the question")
        return value
    if var.source == "prose":
        if var.doc not in corpus.docs or var.doc in corpus.tables:
            raise PlanRejected(f"prose variable {var.name}: {var.doc!r} is not a prose document")
        span = corpus.span_of(var.doc, var.span_text)
        if span is None:
            raise PlanRejected(f"prose variable {var.name}: span_text is not a verbatim quote of {var.doc}")
        if not is_verbatim(var.number_text, span.span_text):
            raise PlanRejected(f"prose variable {var.name}: {var.number_text!r} is not in its span")
        if not any(abs(_number(var.number_text) - n) < 1e-9 for n in numbers_in(span.span_text)):
            raise PlanRejected(f"prose variable {var.name}: {var.number_text!r} is not a number in its span "
                               "(it only appears inside another token such as an identifier)")
        spans.append(span)
        docs.append(var.doc)
        return _number(var.number_text)
    if var.source == "csv":
        table = corpus.tables.get(var.doc)
        if table is None or not var.row_key or var.column not in table.columns:
            raise PlanRejected(f"csv variable {var.name}: unknown table or column")
        matches = [i for i, row in enumerate(table.rows)
                   if all(row.get(kv.column, "").strip() == kv.value.strip() for kv in var.row_key)]
        if len(matches) != 1:
            raise PlanRejected(f"csv variable {var.name}: row_key matches {len(matches)} rows, need exactly 1")
        spans.append(table.row_spans[matches[0]])
        docs.append(var.doc)
        return _number(table.rows[matches[0]][var.column])
    raise PlanRejected(f"variable {var.name}: unknown source {var.source!r}")


def verify_plan(plan: ComputePlan, corpus: EvalCorpus, cross_source: bool = False) -> VerifiedPlan:
    """Check every binding and condition; compute the answer. Raises ``PlanRejected``."""
    names = [v.name for v in plan.variables]
    if len(set(names)) != len(names) or not all(IDENTIFIER.match(n) for n in names):
        raise PlanRejected("variable names must be unique lower-case identifiers")
    if sum(v.source in ("csv", "prose") for v in plan.variables) < 2:
        raise PlanRejected("a computed case needs at least two facts taken from the documents")
    if not plan.formula.strip() or not set(re.findall(r"[a-z_][a-z0-9_]*", plan.formula)) <= set(names) | {
            "round", "min", "max", "abs", "ceil", "floor", "sum", "sqrt"}:
        raise PlanRejected("formula is empty or uses names that are not variables")
    used = set(re.findall(r"[a-z_][a-z0-9_]*", " ".join([plan.formula, *plan.conditions])))
    if unused := [n for n in names if n not in used]:
        raise PlanRejected(f"variables not used in the formula: {unused}")
    if cross_source and not ({"csv", "prose"} <= {v.source for v in plan.variables}):
        raise PlanRejected("the case must combine a table value with a rule from the prose")
    spans: list[SeedSpan] = []
    docs: list[str] = []
    values = {v.name: _bind(v, corpus, plan.question, spans, docs) for v in plan.variables}
    try:
        for condition in plan.conditions:
            if not evaluate_condition(condition, values):
                raise PlanRejected(f"condition does not hold for these values: {condition}")
        answer = float(evaluate(plan.formula, values))
    except ToolError as exc:
        raise PlanRejected(f"formula or condition cannot be evaluated: {exc}") from exc
    answer = round(answer, 2)
    return VerifiedPlan(plan.question, answer, format_number(answer), values, spans,
                        list(dict.fromkeys(docs)), any(v.source == "csv" for v in plan.variables))

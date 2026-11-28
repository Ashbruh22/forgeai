"""Layer 1 of the judge (spec §6.2): deterministic checks, authoritative wherever they apply.

Everything here is code: no model is consulted. Layer 2 (``layer2.py``) only runs where this layer cannot decide
(a key fact that fails matching, whether the claims of an answer are supported, whether a refusal says more than
a redirect).
"""

from __future__ import annotations

import re
from datetime import date

from rapidfuzz import fuzz

from forgeai.evals.matching import NEGATIONS, dates_in, numbers_in
from forgeai.evals.models import EvalCase, KeyFact
from forgeai.evals.templates import resolve_relative_fact
from forgeai.ingest.spans import SeedSpan
from forgeai.judge.models import CaseRun, FactCheck, Layer1
from forgeai.textmatch import normalize

EXACT_TOLERANCE = (1e-9, 0.0)
TOOL_TOLERANCE = (0.01, 0.001)  # absolute, relative: a computed value may be rounded by the agent
_LABEL = re.compile(r"^\s*([A-Za-z_][\w ]*?)\s*:\s*(.+)$")
_WORDS = re.compile(r"[a-z0-9]+")


def _close(found: float, wanted: float, tolerance: tuple[float, float]) -> bool:
    return abs(found - wanted) <= max(tolerance[0], tolerance[1] * abs(wanted))


def _negations(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower())) & NEGATIONS


def _phrase_candidates(fact: str) -> list[str]:
    """The fact as written, and (for a table cell written "column_name: value") the value alone."""
    found = [fact]
    match = _LABEL.match(fact)
    if match and "_" in match.group(1):
        found.append(match.group(2))
    return found


def match_fact(fact: KeyFact, text: str, *, tolerance: tuple[float, float], today: date, fuzzy_min: int) -> FactCheck:
    """Is ``fact`` expressed in ``text``? Numbers, dates and currency by value; phrases by substring then rapidfuzz."""
    base = {"fact": fact.text, "kind": fact.kind}
    if fact.kind == "number":
        wanted = numbers_in(fact.text)
        have = numbers_in(text)
        ok = bool(wanted) and all(any(_close(h, w, tolerance) for h in have) for w in wanted)
        return FactCheck(**base, matched=ok, method="number" if ok else "none")
    if fact.kind in ("date", "snapshot_date", "relative_date"):
        wanted_text = resolve_relative_fact(fact.text, today) if fact.kind == "relative_date" else fact.text
        wanted = dates_in(wanted_text)
        ok = bool(wanted) and wanted[0] in dates_in(text)
        return FactCheck(**base, matched=ok, method="date" if ok else "none",
                         note=f"expected {wanted[0].isoformat()}" if wanted else "unparseable date")
    haystack = normalize(text).lower()
    for candidate in _phrase_candidates(fact.text):
        needle = normalize(candidate).lower()
        if not needle:
            continue
        nums = numbers_in(candidate)
        if nums and not all(any(_close(h, n, EXACT_TOLERANCE) for h in numbers_in(text)) for n in nums):
            continue  # a phrase that carries a quantity is wrong if the quantity is
        if _negations(candidate) - _negations(text):
            continue  # "not permitted" must not match "permitted"
        if needle in haystack:
            return FactCheck(**base, matched=True, method="exact")
        if fuzz.token_set_ratio(needle, haystack) >= fuzzy_min:
            return FactCheck(**base, matched=True, method="fuzzy", note=f"token-set score >= {fuzzy_min}")
    return FactCheck(**base, matched=False, method="none")


def _overlaps(chunk_doc: str, start: int | None, end: int | None, span: SeedSpan) -> bool:
    return chunk_doc == span.doc and start is not None and end is not None and start < span.char_end and span.char_start < end


def expected_spans(case: EvalCase) -> list[SeedSpan]:
    return [*case.seed_spans, *[i.seed_span for i in case.interpretations]]


def check_layer1(case: EvalCase, run: CaseRun, *, today: date, fuzzy_min: int = 85) -> Layer1:
    answer = run.answer
    citation_issues = [i for i in run.issues if i.startswith("citation:")]
    contract_issues = [i for i in run.issues if not i.startswith("citation:")]
    if case.type == "ambiguous":
        status_ok = answer.status in ("needs_clarification", "answered")
    else:
        status_ok = answer.status == case.expected_status
    has_tools = bool(case.expected_tools) or case.type == "tool_required"
    tolerance = TOOL_TOLERANCE if has_tools else EXACT_TOLERANCE

    facts = [match_fact(f, answer.answer, tolerance=tolerance, today=today, fuzzy_min=fuzzy_min) for f in case.key_facts]
    snapshot = [f for f, k in zip(facts, case.key_facts, strict=True) if k.kind == "snapshot_date"]

    interp_facts = [[match_fact(k, answer.answer, tolerance=EXACT_TOLERANCE, today=today, fuzzy_min=fuzzy_min)
                     for k in i.key_facts] for i in case.interpretations]
    covered = sum(1 for group in interp_facts if group and all(f.matched for f in group))

    spans = expected_spans(case)
    wants_evidence = case.expected_status != "refused" and bool(spans)
    hit, rank, docs_hit = None, None, []
    if wants_evidence:
        for position, chunk in enumerate(run.retrieved, 1):
            overlapping = [s for s in spans if _overlaps(chunk.doc, chunk.char_start, chunk.char_end, s)]
            if overlapping:
                rank = rank or position
                docs_hit.extend(s.doc for s in overlapping)
        hit = rank is not None

    source_ok, missing = None, []
    if case.type in ("answerable", "multidoc", "tool_required") and case.seed_spans:
        if answer.status == "answered":
            for doc in sorted({s.doc for s in case.seed_spans}):
                cited_here = any(_overlaps(c.doc, c.char_start, c.char_end, s)
                                 for c in run.cited for s in case.seed_spans if s.doc == doc)
                if not cited_here:
                    missing.append(doc)
        else:
            missing = sorted({s.doc for s in case.seed_spans})  # not answered, so nothing was cited
        source_ok = not missing

    called = [r.tool for r in answer.tool_calls]
    tool_ok, tool_notes = None, []
    if case.expected_tools:
        tool_notes = [f"expected tool {t!r} was not called" for t in case.expected_tools if t not in called]
        tool_notes += [f"tool {r.tool!r} returned an error: {r.result[:80]}" for r in answer.tool_calls
                       if r.tool in case.expected_tools and r.result.startswith("ERROR")]
        values = [f for f, k in zip(facts, case.key_facts, strict=True) if k.kind in ("number", "relative_date", "date")]
        tool_notes += [f"final value {f.fact!r} not in the answer within tolerance" for f in values if not f.matched]
        tool_ok = not tool_notes

    figures = answer.status == "refused" and bool(set(numbers_in(answer.answer)) - set(numbers_in(case.question)))
    return Layer1(
        contract_valid=not contract_issues, contract_issues=contract_issues, status_ok=status_ok, status=answer.status,
        citations_total=len(answer.citations), citations_valid=max(len(answer.citations) - len(citation_issues), 0),
        citation_issues=citation_issues, source_ok=source_ok, sources_missing=missing, facts=facts,
        tools_called=called, tool_ok=tool_ok, tool_notes=tool_notes, retrieval_hit=hit, retrieval_rank=rank,
        retrieval_docs_hit=sorted(set(docs_hit)), interpretation_facts=interp_facts,
        interpretations_covered=covered if case.interpretations else None,
        interpretations_total=len(case.interpretations) if case.interpretations else None,
        snapshot_date_ok=all(f.matched for f in snapshot) if snapshot else None, refusal_has_figures=figures,
    )

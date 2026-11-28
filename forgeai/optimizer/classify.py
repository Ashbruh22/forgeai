"""Failure classification (spec §6.5): a deterministic decision tree, code only.

Every failing dev case gets exactly ONE of the assignment's four categories and, for prompt defects, the prompt
section a patch would rewrite. A language model is never consulted here: it only drafts the text of a patch once the
category and the section are decided (``critic.py``).

::

    contract invalid ......................................... PROMPT DEFECT (output_contract)
    expected refused, got answered:
        any unsupported claim ................................ HALLUCINATION
        else ................................................. PROMPT DEFECT (scope)
    expected answered, got refused:
        gold chunk in top-k .................................. WRONG REFUSAL
        else ................................................. RETRIEVAL MISS
    expected answered, got answered but failed:
        gold chunk not in top-k .............................. RETRIEVAL MISS
        unsupported claims present ........................... HALLUCINATION
        supported but facts missing / wrong citation ......... PROMPT DEFECT
    ambiguous, answered one reading only ..................... PROMPT DEFECT (ambiguity_policy)

A failure whose deciding verdict is a layer-2 verdict of a type the judge cannot be trusted on (the canaries put it
below 90%) carries ``low_confidence``; the optimizer prefers to target the others.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from forgeai.evals.models import EvalCase
from forgeai.judge.models import CaseScore

RETRIEVAL_MISS, PROMPT_DEFECT, HALLUCINATION, WRONG_REFUSAL = "retrieval_miss", "prompt_defect", "hallucination", "wrong_refusal"
CATEGORIES = (HALLUCINATION, WRONG_REFUSAL, RETRIEVAL_MISS, PROMPT_DEFECT)  # also the tie-break order (order of harm)


@dataclass(frozen=True)
class Failure:
    case_id: str
    category: str
    section: str  # the prompt section a rewrite would target ("" for a retrieval miss)
    reason: str
    low_confidence: bool = False


def _retrieval_incomplete(case: EvalCase, score: CaseScore) -> bool:
    """Was a gold chunk missing from the retrieved set? A multi-document case needs every expected document hit."""
    l1 = score.layer1
    if l1.retrieval_hit is None:
        return False
    if not l1.retrieval_hit:
        return True
    if case.type == "multidoc":
        wanted = {s.doc for s in case.seed_spans}
        return not wanted <= set(l1.retrieval_docs_hit)
    return False


def classify(case: EvalCase, score: CaseScore) -> Failure:
    """The single category of one failing case (call only for a case with ``score.passed`` false)."""
    l1, l2, cid = score.layer1, score.layer2, case.id
    if not l1.contract_valid:
        return Failure(cid, PROMPT_DEFECT, "output_contract", "the answer does not satisfy the output contract: " + "; ".join(l1.contract_issues))

    if case.expected_status == "refused":
        if l1.status == "refused":  # refused, but with claims beyond a redirect
            return Failure(cid, HALLUCINATION, "grounding_rules", "the refusal states factual claims", score.low_confidence)
        if l2.unsupported:
            return Failure(cid, HALLUCINATION, "grounding_rules", "answered a question that should be refused, with unsupported claims",
                           score.low_confidence)
        return Failure(cid, PROMPT_DEFECT, "scope", "answered (or asked to clarify) where the question is out of scope and should be refused")

    if case.type == "ambiguous":
        if l1.status == "answered":
            return Failure(cid, PROMPT_DEFECT, "ambiguity_policy", f"answered covering {l1.interpretations_covered} of {l1.interpretations_total} readings")
        if l1.status == "refused":
            if _retrieval_incomplete(case, score):
                return Failure(cid, RETRIEVAL_MISS, "", "refused an ambiguous question and the passages were not retrieved")
            return Failure(cid, WRONG_REFUSAL, "refusal_policy", "refused an ambiguous question whose passages were retrieved")

    if l1.status == "refused":
        if _retrieval_incomplete(case, score):
            return Failure(cid, RETRIEVAL_MISS, "", "refused and no gold chunk was retrieved")
        return Failure(cid, WRONG_REFUSAL, "refusal_policy", "refused although a gold chunk was retrieved")
    if l1.status == "needs_clarification":
        return Failure(cid, PROMPT_DEFECT, "ambiguity_policy", "asked for clarification where the question has one answer")

    # expected answered, got answered
    if _retrieval_incomplete(case, score):
        return Failure(cid, RETRIEVAL_MISS, "", "no gold chunk (or not every expected document) was among the retrieved chunks")
    if l2.unsupported:
        return Failure(cid, HALLUCINATION, "grounding_rules", "; ".join(c.missing for c in l2.claims if not c.supported)[:200], score.low_confidence)
    if l1.citation_issues or l1.source_ok is False or l1.grounded_ok is False:
        return Failure(cid, PROMPT_DEFECT, "citation_rules", "wrong or invalid citation: the expected source was not cited")
    if l1.tool_ok is False:
        return Failure(cid, PROMPT_DEFECT, "tool_policy", "; ".join(l1.tool_notes))
    kinds = sorted({f.kind for f in l1.facts if not f.matched})
    return Failure(cid, PROMPT_DEFECT, "scope", f"the answer lacks required facts (kinds: {', '.join(kinds) or 'phrase'}) that were supported by the evidence")


def classify_all(cases: dict[str, EvalCase], scores: dict[str, CaseScore]) -> list[Failure]:
    return [classify(cases[cid], s) for cid, s in scores.items() if not s.passed and cid in cases]


def pick_category(failures: list[Failure]) -> tuple[str | None, dict[str, int]]:
    """The category with the most failures (high-confidence ones first); ties go hallucination > wrong refusal >
    retrieval miss > prompt defect. Returns the category and the counts it was chosen from."""
    if not failures:
        return None, {}
    solid = [f for f in failures if not f.low_confidence]
    pool = solid or failures  # only if every failure is low confidence is one of those targeted
    counts = Counter(f.category for f in pool)
    best = max(counts.values())
    return next(c for c in CATEGORIES if counts.get(c) == best), dict(counts)

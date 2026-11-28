"""Layer 2 of the judge (spec §6.2): a language model, only where layer 1 cannot decide.

* Claim support: the answer is split into atomic claims, repeated claims are dropped, and each claim is rated against
  the RETRIEVED CONTEXT and the FULL results of the tools the agent called, and nothing else. A claim with a number or
  a date is decided by code (``claims.numeric_verdict``); any other claim by the judge model, whose "supported"
  counts only with a quote copied from the evidence and verified as a substring, and which an optional NLI check
  can turn into "contradicted". Only an answer-bearing claim that is unsupported or contradicted fails a case; one
  unsupported claim that bears no answer is logged as low confidence.
* Fact equivalence: a key fact that failed deterministic matching may be upgraded to a pass only if the judge quotes
  the span of the answer that states it; the quote is verified as a substring of the answer, must carry the fact's
  numbers, and must agree with it on negation.
* Refusals: a refusal that carries figures is checked for factual claims beyond a redirect.

The judge sees the question, the answer, the claim, the fact and the context. It never receives a model name, an
iteration number, a case id or a candidate id (``tests/test_judge.py`` checks every prompt it renders).
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field

from forgeai.config import ModelSpec
from forgeai.evals.matching import NEGATIONS, numbers_in
from forgeai.evals.models import EvalCase
from rapidfuzz import fuzz

from forgeai.judge.claims import (
    answer_bearing, best_span, build_spans, dedupe_claims, negation_mismatch, numeric_verdict, sentence_negation_checks,
    answer_number_checks,
)
from forgeai.judge.models import CaseRun, ClaimVerdict, FactCheck, Layer1, Layer2
from forgeai.judge.nli import NliChecker
from forgeai.llm import call_structured
from forgeai.textmatch import is_verbatim, normalize

MAX_CLAIMS = 8
CONTEXT_CHARS = 9000
LONG_REFUSAL_WORDS = 40
_WORDS = re.compile(r"[a-z0-9]+")


class ClaimList(BaseModel):
    claims: list[str] = Field(default_factory=list)


class SupportVerdict(BaseModel):
    reason: str = ""
    supported: bool = False
    missing: str = ""
    quote: str = ""


class Equivalence(BaseModel):
    reason: str = ""
    expressed: bool = False
    quote: str = ""


def format_context(run: CaseRun) -> str:
    """The evidence the judge may use: FULL tool results first (never cut), then whole retrieved chunks until the budget
    is used up. A tool result is never truncated: a calculation's value and inputs are the evidence for a computed claim."""
    parts = [f"Tool result: {r.tool} {r.args} gave {r.result}" for r in run.answer.tool_calls]
    used = sum(len(p) for p in parts)
    for chunk in run.retrieved:
        if used + len(chunk.text) > CONTEXT_CHARS and parts:
            break
        parts.append(chunk.text)
        used += len(chunk.text)
    return (chr(10) * 2).join(parts) or "No context was retrieved."


def _negations(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower())) & NEGATIONS


def near_verbatim(quote: str, context: str, minimum: int = 93) -> bool:
    """A quote the judge copied with a small slip (an ellipsis, a dropped word) still counts if 93%+ of it is in the evidence."""
    return len(quote.strip()) >= 15 and fuzz.partial_ratio(normalize(quote).lower(), normalize(context).lower()) >= minimum


def judge_claims(case: EvalCase, run: CaseRun, judge: ModelSpec, today: date, *, nli: NliChecker | None = None,
                 nli_min: float = 0.7, fuzzy_min: int = 85) -> tuple[list[ClaimVerdict], int]:
    """Decompose the answer, drop repeated claims, then rate each claim: numbers and dates by code against the
    evidence (context and full tool results), every other claim by the judge model. Returns the verdicts and the
    number of duplicate claims dropped."""
    answer_text = run.answer.answer
    listed = call_structured("judge.decompose", "judge/decompose_claims", {"question": case.question, "answer": answer_text},
                             ClaimList, judge.name, judge.temperature, seed=judge.seed, fallback=ClaimList())
    raw_claims = [c for c in listed.claims if c.strip()]
    claims = dedupe_claims(raw_claims)
    spans = build_spans(run)
    context = format_context(run)
    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        bearing = answer_bearing(claim, case, today, fuzzy_min)
        decided = numeric_verdict(claim, spans, case.question)
        if decided is not None:
            verdicts.append(decided)
            continue
        raw = call_structured("judge.support", "judge/claim_check", {"context": context, "claim": claim},
                              SupportVerdict, judge.name, judge.temperature, seed=judge.seed,
                              fallback=SupportVerdict(reason="the judge returned no valid verdict"))
        verified = bool(raw.quote.strip()) and (is_verbatim(raw.quote, context) or near_verbatim(raw.quote, context))
        verdict, missing = ("supported", "") if raw.supported and verified else ("unsupported", raw.missing or raw.reason)
        if raw.supported and not verified:
            missing = "the judge quoted text that is not in the evidence"
        probs: dict[str, float] = {}
        if verdict == "supported" and negation_mismatch(claim, raw.quote):
            verdict, missing = "contradicted", f"the claim and the quoted evidence {raw.quote[:100]!r} disagree on negation"
        if verdict == "supported" and nli is not None:
            span, _ = best_span(claim, spans)
            premise = span.text if span else raw.quote
            probs = nli.probabilities(premise, claim)
            if probs.get("contradiction", 0.0) >= nli_min:
                verdict, missing = "contradicted", f"an NLI check finds the claim contradicted by {premise[:100]!r}"
        verdicts.append(ClaimVerdict(
            claim=claim, supported=verdict == "supported", quote=raw.quote, quote_verified=verified, verdict=verdict,
            missing=missing, kind="text", answer_bearing=bearing, nli=probs,
            reason=f"claim {claim!r}: {raw.reason}" + ("" if verdict == "supported" else f" Missing: {missing}")))
    seen = {c.claim for c in verdicts}
    extra = sentence_negation_checks(answer_text, spans) + answer_number_checks(answer_text, case.question, spans)
    verdicts += [v for v in extra if v.claim not in seen]
    return verdicts, len(raw_claims) - len(claims)


def try_upgrade(question: str, check: FactCheck, answer_text: str, judge: ModelSpec) -> tuple[FactCheck, dict]:
    """Ask whether the answer states a fact layer 1 could not match; accept only a verified quote that fits the fact."""
    raw = call_structured("judge.equivalence", "judge/fact_equivalence",
                          {"question": question, "fact": check.fact, "answer": answer_text}, Equivalence,
                          judge.name, judge.temperature, seed=judge.seed, fallback=Equivalence())
    log = {"fact": check.fact, "judge_says_expressed": raw.expressed, "quote": raw.quote, "reason": raw.reason}
    if not raw.expressed:
        return check, {**log, "upgraded": False, "why_not": "the judge did not find the fact"}
    if not raw.quote.strip() or not is_verbatim(raw.quote, answer_text):
        return check, {**log, "upgraded": False, "why_not": "no verbatim quote from the answer"}
    wanted_numbers = numbers_in(check.fact)
    if any(all(abs(n - q) > 1e-9 for q in numbers_in(raw.quote)) for n in wanted_numbers):
        return check, {**log, "upgraded": False, "why_not": "the quote does not carry the fact's numbers"}
    if _negations(check.fact) != _negations(raw.quote):
        return check, {**log, "upgraded": False, "why_not": "the quote and the fact disagree on negation"}
    upgraded = check.model_copy(update={"matched": True, "method": "judge_upgrade", "upgraded_by_judge": True, "quote": raw.quote})
    return upgraded, {**log, "upgraded": True}


def run_layer2(case: EvalCase, run: CaseRun, layer1: Layer1, judge: ModelSpec, *, today: date, nli: NliChecker | None = None,
               nli_min: float = 0.7, fuzzy_min: int = 85) -> tuple[Layer2, Layer1]:
    """Layer 2 for one case. Returns the verdicts and layer 1 with any verified fact upgrades applied."""
    answer = run.answer
    out = Layer2(ran=False)
    facts = list(layer1.facts)
    interp = [list(group) for group in layer1.interpretation_facts]

    if answer.status == "answered" and answer.answer.strip():
        out.ran = True
        out.claims, out.duplicates_dropped = judge_claims(case, run, judge, today, nli=nli, nli_min=nli_min, fuzzy_min=fuzzy_min)
        failing = [c for c in out.claims if not c.supported and c.answer_bearing]
        minor = [c for c in out.claims if not c.supported and not c.answer_bearing]
        out.unsupported = len(failing) + (len(minor) if len(minor) > 1 else 0)  # a lone minor claim is logged, not failed
        out.minor_unsupported = [c.claim for c in minor]
        if len(minor) == 1 and not failing:
            out.notes.append(f"low confidence: one claim that bears no answer is unsupported ({minor[0].claim!r}): {minor[0].missing}")
        if not out.claims:
            out.notes.append("the judge found no factual claim in the answer")
        for position, check in enumerate(facts):
            if not check.matched and check.kind == "phrase":
                facts[position], log = try_upgrade(case.question, check, answer.answer, judge)
                out.upgrades.append(log)
        if case.type == "ambiguous":
            for group in interp:
                for position, check in enumerate(group):
                    if not check.matched and check.kind == "phrase":
                        group[position], log = try_upgrade(case.question, check, answer.answer, judge)
                        out.upgrades.append(log)
    elif answer.status == "refused" and (layer1.refusal_has_figures or len(answer.answer.split()) > LONG_REFUSAL_WORDS):
        out.ran = True
        listed = call_structured("judge.decompose", "judge/decompose_claims", {"question": case.question, "answer": answer.answer},
                                 ClaimList, judge.name, judge.temperature, seed=judge.seed, fallback=ClaimList())
        out.refusal_claims = [c for c in listed.claims if c.strip()][:MAX_CLAIMS]

    updated = layer1.model_copy(update={"facts": facts, "interpretation_facts": interp})
    if case.type == "ambiguous" and updated.interpretations_total is not None:
        covered = sum(1 for group in interp if group and all(f.matched for f in group))
        updated = updated.model_copy(update={"interpretations_covered": covered})
    return out, updated

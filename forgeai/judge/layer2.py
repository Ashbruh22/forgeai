"""Layer 2 of the judge (spec §6.2): a language model, only where layer 1 cannot decide.

* Claim support: the answer is split into atomic claims and each is rated against the RETRIEVED CONTEXT (and the
  results of the tools the agent called) and nothing else. "Supported" counts only with a quote copied from that
  context, verified as a substring here.
* Fact equivalence: a key fact that failed deterministic matching may be upgraded to a pass only if the judge quotes
  the span of the answer that states it; the quote is verified as a substring of the answer, must carry the fact's
  numbers, and must agree with it on negation.
* Refusals: a refusal that carries figures is checked for factual claims beyond a redirect.

The judge sees the question, the answer, the claim, the fact and the context. It never receives a model name, an
iteration number, a case id or a candidate id (``tests/test_judge.py`` checks every prompt it renders).
"""

from __future__ import annotations

import re
from pydantic import BaseModel, Field

from forgeai.config import ModelSpec
from forgeai.evals.matching import NEGATIONS, numbers_in
from forgeai.evals.models import EvalCase
from forgeai.judge.models import CaseRun, ClaimVerdict, FactCheck, Layer1, Layer2
from forgeai.llm import call_structured
from forgeai.textmatch import is_verbatim

MAX_CLAIMS = 8
CONTEXT_CHARS = 6000
LONG_REFUSAL_WORDS = 40
_WORDS = re.compile(r"[a-z0-9]+")


class ClaimList(BaseModel):
    claims: list[str] = Field(default_factory=list)


class SupportVerdict(BaseModel):
    reason: str = ""
    supported: bool = False
    quote: str = ""


class Equivalence(BaseModel):
    reason: str = ""
    expressed: bool = False
    quote: str = ""


def format_context(run: CaseRun) -> str:
    """The retrieved chunks and the tool results the agent had: the only evidence a claim may rest on."""
    parts = [chunk.text for chunk in run.retrieved]
    parts += [f"Tool result: {r.tool} {r.args} gave {r.result}" for r in run.answer.tool_calls]
    context = "\n\n".join(parts) or "No context was retrieved."
    return context[:CONTEXT_CHARS]


def _negations(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower())) & NEGATIONS


def judge_claims(question: str, answer_text: str, context: str, judge: ModelSpec) -> list[ClaimVerdict]:
    """Decompose the answer and rate every claim against ``context`` (batched per answer, one call per claim)."""
    listed = call_structured("judge.decompose", "judge/decompose_claims", {"question": question, "answer": answer_text},
                             ClaimList, judge.name, judge.temperature, seed=judge.seed, fallback=ClaimList())
    verdicts: list[ClaimVerdict] = []
    for claim in [c.strip() for c in listed.claims if c.strip()][:MAX_CLAIMS]:
        raw = call_structured("judge.support", "judge/claim_support", {"context": context, "claim": claim},
                              SupportVerdict, judge.name, judge.temperature, seed=judge.seed,
                              fallback=SupportVerdict(reason="the judge returned no valid verdict"))
        verified = bool(raw.quote.strip()) and is_verbatim(raw.quote, context)
        verdicts.append(ClaimVerdict(claim=claim, supported=raw.supported and verified, quote=raw.quote,
                                     reason=raw.reason if verified or not raw.supported else
                                     f"{raw.reason} [rejected: the quote is not a verbatim span of the context]",
                                     quote_verified=verified))
    return verdicts


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


def run_layer2(case: EvalCase, run: CaseRun, layer1: Layer1, judge: ModelSpec) -> tuple[Layer2, Layer1]:
    """Layer 2 for one case. Returns the verdicts and layer 1 with any verified fact upgrades applied."""
    answer = run.answer
    out = Layer2(ran=False)
    facts = list(layer1.facts)
    interp = [list(group) for group in layer1.interpretation_facts]

    if answer.status == "answered" and answer.answer.strip():
        out.ran = True
        out.claims = judge_claims(case.question, answer.answer, format_context(run), judge)
        out.unsupported = sum(1 for c in out.claims if not c.supported)
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

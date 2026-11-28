"""Deterministic handling of the claims of an answer (P6.1), so the language model only judges what code cannot.

* ``dedupe_claims``: a decomposition that repeats one fact four times must not count it four times.
* ``numeric_verdict``: a claim with a number, a date or a currency amount is checked against the retrieved context AND
  the full tool results by code: every number must occur in a span that is also about the claim's subject, and a
  negation the span lacks (or the span has and the claim lacks) makes it contradicted.
* ``answer_bearing``: a claim is answer-bearing if it carries a number or date, a key fact of the case, or an entity
  (a code, a quoted title, a capitalised name) of the question. Only those claims can fail a case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from rapidfuzz import fuzz

from forgeai.evals.matching import dates_in, numbers_in
from forgeai.evals.models import EvalCase
from forgeai.judge.layer1 import match_fact
from forgeai.judge.models import CaseRun, ClaimVerdict
from forgeai.judge.normalize import canonical_terms, coverage
from forgeai.textmatch import normalize

MAX_CLAIMS = 8
DUPLICATE_SIMILARITY = 88
SUBJECT_COVERAGE_MIN = 0.34  # a number counts as support only in a span that shares this much of the claim's subject
TOOL_RESULT_CHARS = 6000
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORDS = re.compile(r"[a-z0-9]+")
_CODE = re.compile(r"(?<![\w-])[A-Za-z]*-?\d[\w-]*")
_QUOTED = re.compile(r"['\"‘“]([^'\"’”]{3,60})['\"’”]")
_CAPITALISED = re.compile(r"(?<!^)(?<![.?!] )\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})*\b")


@dataclass(frozen=True)
class Span:
    text: str
    source: str  # "chunk" or "tool"


def _plain(text: str) -> str:
    return " ".join(_WORDS.findall(text.lower()))


def build_spans(run: CaseRun) -> list[Span]:
    """Sentences and table cells of every retrieved chunk, and every line (and the whole text) of every tool result.
    (What the question gives is handled by ``given_by_question``, not as a span: it is not evidence.)"""
    spans: list[Span] = []
    for chunk in run.retrieved:
        for sentence in _SENTENCE.split(chunk.text):
            if sentence.strip():
                spans.append(Span(sentence.strip(), "chunk"))
                if " | " in sentence:
                    spans.extend(Span(cell.strip(), "chunk") for cell in sentence.split(" | ") if cell.strip())
    for call in run.answer.tool_calls:
        text = f"{call.tool} {call.args} -> {call.result}"
        spans.append(Span(text[:TOOL_RESULT_CHARS], "tool"))
        spans.extend(Span(line.strip(), "tool") for line in call.result.splitlines() if line.strip())
    return spans


def dedupe_claims(claims: list[str], cap: int = MAX_CLAIMS) -> list[str]:
    """Distinct claims in order: exact repeats and near repeats (token-sort similarity) are dropped, a claim that is
    contained in a longer one is replaced by the longer one."""
    kept: list[str] = []
    for claim in (c.strip() for c in claims):
        plain = _plain(claim)
        if not plain:
            continue
        for position, other in enumerate(kept):
            other_plain = _plain(other)
            if fuzz.token_sort_ratio(plain, other_plain) >= DUPLICATE_SIMILARITY or plain in other_plain:
                break
            if other_plain in plain:
                kept[position] = claim
                break
        else:
            kept.append(claim)
    return kept[:cap]


_NEGATION = re.compile(r"(?<![\w])(?:not|never|cannot|without|none|nor|no longer)(?![\w])|n't(?![\w])", re.IGNORECASE)
NEGATION_WINDOW = 5  # words either side of a negation that count as "what it negates"


def has_negation(text: str) -> bool:
    return bool(_NEGATION.search(normalize(text)))


_CLAUSE = re.compile(r";|:|\s+and\s+(?=(?:is|are|was|were|may|must|can|cannot|will|shall|does|do|should|has|have)(?![\w]))|,\s+(?:and|but|or|however)\s+|\s+(?:but|however|except|unless)\s+", re.IGNORECASE)


def negation_mismatch(claim: str, evidence: str) -> bool:
    """The claim says "not" where the evidence never does, or the clause of the evidence that the claim is about
    negates what the claim states without a negation. Different negation words ("cannot", "may not", "never") count
    as the same; a negation in another clause of a long sentence is not a contradiction."""
    if has_negation(claim):
        return not has_negation(evidence)
    if not has_negation(evidence):
        return False
    clauses = [c for c in _CLAUSE.split(evidence) if c.strip()]
    negated = [c for c in clauses if has_negation(c)]
    plain = canonical_terms(" ".join(c for c in clauses if not has_negation(c)))
    subject = canonical_terms(claim)
    # the claim is contradicted when it shares a word with a negated clause that no un-negated clause carries
    return any(subject & canonical_terms(c) - plain for c in negated)


def same_number(found: float, wanted: float) -> bool:
    """Whole numbers (codes, counts, days) must be equal; a decimal may differ by rounding (0.01 or 0.1%)."""
    if float(found).is_integer() and float(wanted).is_integer():
        return found == wanted
    return abs(found - wanted) <= max(0.01, 0.001 * abs(wanted))


def units_agree(claim: str, span_text: str) -> bool:
    """A percentage in the claim must be a percentage in the span, an amount of money an amount of money: the same
    digits with another unit ("15000%" for "USD 15,000") are not the same quantity. An unknown unit is not a conflict."""
    from forgeai.evals.units import MONEY, RATE, quantities_in

    have = quantities_in(span_text)
    for q in quantities_in(claim):
        if q.unit in (MONEY, RATE):
            same_value = [h for h in have if abs(h.value - q.value) < 1e-9 and h.unit is not None]
            if same_value and not any(h.unit == q.unit for h in same_value):
                return False
    return True


def best_span(claim: str, spans: list[Span], need_numbers: list[float] | None = None) -> tuple[Span | None, float]:
    """The span that shares the most of the claim's subject (and, if given, contains all of ``need_numbers``). Among
    spans nearly as good as the best, one whose negation agrees with the claim's is preferred."""
    scored: list[tuple[float, Span]] = []
    for span in spans:
        if need_numbers is not None:
            have = value_numbers(span.text)
            if not all(any(same_number(h, n) for h in have) for n in need_numbers) or not units_agree(claim, span.text):
                continue
        scored.append((coverage(claim, span.text), span))
    if not scored:
        return None, 0.0
    scored.sort(key=lambda pair: -pair[0])
    top = scored[0][0]
    for score, span in scored:
        if score < min(SUBJECT_COVERAGE_MIN, top):
            break
        if not negation_mismatch(claim, span.text):
            return span, score
    return scored[0][1], top


_DIGITS = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


_ISO = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_CODE_TOKEN = re.compile(r"(?<![\w-])[A-Za-z]{1,6}[-_]?\d[\w-]*")


def strip_codes(text: str) -> str:
    """The text without identifier tokens ("MT-8424", "S03"): their digits are part of a name, not a quantity."""
    return _CODE_TOKEN.sub(" ", text)


def claim_codes(text: str) -> set[str]:
    """Identifier tokens of ``text`` (lowercase)."""
    return {m.group(0).lower().strip("-_") for m in _CODE_TOKEN.finditer(text)}


def value_numbers(text: str) -> list[float]:
    """Numbers of ``text`` that are quantities: digits outside identifiers, and spelled-out numbers."""
    return numbers_in(strip_codes(text))


def claim_numbers(text: str) -> list[float]:
    """The quantities written with digits ("one document" is not a quantity; spelled-out numbers go to the judge model;
    the digits of a part code are not a quantity either: the code is checked as a whole)."""
    out = []
    for raw in _DIGITS.findall(strip_codes(_ISO.sub(" ", text))):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def has_figure(claim: str) -> bool:
    return bool(claim_numbers(claim) or claim_codes(claim) or dates_in(claim))


def codes_in(text: str) -> set[str]:
    return {m.group(0).lower() for m in _CODE.finditer(text) if len(m.group(0)) >= 3 and re.search(r"[a-z]", m.group(0), re.IGNORECASE)}


_CELL = re.compile(r"^([A-Za-z_][\w ]{0,30}?)\s*:\s*(.+)$")


def label_names_claim(claim: str, span_text: str) -> bool:
    """A table cell "label: value" is about the claim when its label is the claim's subject, written out ("lead time
    days") or abbreviated by initials ("moq" for "minimum order quantity")."""
    cell = _CELL.match(span_text.strip())
    if not cell:
        return False
    label = cell.group(1).strip()
    if coverage(label.replace("_", " "), claim) >= 0.5:
        return True
    letters = re.sub(r"[^a-z]", "", label.lower())
    words = _WORDS.findall(claim.lower())
    return 2 <= len(letters) <= 6 and any("".join(w[0] for w in words[i:i + len(letters)]) == letters for i in range(len(words)))


def _window(text: str, number: float, size: int = 1) -> str:
    """The words around the first occurrence of ``number`` in ``text``."""
    words = normalize(text).split()
    for position, word in enumerate(words):
        if any(abs(n - number) < 1e-9 for n in claim_numbers(word)):
            return " ".join(words[max(0, position - size): position + size + 1])
    return ""


def given_by_question(claim: str, question: str, number: float) -> bool:
    """The user's own number, restated in the same relation ("15 units" for "15 units"): it needs no evidence. The same
    digits in another relation ("USD 15" for "15 units") are not given."""
    if not question or not any(abs(n - number) < 1e-9 for n in claim_numbers(question)):
        return False
    asked, stated = _window(question, number), _window(claim, number)
    return bool(asked and stated and canonical_terms(asked) & canonical_terms(stated))


def numeric_verdict(claim: str, spans: list[Span], question: str = "") -> ClaimVerdict | None:
    """Support for a claim that has a number or date, decided by code; None if the claim has none. Numbers the question
    gives (in the same relation) are supported by the question; the rest must be found in the evidence."""
    numbers, dates, codes = claim_numbers(claim), dates_in(claim), claim_codes(claim)
    if not numbers and not dates and not codes:
        return None
    base = {"claim": claim, "kind": "numeric", "answer_bearing": True}
    text_of_evidence = " ".join(s.text.lower() for s in spans)
    for code in sorted(codes):  # an identifier must occur as a whole in the evidence, or be the one the question names
        if code not in text_of_evidence and code not in claim_codes(question):
            return ClaimVerdict(**base, supported=False, verdict="unsupported", missing=f"the identifier {code} is in neither the context nor a tool result",
                                reason=f"claim {claim!r}: the identifier {code} is nowhere in the evidence")
    given = [n for n in numbers if given_by_question(claim, question, n)]
    numbers = [n for n in numbers if n not in given]
    if not numbers and not dates and not given:
        return None  # only an identifier that checked out: the rest of the claim is for the judge model
    if not numbers and not dates:
        return ClaimVerdict(**base, supported=True, verdict="supported", quote=question[:300], quote_verified=True,
                            reason="the numbers are the ones the question gives")
    for wanted_date in dates:
        if not any(wanted_date in dates_in(s.text) for s in spans):
            return ClaimVerdict(**base, supported=False, verdict="unsupported", missing=f"the date {wanted_date.isoformat()} is in neither the context nor a tool result",
                                reason=f"claim {claim!r}: the date {wanted_date.isoformat()} is nowhere in the evidence")
    span, score = best_span(claim, spans, numbers)
    if span is None:
        return ClaimVerdict(**base, supported=False, verdict="unsupported",
                            missing=f"no span of the context or tool results contains {', '.join(f'{n:g}' for n in numbers)}",
                            reason=f"claim {claim!r}: the number(s) {', '.join(f'{n:g}' for n in numbers)} do not occur together in any evidence")
    names_row = bool(codes_in(claim)) and codes_in(claim) <= {c.lower() for c in _CODE.findall(span.text)}
    if span.source != "tool" and score < SUBJECT_COVERAGE_MIN and not names_row and not label_names_claim(claim, span.text):  # a row that names the claim's code is about it
        return ClaimVerdict(**base, supported=False, verdict="unsupported", quote=span.text[:300], quote_verified=True,
                            missing="the number occurs only in text about something else",
                            reason=f"claim {claim!r}: the number occurs only in {span.text[:90]!r}, which shares too little of the claim's subject")
    if negation_mismatch(claim, span.text):
        return ClaimVerdict(**base, supported=False, verdict="contradicted", quote=span.text[:300], quote_verified=True,
                            missing=f"the evidence {'negates' if has_negation(span.text) else 'does not negate'} what the claim says",
                            reason=f"claim {claim!r} disagrees with {span.text[:120]!r} on negation")
    return ClaimVerdict(**base, supported=True, verdict="supported", quote=span.text[:300], quote_verified=True,
                        reason=f"found in {'a tool result' if span.source == 'tool' else 'the context'}")


def question_entities(question: str) -> set[str]:
    """Codes, quoted titles and capitalised names of the question (lowercase)."""
    codes = {m.group(0).lower() for m in _CODE.finditer(question) if len(m.group(0)) >= 2}
    names = {m.group(1).lower() for m in _QUOTED.finditer(question)}
    names |= {m.group(0).lower() for m in _CAPITALISED.finditer(question)}
    return codes | {n for n in names if len(n) >= 3}


def answer_bearing(claim: str, case: EvalCase, today: date, fuzzy_min: int = 85) -> bool:
    """Does the claim carry a number or date, a key fact of the case, or an entity of the question?"""
    if has_figure(claim):
        return True
    facts = [*case.key_facts, *[k for i in case.interpretations for k in i.key_facts]]
    if any(match_fact(k, claim, tolerance=(1e-9, 0.0), today=today, fuzzy_min=fuzzy_min).matched for k in facts):
        return True
    text = normalize(claim).lower()
    return any(entity in text for entity in question_entities(case.question))


SENTENCE_COVERAGE_MIN = 0.5


def sentence_negation_checks(answer: str, spans: list[Span]) -> list[ClaimVerdict]:
    """A decomposition can drop a "not" when it rewrites a sentence. So each sentence of the answer is also compared with
    the span of the evidence it is about: if the sentence negates what that span states (or the span negates what the
    sentence states), the sentence is a contradicted claim whatever the decomposition said."""
    found: list[ClaimVerdict] = []
    for sentence in (s.strip() for s in _SENTENCE.split(answer)):
        if len(sentence.split()) < 4 or not canonical_terms(sentence):
            continue
        span, score = best_span(sentence, [s for s in spans if s.source == "chunk"])
        if span is None or score < SENTENCE_COVERAGE_MIN or not negation_mismatch(sentence, span.text):
            continue
        found.append(ClaimVerdict(
            claim=sentence, supported=False, verdict="contradicted", kind="negation", answer_bearing=True, quote=span.text[:300],
            quote_verified=True, missing=f"the evidence {span.text[:100]!r} "
            + ("does not negate this" if has_negation(sentence) else "negates this"),
            reason=f"sentence {sentence!r} and the evidence it is about disagree on negation"))
    return found


def answer_number_checks(answer: str, question: str, spans: list[Span]) -> list[ClaimVerdict]:
    """Every figure written with digits in the answer must occur in the evidence (or in the question, which the user
    wrote). A decomposition can leave a swapped or invented number out of its claims; this check cannot."""
    evidence = [n for s in spans for n in value_numbers(s.text)]
    evidence_text = " ".join(s.text.lower() for s in spans)
    found: list[ClaimVerdict] = []
    for sentence in (s.strip() for s in _SENTENCE.split(answer)):
        missing = [n for n in claim_numbers(sentence) if not given_by_question(sentence, question, n)
                   and not any(same_number(e, n) for e in evidence)]
        missing += [c for c in sorted(claim_codes(sentence)) if c not in evidence_text and c not in claim_codes(question)]
        if missing and len(sentence.split()) >= 2:
            shown = ", ".join(c if isinstance(c, str) else f"{c:g}" for c in missing)
            found.append(ClaimVerdict(claim=sentence, supported=False, verdict="unsupported", kind="numeric", answer_bearing=True,
                                      missing=f"the number(s) {shown} appear in neither the context, the tool results nor the question",
                                      reason=f"sentence {sentence!r} states {shown}, which no evidence contains"))
    return found

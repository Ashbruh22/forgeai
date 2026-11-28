"""Anti-overfitting guard for prompt patches (P7).

The critic may write the TEXT of one named prompt section, and that text must be a general rule, never an answer.
A text is rejected if it contains

* a run of words copied from a dev question (n-gram check),
* a key-fact value of a dev case (a number, a date, a phrase),
* a code or part number of a dev case, or a name or title written in a dev question that the section did not
  already contain.

The check reads dev cases only. A rejected patch is logged with its reasons (``rejected_patches.json``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from forgeai.evals.matching import numbers_in
from forgeai.evals.models import EvalCase
from forgeai.textmatch import normalize

NGRAM = 5
_WORDS = re.compile(r"[a-z0-9]+")
_CODE = re.compile(r"(?<![\w-])[A-Za-z]{1,6}[-_]?\d[\w-]*")
_NAME = re.compile(r"(?<![.?!] )(?<!^)\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})*\b")
_QUOTED = re.compile(r"['\"‘“]([^'\"’”]{3,60})['\"’”]")
MIN_PHRASE_FACT = 4


def _grams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = _WORDS.findall(text.lower())
    return {tuple(words[i: i + n]) for i in range(len(words) - n + 1)}


def case_specific_entities(cases: Iterable[EvalCase]) -> set[str]:
    """Codes, quoted titles and capitalised names written in the dev questions, and codes in the seed spans (lowercase)."""
    found: set[str] = set()
    for case in cases:
        found |= {m.group(0).lower().strip("-_") for m in _CODE.finditer(case.question)}
        found |= {m.group(1).lower() for m in _QUOTED.finditer(case.question)}
        found |= {m.group(0).lower() for m in _NAME.finditer(case.question)}
        for span in case.seed_spans:
            found |= {m.group(0).lower().strip("-_") for m in _CODE.finditer(span.span_text)}
    return {e for e in found if len(e) >= 2}


def violations(text: str, cases: list[EvalCase], original: str = "", allowed_vocabulary: str = "") -> list[str]:
    """Why ``text`` is not a general rule (empty list: it is)."""
    problems: list[str] = []
    body = normalize(text).lower()
    original_grams = _grams(original) | _grams(allowed_vocabulary)
    for case in cases:
        copied = (_grams(text) & _grams(case.question)) - original_grams
        if copied:
            problems.append(f"copies {NGRAM} consecutive words of a dev question: {' '.join(sorted(copied)[0])!r}")
            break
    words = set(_WORDS.findall(body))
    for case in cases:
        for fact in case.key_facts:
            fact_text = normalize(fact.text).lower()
            if fact.kind in ("number", "snapshot_date", "date"):
                if fact.kind != "number" and fact_text in body:
                    problems.append(f"contains the date {fact.text!r} of a dev case")
                elif fact.kind == "number" and any(re.search(rf"(?<![\w.]){re.escape(f'{n:g}')}(?![\w])", body) for n in numbers_in(fact.text)):
                    problems.append(f"contains the number {fact.text!r}, a key fact of a dev case")
            elif len(fact_text) >= MIN_PHRASE_FACT and fact_text not in normalize(original).lower() and re.search(rf"(?<!\w){re.escape(fact_text)}(?!\w)", body):
                problems.append(f"contains the value {fact.text!r}, a key fact of a dev case")
    known = set(_WORDS.findall(normalize(original + " " + allowed_vocabulary).lower()))  # words the section or the requirements already use
    for entity in sorted(case_specific_entities(cases)):
        if " " in entity or "-" in entity or any(c.isdigit() for c in entity):
            hit = entity in body
        else:
            hit = entity in words
        if hit and not all(part in known for part in _WORDS.findall(entity)):
            problems.append(f"names {entity!r}, which is specific to a dev case")
    return list(dict.fromkeys(problems))


def refusal_items_kept(original: str, new: str) -> list[str]:
    """For the refusal policy: every bullet of the original must still be there, word for word."""
    kept = normalize(new)
    return [f"drops the refusal item {line.strip()!r}" for line in original.splitlines() if line.strip().startswith("- ") and normalize(line) not in kept]


def size_ok(original: str, new: str) -> list[str]:
    limit = max(int(len(original) * 1.5), len(original) + 800)
    return [] if 20 <= len(new) <= limit else [f"length {len(new)} is outside 20..{limit} characters"]
